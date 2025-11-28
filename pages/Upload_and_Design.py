# pages/1_Upload_and_Design.py
import io
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Upload & Design", layout="wide")


# ---------- UTILITIES ----------

def clear_upload_state():
    keys = [
        # protein
        "protein_bytes",
        "protein_name",
        # peptide
        "peptide_bytes",
        "peptide_name",
        # shared metadata
        "metadata_bytes",
        "metadata_name",
        # processed bundles
        "protein_upload",
        "peptide_upload",
    ]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]


@st.cache_data(show_spinner="Loading file...")
def _read_table(bytes_data: bytes) -> pd.DataFrame:
    text = bytes_data.decode("utf-8", errors="replace")
    if text.startswith("\ufeff"):
        text = text[1:]
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")


def _load_and_map(
    data_bytes_key: str,
    label: str,
    id_label: str = "ID / group column",
) -> dict:
    """
    Generic loader + mapper for either protein or peptide table.

    Expects:
      - st.session_state[data_bytes_key] : bytes
      - st.session_state["metadata_bytes"] : bytes

    Returns a dict:
      {
        "df_wide": DataFrame,
        "design": DataFrame,
        "id_col": str or None,
        "quant_cols": list[str],
        "level": "protein" or "peptide",
      }
    """
    if data_bytes_key not in st.session_state or "metadata_bytes" not in st.session_state:
        st.warning(f"Upload {label} data and metadata first.")
        st.stop()

    df_raw = _read_table(st.session_state[data_bytes_key])
    df_meta = _read_table(st.session_state["metadata_bytes"])

    st.markdown(f"### {label} data overview")
    st.write(f"Table: {df_raw.shape[0]:,} rows × {df_raw.shape[1]:,} columns")
    st.write(f"Metadata: {df_meta.shape[0]:,} rows × {df_meta.shape[1]:,} columns")

    # --- METADATA MAPPING ---
    st.markdown(f"#### {label}: map metadata columns")

    meta_cols = df_meta.columns.tolist()
    with st.expander("Map sample / condition / replicate", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            sample_col = st.selectbox(
                "Metadata column: sample/run ID",
                options=meta_cols,
                index=meta_cols.index("Run Label") if "Run Label" in meta_cols else 0,
                key=f"{label}_sample_col",
            )
        with col2:
            cond_col = st.selectbox(
                "Metadata column: condition",
                options=meta_cols,
                index=meta_cols.index("Condition") if "Condition" in meta_cols else 0,
                key=f"{label}_cond_col",
            )
        with col3:
            repl_col = st.selectbox(
                "Metadata column: replicate",
                options=meta_cols,
                index=meta_cols.index("Replicate") if "Replicate" in meta_cols else 0,
                key=f"{label}_repl_col",
            )

    design = (
        df_meta[[sample_col, cond_col, repl_col]]
        .rename(
            columns={
                sample_col: "sample_id",
                cond_col: "condition",
                repl_col: "replicate",
            }
        )
        .astype({"sample_id": str, "condition": str})
    )

    # --- REPLICATE BALANCE CHECK ---
    counts = design.groupby("condition")["replicate"].nunique()
    balanced = counts.nunique() == 1

    st.markdown(f"#### {label}: design check")
    st.write("Replicates per condition:")
    st.dataframe(counts.to_frame("n_replicates"))

    if not balanced:
        st.warning(
            "Conditions do not have equal numbers of replicates. "
            "Downstream stats may need unbalanced-design handling."
        )

    # --- MAP SAMPLE IDS TO DATA COLUMNS ---
    st.markdown(f"#### {label}: map sample IDs to intensity columns")
    prot_cols = df_raw.columns.tolist()

    rename_dict: dict[str, str] = {}
    used_cols: set[str] = set()

    for _, row in design.iterrows():
        sid = str(row["sample_id"]).strip()
        if not sid:
            continue
        matches = [c for c in prot_cols if sid in str(c)]
        if len(matches) == 0:
            st.warning(f"Sample ID not found in headers: `{sid}`")
            continue
        if len(matches) > 1:
            st.error(f"Multiple columns contain sample ID `{sid}`: {matches}")
            st.stop()
        col = matches[0]
        if col in used_cols:
            st.error(f"Column `{col}` matched more than once: {col}")
            st.stop()
        rename_dict[col] = sid
        used_cols.add(col)

    if not rename_dict:
        st.error("No intensity columns were matched using metadata sample IDs.")
        st.stop()

    df_wide = df_raw.rename(columns=rename_dict).copy()

    # --- AUTO-DETECT QUANTITATIVE COLUMNS ---
    numeric_cols = df_wide.select_dtypes(include="number").columns.tolist()
    mapped_sample_ids = list(rename_dict.values())
    quant_cols_default = [c for c in numeric_cols if c in mapped_sample_ids]

    with st.expander(f"{label}: select quantitative (intensity) columns", expanded=True):
        quant_cols = st.multiselect(
            "Quantitative columns",
            options=numeric_cols,
            default=quant_cols_default or numeric_cols,
            help="All numeric columns are shown; by default those that match sample IDs are selected.",
            key=f"{label}_quant_cols",
        )

    if not quant_cols:
        st.error("Select at least one quantitative column.")
        st.stop()

    # --- ID MAPPING ---
    st.markdown(f"#### {label}: map identifiers")
    non_numeric = df_wide.columns.difference(numeric_cols).tolist()
    if non_numeric:
        id_col = st.selectbox(
            id_label,
            options=non_numeric,
            index=0,
            key=f"{label}_id_col",
        )
    else:
        st.warning("No non-numeric columns available to use as identifiers.")
        id_col = None

    # --- SUMMARY ---
    st.markdown(f"#### {label}: summary")
    st.write(f"Quantitative columns: {len(quant_cols)}")
    st.write(f"Conditions: {design['condition'].nunique()}")
    st.write(f"Total samples: {design['sample_id'].nunique()}")

    st.dataframe(
        design.sort_values(["condition", "replicate", "sample_id"]).reset_index(drop=True),
        use_container_width=True,
    )

    level = "protein" if "protein" in label.lower() else "peptide"

    return {
        "df_wide": df_wide,
        "design": design,
        "id_col": id_col,
        "quant_cols": quant_cols,
        "level": level,
    }


# ---------- MAIN PAGE ----------

def main():
    # Styling header
    st.markdown(
        """
        <style>
            .header {
                background: linear-gradient(90deg,#E71316,#A6192E);
                padding: 20px 40px;
                color: white;
                margin: -80px -80px 40px;
            }
            .header h1,.header p {margin:0;}
            .stButton>button {background:#E71316 !important; color:white !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="header">'
        '<h1>DIA Proteomics Pipeline</h1>'
        '<p>Upload & Experimental Design</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Data level selector
    st.subheader("Data level")
    data_level = st.radio(
        "What data do you want to upload?",
        options=["Protein only", "Peptide only", "Protein + Peptide"],
        index=0,
        horizontal=True,
    )

    st.divider()

    # --- METADATA UPLOAD (shared) ---
    st.subheader("1. Metadata upload (shared)")

    if "metadata_bytes" not in st.session_state:
        uploaded_meta = st.file_uploader(
            "Upload metadata file (e.g. metadata.tsv/csv)",
            type=["tsv", "csv", "txt"],
            key="metadata_uploader",
        )
        if uploaded_meta:
            st.session_state["metadata_bytes"] = uploaded_meta.getvalue()
            st.session_state["metadata_name"] = uploaded_meta.name
            st.rerun()
    else:
        st.success(f"Metadata: **{st.session_state['metadata_name']}**")
        if st.button("Clear metadata"):
            for k in ("metadata_bytes", "metadata_name"):
                st.session_state.pop(k, None)
            st.rerun()

    if "metadata_bytes" not in st.session_state:
        st.info("Upload metadata to proceed with data uploads.")
        if st.button("Restart / Clear All"):
            clear_upload_state()
            st.rerun()
        return

    st.divider()

    # --- PROTEIN UPLOAD (optional, depending on selection) ---
    if data_level in ("Protein only", "Protein + Peptide"):
        st.subheader("2. Protein-level data upload")

        col_p1, col_p2 = st.columns([3, 1])
        with col_p1:
            if "protein_bytes" not in st.session_state:
                uploaded_prot = st.file_uploader(
                    "Upload wide-format protein table",
                    type=["csv", "tsv", "txt"],
                    key="protein_uploader",
                )
                if uploaded_prot:
                    st.session_state["protein_bytes"] = uploaded_prot.getvalue()
                    st.session_state["protein_name"] = uploaded_prot.name
                    st.rerun()
            else:
                st.success(f"Protein table: **{st.session_state['protein_name']}**")
        with col_p2:
            if "protein_bytes" in st.session_state:
                if st.button("Clear protein", key="clear_protein"):
                    for k in ("protein_bytes", "protein_name", "protein_upload"):
                        st.session_state.pop(k, None)
                    st.rerun()

        if "protein_bytes" in st.session_state:
            st.markdown("---")
            st.markdown("#### Protein mapping")
            protein_result = _load_and_map(
                data_bytes_key="protein_bytes",
                label="Protein",
                id_label="Protein ID / group column",
            )
            st.session_state["protein_upload"] = protein_result

    st.divider()

    # --- PEPTIDE UPLOAD (optional) ---
    if data_level in ("Peptide only", "Protein + Peptide"):
        st.subheader("3. Peptide-level data upload")

        col_pep1, col_pep2 = st.columns([3, 1])
        with col_pep1:
            if "peptide_bytes" not in st.session_state:
                uploaded_pep = st.file_uploader(
                    "Upload wide-format peptide/precursor table",
                    type=["csv", "tsv", "txt"],
                    key="peptide_uploader",
                )
                if uploaded_pep:
                    st.session_state["peptide_bytes"] = uploaded_pep.getvalue()
                    st.session_state["peptide_name"] = uploaded_pep.name
                    st.rerun()
            else:
                st.success(f"Peptide table: **{st.session_state['peptide_name']}**")
        with col_pep2:
            if "peptide_bytes" in st.session_state:
                if st.button("Clear peptide", key="clear_peptide"):
                    for k in ("peptide_bytes", "peptide_name", "peptide_upload"):
                        st.session_state.pop(k, None)
                    st.rerun()

        if "peptide_bytes" in st.session_state:
            st.markdown("---")
            st.markdown("#### Peptide mapping")
            peptide_result = _load_and_map(
                data_bytes_key="peptide_bytes",
                label="Peptide",
                id_label="Peptide / precursor ID column",
            )
            st.session_state["peptide_upload"] = peptide_result

    st.divider()

    # --- READY STATUS + NAVIGATION HINT ---
    st.subheader("4. Status & next steps")

    protein_ready = "protein_upload" in st.session_state
    peptide_ready = "peptide_upload" in st.session_state

    st.write(f"Protein data ready: **{protein_ready}**")
    st.write(f"Peptide data ready: **{peptide_ready}**")

    if protein_ready or peptide_ready:
        st.success(
            "Upload and design mapping complete. "
            "Open the next page in the sidebar (e.g. QC / Analysis) to continue."
        )
    else:
        st.info("Finish at least one data level (protein or peptide) to proceed.")

    # Restart overlay + button
    st.markdown(
        """
        <style>
            .restart-fixed {
                position: fixed;
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%);
                z-index: 999;
                background: #E71316;
                color: white;
                padding: 12px 24px;
                border-radius: 10px;
                font-weight: bold;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                text-align: center;
            }
        </style>
        <div class="restart-fixed">
            🔄 Restart Entire Upload & Design
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("RESTART", key="restart_all", help="Clear all uploaded data and mappings"):
        st.cache_data.clear()
        st.cache_resource.clear()
        clear_upload_state()
        st.rerun()


if __name__ == "__main__":
    main()
