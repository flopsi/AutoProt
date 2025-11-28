import io
import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner="Loading files...")
def _read_table(bytes_data: bytes) -> pd.DataFrame:
    text = bytes_data.decode("utf-8", errors="replace")
    if text.startswith("\ufeff"):
        text = text[1:]
    # Let pandas auto-detect separator (csv/tsv/txt) using Python engine
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")


def load_and_map():
    """
    1) Load protein + metadata from st.session_state.<protein_bytes, metadata_bytes>.
    2) Ask user to map key metadata fields (sample/run id, condition, replicate).
    3) Auto-detect numeric vs non-numeric columns for quantification.
    4) Build and return:
       - df_wide: wide protein table (rows=proteins, cols=samples)
       - design: sample design table (sample_id, condition, replicate, ...)
       - id_cols: chosen ID/name columns
       - quant_cols: numeric columns used as intensities
    """
    if "protein_bytes" not in st.session_state or "metadata_bytes" not in st.session_state:
        st.warning("Upload protein and metadata files first.")
        st.stop()

    df_raw = _read_table(st.session_state.protein_bytes)
    df_meta = _read_table(st.session_state.metadata_bytes)

    st.subheader("1. Column overview")
    st.write(f"Protein table: {df_raw.shape[0]:,} rows × {df_raw.shape[1]:,} columns")
    st.write(f"Metadata table: {df_meta.shape[0]:,} rows × {df_meta.shape[1]:,} columns")

    # --- USER MAPPING OF METADATA FIELDS ---
    st.subheader("2. Map metadata columns")

    meta_cols = df_meta.columns.tolist()
    with st.expander("Map sample / condition / replicate", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            sample_col = st.selectbox(
                "Metadata column: sample/run ID",
                options=meta_cols,
                index=meta_cols.index("Run Label") if "Run Label" in meta_cols else 0,
            )
        with col2:
            cond_col = st.selectbox(
                "Metadata column: condition",
                options=meta_cols,
                index=meta_cols.index("Condition") if "Condition" in meta_cols else 0,
            )
        with col3:
            repl_col = st.selectbox(
                "Metadata column: replicate",
                options=meta_cols,
                index=meta_cols.index("Replicate") if "Replicate" in meta_cols else 0,
            )

    # Sample design table
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

    # --- VALIDATE REPLICATE BALANCE ---
    counts = design.groupby("condition")["replicate"].nunique()
    balanced = counts.nunique() == 1
    st.subheader("3. Design check")
    st.write("Replicates per condition:")
    st.dataframe(counts.to_frame("n_replicates"))
    if not balanced:
        st.warning(
            "Conditions do not have equal numbers of replicates. "
            "Downstream stats may need unbalanced design handling."
        )

    # --- MAP METADATA SAMPLE IDS TO PROTEIN COLUMNS ---
    st.subheader("4. Map sample IDs to protein intensity columns")
    prot_cols = df_raw.columns.tolist()

    # Heuristic: find columns whose header contains sample_id (substring match)
    rename_dict = {}
    used_cols = set()
    for _, row in design.iterrows():
        sid = str(row["sample_id"]).strip()
        if not sid:
            continue
        matches = [c for c in prot_cols if sid in str(c)]
        if len(matches) == 0:
            st.warning(f"Sample ID not found in protein headers: `{sid}`")
            continue
        if len(matches) > 1:
            st.error(f"Multiple columns contain sample ID `{sid}`: {matches}")
            st.stop()
        col = matches[0]
        if col in used_cols:
            st.error(f"Protein column `{col}` matched more than once!")
            st.stop()
        rename_dict[col] = sid
        used_cols.add(col)

    if not rename_dict:
        st.error("No intensity columns were matched using metadata sample IDs.")
        st.stop()

    df_wide = df_raw.rename(columns=rename_dict).copy()

    # --- AUTO-DETECT QUANTITATIVE COLUMNS FROM df_wide ---
    numeric_cols = df_wide.select_dtypes(include="number").columns.tolist()  # all numeric candidate cols [web:18]
    mapped_sample_ids = list(rename_dict.values())
    quant_cols_default = [c for c in numeric_cols if c in mapped_sample_ids]

    with st.expander("Select quantitative (intensity) columns", expanded=True):
        quant_cols = st.multiselect(
            "Quantitative columns",
            options=numeric_cols,
            default=quant_cols_default or numeric_cols,
            help="All numeric columns are shown; by default those that match sample IDs are selected.",
        )

    if not quant_cols:
        st.error("Select at least one quantitative column.")
        st.stop()

    # --- MAP PROTEIN IDENTIFIERS ---
    st.subheader("5. Map protein identifiers")
    non_numeric = df_wide.columns.difference(numeric_cols).tolist()
    id_col = None
    if non_numeric:
        id_col = st.selectbox(
            "Protein ID / group column",
            options=non_numeric,
            index=0,
        )
    else:
        st.warning("No non-numeric columns available to use as protein IDs.")
        id_col = None

    # --- SUMMARY ---
    st.subheader("6. Upload summary")
    st.write(f"Quantitative columns: {len(quant_cols)}")
    st.write(f"Conditions: {design['condition'].nunique()}")
    st.write(f"Total samples: {design['sample_id'].nunique()}")

    # Return core objects for downstream use
    return {
        "df_wide": df_wide,
        "design": design,
        "id_col": id_col,
        "quant_cols": quant_cols,
    }
