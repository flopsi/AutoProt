# pages/1_Protein_Upload.py
import io
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Protein Upload", layout="wide")


@st.cache_data(show_spinner="Loading file...")
def _read_table(bytes_data: bytes) -> pd.DataFrame:
    text = bytes_data.decode("utf-8", errors="replace")
    if text.startswith("\ufeff"):
        text = text[1:]
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")


def clear_protein_state():
    for k in [
        "protein_bytes",
        "protein_name",
        "protein_upload",
    ]:
        if k in st.session_state:
            del st.session_state[k]


def load_and_map_protein() -> dict:
    """
    Load single wide-format protein table (no external metadata file)
    and let the user define:
      - Protein Group Identifier column
      - Condition column (from non-numeric columns)
      - Replicate column (from non-numeric columns)
      - Quantitative (intensity) columns (from numeric columns)

    Returns:
      {
        "df_wide": DataFrame,
        "design": DataFrame with [sample_id, condition, replicate],
        "protein_id_col": str,
        "quant_cols": list[str],
        "level": "protein",
      }
    """
    if "protein_bytes" not in st.session_state:
        st.warning("Upload a protein table first.")
        st.stop()

    df = _read_table(st.session_state.protein_bytes)

    st.subheader("1. Table overview")
    st.write(f"Protein table: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

    # Split columns by dtype (numeric vs non-numeric)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()        # [web:18]
    non_numeric_cols = df.columns.difference(numeric_cols).tolist()

    with st.expander("Inspect column types", expanded=False):
        st.markdown("**Numeric (candidate intensity) columns:**")
        st.write(numeric_cols or "None")
        st.markdown("**Non-numeric (candidate metadata / ID) columns:**")
        st.write(non_numeric_cols or "None")

    # --- Protein Group Identifier ---
    st.subheader("2. Protein group identifier")
    if not non_numeric_cols:
        st.error("No non-numeric columns found to use as protein group identifiers.")
        st.stop()

    # Try to auto-suggest a PG-like column if present (MaxQuant / DIA-NN style) [web:88]
    default_pg_idx = 0
    for i, c in enumerate(non_numeric_cols):
        cname = c.lower()
        if "pg." in cname or "protein group" in cname or "protein ids" in cname:
            default_pg_idx = i
            break

    protein_id_col = st.selectbox(
        "Select the Protein Group Identifier column",
        options=non_numeric_cols,
        index=default_pg_idx,
    )

    # --- Condition & replicate mapping (from non-numeric columns) ---
    st.subheader("3. Condition and replicate mapping")

    if len(non_numeric_cols) < 2:
        st.warning(
            "Only one non-numeric column detected. You can still continue, but "
            "condition/replicate modeling will be limited."
        )

    col1, col2 = st.columns(2)
    with col1:
        condition_col = st.selectbox(
            "Condition column",
            options=non_numeric_cols,
            index=0,
            help="Choose a column that encodes experimental groups (e.g. A/B, treatment/control).",
            key="condition_col",
        )
    with col2:
        replicate_col = st.selectbox(
            "Replicate column",
            options=non_numeric_cols,
            index=min(1, len(non_numeric_cols) - 1),
            help="Choose a column that encodes replicate index or run label.",
            key="replicate_col",
        )

    # Sample_id is simply the column headers of intensity columns; we let user
    # decide which numeric columns are intensities in the next step.
    st.subheader("4. Select quantitative (intensity) columns")

    if not numeric_cols:
        st.error(
            "No numeric columns detected. Cannot infer quantitative intensities."
        )
        st.stop()

    # Heuristic: auto-select all numeric columns, user can deselect meta numerics
    quant_cols = st.multiselect(
        "Quantitative columns (intensities)",
        options=numeric_cols,
        default=numeric_cols,
        help="All numeric columns are shown; deselect any that are not intensities.",
    )

    if not quant_cols:
        st.error("Select at least one quantitative column.")
        st.stop()

    # Build design table from column names of quant_cols
    # For now, sample_id == column name; downstream pages can parse/derive more.
    design = pd.DataFrame(
        {
            "sample_id": quant_cols,
            "condition": df[condition_col].iloc[0] if len(df) > 0 else None,
            "replicate": df[replicate_col].iloc[0] if len(df) > 0 else None,
        }
    )

    # If condition/replicate vary by row, this simple approach may not capture
    # all complexity; you can later extend this to parse them from column names.

    # Check replicate balance based on inferred design
    replicate_counts = design.groupby("condition")["replicate"].nunique(dropna=False)
    balanced = replicate_counts.nunique() == 1

    st.subheader("5. Design check (per condition)")
    st.dataframe(replicate_counts.to_frame("n_replicates"))

    if not balanced:
        st.warning(
            "Conditions do not have equal numbers of replicates according to "
            "the current mapping. Unbalanced designs are allowed but may "
            "require more careful statistical modeling."
        )

    st.subheader("6. Summary")
    st.write(f"Protein ID column: `{protein_id_col}`")
    st.write(f"Number of quantitative columns: {len(quant_cols)}")
    st.write(f"Conditions detected: {design['condition'].nunique()}")

    return {
        "df_wide": df,
        "design": design,
        "protein_id_col": protein_id_col,
        "quant_cols": quant_cols,
        "level": "protein",
    }


def main():
    # Header / styling
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
        '<p>Protein Upload (no separate metadata file)</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.subheader("1. Upload protein table")
    col1, col2 = st.columns([3, 1])
    with col1:
        if "protein_bytes" not in st.session_state:
            uploaded = st.file_uploader(
                "Upload wide-format protein table (CSV/TSV/TXT)",
                type=["csv", "tsv", "txt"],
            )
            if uploaded:
                st.session_state["protein_bytes"] = uploaded.getvalue()
                st.session_state["protein_name"] = uploaded.name
                st.rerun()
        else:
            st.success(f"Protein table: **{st.session_state['protein_name']}**")
    with col2:
        if "protein_bytes" in st.session_state:
            if st.button("Clear protein"):
                clear_protein_state()
                st.rerun()

    if "protein_bytes" not in st.session_state:
        st.info("Upload a protein table to continue.")
        return

    st.divider()

    # Run mapping logic
    result = load_and_map_protein()
    st.session_state["protein_upload"] = result

    st.success(
        "Protein data and basic experimental design mapping are ready for "
        "downstream analysis (QC, statistics, etc.)."
    )

    st.info(
        "Open a downstream page in the sidebar (e.g. QC / Analysis) "
        "and use `st.session_state['protein_upload']` there."
    )

    # Floating restart control
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
            🔄 Restart Protein Upload
        </div>
        """,
        unsafe_allow_html=True,
    )

    if
