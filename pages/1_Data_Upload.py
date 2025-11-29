import streamlit as st
import pandas as pd
import numpy as np
import re
from dataclasses import dataclass
from typing import List

from components import inject_custom_css, render_header, render_navigation, render_footer, COLORS

st.set_page_config(
    page_title="Data Upload | Thermo Fisher Scientific",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
render_header()


@dataclass
class MSData:
    original: pd.DataFrame
    filled: pd.DataFrame
    log2_filled: pd.DataFrame
    numeric_cols: List[str]


def auto_rename_columns(columns: List[str]) -> dict:
    rename_map = {}
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, col in enumerate(columns):
        cond_idx = i // 3
        rep = (i % 3) + 1
        if cond_idx < len(letters):
            new_name = f"{letters[cond_idx]}{rep}"
        else:
            new_name = f"C{cond_idx+1}_{rep}"
        rename_map[col] = new_name
    return rename_map


def filter_by_species(df: pd.DataFrame, col: str, species_tags: list[str]) -> pd.DataFrame:
    if not species_tags or not col:
        return df
    pattern = "|".join(re.escape(tag) for tag in species_tags)
    return df[df[col].astype(str).str.contains(pattern, case=False, na=False)]


def build_msdata(processed_df: pd.DataFrame, numeric_cols_renamed: List[str]) -> MSData:
    original = processed_df.copy()

    filled = processed_df.copy()
    vals = filled[numeric_cols_renamed]
    vals = vals.fillna(1)
    vals = vals.where(~vals.isin([0, 1]), 1)
    filled[numeric_cols_renamed] = vals

    log2_filled = filled.copy()
    log2_filled[numeric_cols_renamed] = np.log2(log2_filled[numeric_cols_renamed])

    return MSData(
        original=original,
        filled=filled,
        log2_filled=log2_filled,
        numeric_cols=numeric_cols_renamed,
    )


# -----------------------
# Session state init
# -----------------------
defaults = {
    "protein_model": None,
    "peptide_model": None,
    "protein_index_col": None,
    "peptide_index_col": None,
    "protein_missing_mask": None,
    "peptide_missing_mask": None,
    "upload_key": 0,
    "raw_df": None,
    "column_renames": {},
    "original_numeric_cols": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


st.markdown("## Data upload")
st.caption("Import and configure mass spectrometry output matrices")

uploaded_file = st.file_uploader(
    "Upload proteomics CSV",
    type=["csv"],
    key=f"uploader_{st.session_state.upload_key}",
)

if uploaded_file:
    if st.session_state.raw_df is None:
        raw_df = pd.read_csv(uploaded_file)
        st.session_state.raw_df = raw_df
        # Detect all numeric columns (candidates)
        numeric_all = [c for c in raw_df.columns if pd.api.types.is_numeric_dtype(raw_df[c])]
        if not numeric_all:
            st.error("No numeric columns detected. Please upload a matrix with numeric intensities.")
            st.stop()
    
        st.session_state.original_numeric_cols = numeric_all
        st.session_state.column_renames = auto_rename_columns(numeric_all)

        raw_df = st.session_state.raw_df
        numeric_all = st.session_state.original_numeric_cols
    
        st.success(f"Loaded {len(raw_df):,} rows, {len(raw_df.columns)} columns")

    # --------------------------
    # Quant column selection with last-25-char labels
    # --------------------------
st.markdown("### Select quantitative columns")
st.caption("Each option shows the last 25 characters of the original numeric column name.")

    # Build display labels: "…<last 25 chars>"
def last25(name: str) -> str:
    s = str(name)
    return s[-25:] if len(s) > 25 else s

options = numeric_all
format_func = lambda col: last25(col)

selected_numeric = st.multiselect(
    "Quantitative intensity columns",
    options=options,
    default=options,         # default: all numeric are quant
    format_func=format_func,
    key="quant_cols_select",
)

if not selected_numeric:
    st.error("Select at least one quantitative column to continue.")
    st.stop()

    st.session_state.original_numeric_cols = selected_numeric
    st.session_state.column_renames = auto_rename_columns(selected_numeric)
    original_numeric_cols = st.session_state.original_numeric_cols
    non_numeric_cols = [c for c in raw_df.columns if c not in original_numeric_cols]

@st.fragment
def config_fragment():
    # 1) Column configuration
    with st.expander("Column configuration", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Protein group column**")
            protein_group_col = st.selectbox(
                "Select protein group IDs",
                options=["None"] + non_numeric_cols,
                index=1 if non_numeric_cols else 0,
                label_visibility="collapsed",
                key="pg_col",
            )
            protein_group_col = None if protein_group_col == "None" else protein_group_col

        with col2:
            st.markdown("**Sequence column (peptide data)**")
            sequence_col = st.selectbox(
                "Select peptide sequences",
                options=["None"] + non_numeric_cols,
                index=0,
                label_visibility="collapsed",
                key="seq_col",
            )
            sequence_col = None if sequence_col == "None" else sequence_col
            is_peptide_data = sequence_col is not None

        with col3:
            st.markdown("**Data level**")
            if is_peptide_data:
                st.markdown('<span class="status-badge badge-peptide">Peptide</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge badge-protein">Protein</span>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_sp1, col_sp2 = st.columns([3, 1])
        with col_sp1:
            species_tags = st.multiselect(
                "Species filter tags",
                options=["_HUMAN", "_YEAST", "_ECOLI", "_MOUSE"],
                default=["_HUMAN"],
                key="species",
            )
        with col_sp2:
            custom_species = st.text_input("Custom tag", key="custom_sp")
            if custom_species and custom_species not in species_tags:
                species_tags = species_tags + [custom_species]

    # 2) Column renaming
    with st.expander("Edit column names (auto-named as A1,A2,A3,B1,B2,B3,...)", expanded=False):
        st.caption("Columns are grouped in sets of 3 replicates per condition.")
        edited_names = {}
        cols_per_row = 6
        for i in range(0, len(original_numeric_cols), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j, orig_col in enumerate(original_numeric_cols[i : i + cols_per_row]):
                with row_cols[j]:
                    edited_names[orig_col] = st.text_input(
                        f"Col {i + j + 1}",
                        value=st.session_state.column_renames.get(orig_col, orig_col),
                        key=f"edit_{i + j}",
                    )

        if st.button("Apply renames"):
            st.session_state.column_renames.update(edited_names)
            st.rerun(scope="fragment")

        # Apply renames
        rename_map = {k: v for k, v in st.session_state.column_renames.items() if k in raw_df.columns}
        working_df = raw_df.rename(columns=rename_map)

        numeric_cols_renamed = [st.session_state.column_renames.get(c, c) for c in original_numeric_cols]

        # 3) Species filter
        filter_col = None
        for potential in ["PG.ProteinNames", "ProteinNames", "Protein.Names"]:
            if potential in working_df.columns:
                filter_col = potential
                break

        processed_df = working_df.copy()
        if species_tags and filter_col:
            processed_df = filter_by_species(processed_df, filter_col, species_tags)
            st.info(f"Filtered to {len(processed_df):,} rows matching: {', '.join(species_tags)}")

        # 4) Preview
        st.markdown("### Data preview")

        conditions = set()
        for c in numeric_cols_renamed:
            if len(c) >= 1:
                conditions.add(c[0] if c[0].isalpha() else c.rsplit("_", 1)[0])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(processed_df):,}")
        c2.metric("Samples", len(numeric_cols_renamed))
        c3.metric("Conditions", len(conditions))
        c4.metric("Data level", "Peptide" if is_peptide_data else "Protein")

        st.dataframe(processed_df.head(10), use_container_width=True, height=300)

        # 5) Cache
        st.markdown("---")
        data_type = "peptide" if is_peptide_data else "protein"
        existing_key = f"{data_type}_model"
        index_key = f"{data_type}_index_col"
        mask_key = f"{data_type}_missing_mask"

        if st.session_state.get(existing_key) is not None:
            st.warning(f"{data_type.capitalize()} data already cached. Confirming will overwrite.")

        st.markdown(f"### Cache as {data_type} data?")
        st.caption("Missing values (NaN, 0, 1) in numeric columns will be replaced with 1.")

        col_btn1, col_btn2, _ = st.columns([1, 1, 3])
        with col_btn1:
            if st.button("Confirm & cache", type="primary"):
                model = build_msdata(processed_df, numeric_cols_renamed)

                missing_mask = processed_df[numeric_cols_renamed].isna() | (
                    processed_df[numeric_cols_renamed] <= 1
                )

                st.session_state[existing_key] = model
                st.session_state[index_key] = protein_group_col
                st.session_state[mask_key] = missing_mask

                st.session_state.raw_df = None
                st.session_state.column_renames = {}
                st.session_state.original_numeric_cols = []
                st.session_state.upload_key += 1
                st.rerun()

        with col_btn2:
            if st.button("Cancel"):
                st.session_state.raw_df = None
                st.session_state.column_renames = {}
                st.session_state.original_numeric_cols = []
                st.session_state.upload_key += 1
                st.rerun()

    config_fragment()

else:
    st.info("Upload a CSV file to begin")


@st.fragment
def cached_data_fragment():
    if st.session_state.protein_model is None and st.session_state.peptide_model is None:
        return

    st.markdown("---")
    st.markdown("### Cached datasets")

    tab1, tab2 = st.tabs(["Protein data", "Peptide data"])

    with tab1:
        model: MSData | None = st.session_state.protein_model
        if model is not None:
            st.caption(f"Index column: `{st.session_state.protein_index_col}`")
            st.dataframe(model.original.head(5), use_container_width=True)

            mask = st.session_state.protein_missing_mask
            if mask is not None:
                total = mask.size
                missing = mask.sum().sum()
                st.caption(f"Missing values: {missing:,} ({100 * missing / total:.1f}%)")

            if st.button("Clear protein data", key="clear_protein"):
                st.session_state.protein_model = None
                st.session_state.protein_index_col = None
                st.session_state.protein_missing_mask = None
                st.rerun()
        else:
            st.caption("No protein data uploaded yet")

    with tab2:
        model: MSData | None = st.session_state.peptide_model
        if model is not None:
            st.caption(f"Index column: `{st.session_state.peptide_index_col}`")
            st.dataframe(model.original.head(5), use_container_width=True)

            mask = st.session_state.peptide_missing_mask
            if mask is not None:
                total = mask.size
                missing = mask.sum().sum()
                st.caption(f"Missing values: {missing:,} ({100 * missing / total:.1f}%)")

            if st.button("Clear peptide data", key="clear_peptide"):
                st.session_state.peptide_model = None
                st.session_state.peptide_index_col = None
                st.session_state.peptide_missing_mask = None
                st.rerun()
        else:
            st.caption("No peptide data uploaded yet")


cached_data_fragment()

render_navigation(back_page="app.py", next_page="pages/2_EDA.py")
render_footer()
