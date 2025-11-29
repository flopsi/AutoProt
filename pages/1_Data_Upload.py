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

@st.cache_data
def get_msdata(df, quant_cols):
    return build_msdata(df, quant_cols)

@dataclass
class MSData:
    original: pd.DataFrame
    filled: pd.DataFrame
    log2_filled: pd.DataFrame
    numeric_cols: List[str]
    ones_count: int  # new


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

    # 1) Quant columns → numeric; non-numeric (e.g. '#NUM!') → NaN
    numeric_block = original[numeric_cols_renamed].apply(
        pd.to_numeric, errors="coerce"
    )
    original[numeric_cols_renamed] = numeric_block

    # 2) Filled: NaN, 0, 1 → 1
    filled = original.copy()
    vals = filled[numeric_cols_renamed]
    vals = vals.fillna(1)
    vals = vals.where(~vals.isin([0, 1]), 1)
    filled[numeric_cols_renamed] = vals

    # Count cells equal to 1 after filling/reformatting
    ones_count = (filled[numeric_cols_renamed] == 1).sum().sum()

    # 3) log2(filled)
    log2_filled = filled.copy()
    log2_filled[numeric_cols_renamed] = np.log2(log2_filled[numeric_cols_renamed])

    return MSData(
        original=original,
        filled=filled,
        log2_filled=log2_filled,
        numeric_cols=numeric_cols_renamed,
        ones_count=ones_count,
    )

# ------------- session state -------------
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
    "selected_quant_cols": [],
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
        raw_df = pd.read_csv(uploaded_file)  # first row is header
        st.session_state.raw_df = raw_df
    raw_df = st.session_state.raw_df

    st.success(f"Loaded {len(raw_df):,} rows, {len(raw_df.columns)} columns")
    st.dataframe(raw_df.head(5), use_container_width=True, height=250)

    all_cols = list(raw_df.columns)

    # ------- Step 1: select metadata columns (PG, species, etc.) -------
    st.markdown("### Select metadata columns")

    col1, col2 = st.columns(2)
    with col1:
        pg_col = st.selectbox(
            "Protein group / ID column",
            options=["None"] + all_cols,
            index=1 if len(all_cols) > 0 else 0,
            key="pg_col",
        )
        pg_col = None if pg_col == "None" else pg_col

    with col2:
        species_col = st.selectbox(
            "Species annotation column (optional)",
            options=["None"] + all_cols,
            index=0,
            key="species_col_select",
        )
        species_col = None if species_col == "None" else species_col

    other_metadata = st.multiselect(
        "Additional metadata columns (optional)",
        options=[c for c in all_cols if c not in {pg_col, species_col} if c is not None],
        key="other_metadata",
    )

    meta_cols = [c for c in [pg_col, species_col, *other_metadata] if c is not None]

    # ------- Step 2: candidate quant columns = everything else -------
    candidate_quant = [c for c in all_cols if c not in meta_cols]

    if not candidate_quant:
        st.error("No candidate quant columns left after metadata selection.")
        st.stop()

    st.markdown("### Select quantitative columns (in groups of 3)")
    st.caption(
        "Columns not marked as metadata are grouped in packages of 3. "
        "Labels show the last 25 characters of each header."
    )

    def last25(s: str) -> str:
        s = str(s)
        return s[-25:] if len(s) > 25 else s

    # Initialize selection state (default: all groups selected)
    if not st.session_state.selected_quant_cols:
        st.session_state.selected_quant_cols = candidate_quant.copy()

    selected_quant = []

    group_size = 3
    for i in range(0, len(candidate_quant), group_size):
        group = candidate_quant[i : i + group_size]
        label_parts = [last25(c) for c in group]
        group_label = " | ".join(label_parts)

        # one checkbox per group
        checked = st.checkbox(
            group_label,
            key=f"quant_group_{i}",
            value=all(c in st.session_state.selected_quant_cols for c in group),
        )

        if checked:
            selected_quant.extend(group)

    if not selected_quant:
        st.error("Select at least one group of quantitative columns to continue.")
        st.stop()

    st.session_state.selected_quant_cols = selected_quant

    # From here on, treat selected_quant as numeric/quant columns
    numeric_cols_orig = selected_quant
    st.session_state.column_renames = auto_rename_columns(numeric_cols_orig)

    non_numeric_cols = [c for c in all_cols if c not in numeric_cols_orig]

    @st.fragment
    def config_fragment():
        # 3) species filter options (on species_col if given)
        with st.expander("Species filter and column renaming", expanded=True):
            col_sp1, col_sp2 = st.columns([3, 1])
            with col_sp1:
                species_tags = st.multiselect(
                    "Species filter tags",
                    options=["_HUMAN", "_YEAST", "_ECOLI", "_MOUSE"],
                    default=["_HUMAN"],
                    key="species_tags",
                )
            with col_sp2:
                custom_species = st.text_input("Custom tag", key="custom_sp_tag")
                if custom_species and custom_species not in species_tags:
                    species_tags = species_tags + [custom_species]

            with st.expander("Edit quant column names (A1,A2,A3,B1,B2,B3,...)", expanded=False):
                edited = {}
                cols_per_row = 6
                for i in range(0, len(numeric_cols_orig), cols_per_row):
                    row_cols = st.columns(cols_per_row)
                    for j, orig in enumerate(numeric_cols_orig[i : i + cols_per_row]):
                        with row_cols[j]:
                            edited[orig] = st.text_input(
                                f"Col {i + j + 1}",
                                value=st.session_state.column_renames.get(orig, orig),
                                key=f"rename_{i+j}",
                            )
                if st.button("Apply renames"):
                    st.session_state.column_renames.update(edited)
                    st.rerun(scope="fragment")

        # Apply renames
        rename_map = {k: v for k, v in st.session_state.column_renames.items() if k in raw_df.columns}
        working_df = raw_df.rename(columns=rename_map)

        numeric_cols_renamed = [st.session_state.column_renames.get(c, c) for c in numeric_cols_orig]

def build_msdata(processed_df: pd.DataFrame, numeric_cols_renamed: List[str]) -> MSData:
    original = processed_df.copy()

    # 1) Force all selected quant columns to numeric float64 BEFORE caching/fragment logic
    original[numeric_cols_renamed] = (
        original[numeric_cols_renamed]
        .apply(pd.to_numeric, errors="coerce")
        .astype("float64")
    )

    # 2) Filled: NaN, 0, 1 → 1
    filled = original.copy()
    vals = filled[numeric_cols_renamed]
    vals = vals.fillna(1.0)
    vals = vals.where(~vals.isin([0, 1, 0.0, 1.0]), 1.0)
    filled[numeric_cols_renamed] = vals

    # Count cells == 1 after filling
    ones_count = (filled[numeric_cols_renamed] == 1.0).to_numpy().sum()

    # 3) log2(filled)
    log2_filled = filled.copy()
    log2_filled[numeric_cols_renamed] = np.log2(log2_filled[numeric_cols_renamed])

    return MSData(
        original=original,
        filled=filled,
        log2_filled=log2_filled,
        numeric_cols=numeric_cols_renamed,
        ones_count=ones_count,
    )

        # Species filter
        processed_df = working_df.copy()
        if species_col and species_tags:
            processed_df = filter_by_species(processed_df, species_col, species_tags)
            st.info(f"Filtered to {len(processed_df):,} rows matching: {', '.join(species_tags)}")

        # Preview
        st.markdown("### Data preview")
        st.dataframe(processed_df.head(10), use_container_width=True, height=300)

        conditions = set()
        for c in numeric_cols_renamed:
            if len(c) >= 1:
                conditions.add(c[0] if c[0].isalpha() else c.rsplit("_", 1)[0])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(processed_df):,}")
        c2.metric("Samples", len(numeric_cols_renamed))
        c3.metric("Conditions", len(conditions))
        c4.metric("Data level", "Peptide" if species_col and "PEPTIDE" in species_col.upper() else "Protein")

        # Cache
        st.markdown("---")
        data_type = "protein"  # or determine from context if you have a toggle
        existing_key = f"{data_type}_model"
        index_key = f"{data_type}_index_col"
        mask_key = f"{data_type}_missing_mask"

        if st.session_state.get(existing_key) is not None:
            st.warning(f"{data_type.capitalize()} data already cached. Confirming will overwrite.")

        col_b1, col_b2, _ = st.columns([1, 1, 3])
        with col_b1:
            if st.button("Confirm & cache", type="primary"):
                model = build_msdata(processed_df, numeric_cols_renamed)
                missing_mask = processed_df[numeric_cols_renamed].isna() | (
                    processed_df[numeric_cols_renamed] <= 1
                )

                st.session_state[existing_key] = model
                st.session_state[index_key] = pg_col
                st.session_state[mask_key] = missing_mask

                st.session_state.raw_df = None
                st.session_state.column_renames = {}
                st.session_state.selected_quant_cols = []
                st.session_state.upload_key += 1
                st.rerun()

        with col_b2:
            if st.button("Cancel"):
                st.session_state.raw_df = None
                st.session_state.column_renames = {}
                st.session_state.selected_quant_cols = []
                st.session_state.upload_key += 1
                st.rerun()

    config_fragment()

else:
    st.info("Upload a CSV file to begin")

render_navigation(back_page="app.py", next_page="pages/2_EDA.py")
render_footer()
