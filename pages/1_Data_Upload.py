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
class TransformsCache:
    log2: pd.DataFrame
    log10: pd.DataFrame
    sqrt: pd.DataFrame
    cbrt: pd.DataFrame  # cube root
    yeo_johnson: pd.DataFrame
    quantile: pd.DataFrame
    condition_wise_cvs: pd.DataFrame  # CV per condition per protein/peptide


@dataclass
class MSData:
    raw: pd.DataFrame  # untouched original data
    raw_filled: pd.DataFrame  # NaN, 0, 1 → 1, count these as missing
    missing_count: int  # count of cells that were 0, 1, or NaN
    numeric_cols: List[str]
    transforms: TransformsCache

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


from scipy.stats import yeojohnson

def _compute_yeo_johnson(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Yeo-Johnson power transform (handles 0 and negative values)."""
    result = df.copy()
    for col in df.columns:
        result[col], _ = yeojohnson(df[col].dropna())
    return result

def _compute_quantile_norm(df: pd.DataFrame) -> pd.DataFrame:
    """Quantile normalization: each sample gets same distribution."""
    # sort each column, replace with mean ranks across columns
    quantiles = np.sort(df.values, axis=0)
    quantiles_mean = quantiles.mean(axis=1)
    ranks = df.rank(axis=0, method='average').values.astype(int)
    return pd.DataFrame(
        quantiles_mean[ranks - 1],
        columns=df.columns,
        index=df.index,
    )

def _compute_condition_cvs(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CV per condition (inferred from column names: A1,A2,A3 → A; B1,B2,B3 → B)."""
    cvs = {}
    for col in df.columns:
        condition = col[0] if col[0].isalpha() else col.rsplit("_", 1)[0]
        if condition not in cvs:
            cvs[condition] = []
        cvs[condition].append(df[col])
    
    result = {}
    for condition, cols_data in cvs.items():
        stacked = pd.concat(cols_data, axis=1)
        result[condition] = (stacked.std(axis=1) / stacked.mean(axis=1)) * 100  # CV%
    
    return pd.DataFrame(result)


def build_msdata(processed_df: pd.DataFrame, numeric_cols_renamed: List[str]) -> MSData:
    # 1) raw: untouched copy
    raw = processed_df.copy()
    
    # Force numeric on selected columns
    raw[numeric_cols_renamed] = (
        raw[numeric_cols_renamed]
        .apply(pd.to_numeric, errors="coerce")
        .astype("float64")
    )

    # 2) raw_filled: NaN, 0, 1 → 1
    raw_filled = raw.copy()
    vals = raw_filled[numeric_cols_renamed]
    vals = vals.fillna(1.0)
    vals = vals.where(~vals.isin([0.0, 1.0]), 1.0)
    raw_filled[numeric_cols_renamed] = vals
    
    # Count missing (cells that were NaN, 0, or 1)
    missing_count = (raw_filled[numeric_cols_renamed] == 1.0).to_numpy().sum()

    # 3) Compute all transformations at once
    transforms = TransformsCache(
        log2=np.log2(raw_filled[numeric_cols_renamed]).copy(),
        log10=np.log10(raw_filled[numeric_cols_renamed]).copy(),
        sqrt=np.sqrt(raw_filled[numeric_cols_renamed]).copy(),
        cbrt=np.cbrt(raw_filled[numeric_cols_renamed]).copy(),
        yeo_johnson=_compute_yeo_johnson(raw_filled[numeric_cols_renamed]),
        quantile=_compute_quantile_norm(raw_filled[numeric_cols_renamed]),
        condition_wise_cvs=_compute_condition_cvs(raw_filled[numeric_cols_renamed]),
    )

    return MSData(
        raw=raw,
        raw_filled=raw_filled,
        missing_count=missing_count,
        numeric_cols=numeric_cols_renamed,
        transforms=transforms,
    )


# ------------- session state -------------
defaults = {
    "protein_model": None,
    "peptide_model": None,
    "protein_index_col": None,
    "peptide_index_col": None,
    "peptide_seq_col": None,
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
        raw_df = pd.read_csv(uploaded_file)
        st.session_state.raw_df = raw_df
    raw_df = st.session_state.raw_df

    st.success(f"Loaded {len(raw_df):,} rows, {len(raw_df.columns)} columns")
    st.dataframe(raw_df.head(5), use_container_width=True, height=250)

    all_cols = list(raw_df.columns)

    # ------- Step 1: select metadata columns (PG, species, etc.) -------
    st.markdown("### Select metadata columns")

    col1, col2, col3 = st.columns(3)
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

    with col3:
        peptide_seq_col = st.selectbox(
            "Peptide sequence column (optional)",
            options=["None"] + all_cols,
            index=0,
            key="peptide_seq_col_select",
        )
        peptide_seq_col = None if peptide_seq_col == "None" else peptide_seq_col

    other_metadata = st.multiselect(
        "Additional metadata columns (optional)",
        options=[c for c in all_cols if c not in {pg_col, species_col, peptide_seq_col} if c is not None],
        key="other_metadata",
    )

    meta_cols = [c for c in [pg_col, species_col, peptide_seq_col, *other_metadata] if c is not None]

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

        # Force numeric conversion on working_df BEFORE any comparisons
        working_df[numeric_cols_renamed] = (
            working_df[numeric_cols_renamed].apply(pd.to_numeric, errors="coerce").astype("float64")
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
        c4.metric("Data level", "Peptide" if peptide_seq_col else "Protein")

        # Cache
        st.markdown("---")
        data_type = "protein"
        existing_key = f"{data_type}_model"
        index_key = f"{data_type}_index_col"
        seq_key = f"{data_type}_seq_col"
        mask_key = f"{data_type}_missing_mask"

        if st.session_state.get(existing_key) is not None:
            st.warning(f"{data_type.capitalize()} data already cached. Confirming will overwrite.")

        col_b1, col_b2, _ = st.columns([1, 1, 3])
        with col_b1:
            if st.button("Confirm & cache", type="primary"):
                model = build_msdata(processed_df, numeric_cols_renamed)
                missing_mask = processed_df[numeric_cols_renamed].isna() | (
                    processed_df[numeric_cols_renamed] <= 1.0
                )

                st.session_state[existing_key] = model
                st.session_state[index_key] = pg_col
                st.session_state[seq_key] = peptide_seq_col
                st.session_state[mask_key] = missing_mask

                st.session_state.raw_df = None
                st.session_state.column_renames = {}
                st.session_state.selected_quant_cols = []
                st.session_state.upload_key += 1
                st.cache_data.clear()
                st.rerun()

        with col_b2:
            if st.button("Cancel"):
                st.cache_data.clear()
                st.session_state.raw_df = None
                st.session_state.column_renames = {}
                st.session_state.selected_quant_cols = []
                st.session_state.upload_key += 1
                st.rerun()


    config_fragment()

render_navigation(back_page="app.py", next_page="pages/2_EDA.py")
render_footer()
