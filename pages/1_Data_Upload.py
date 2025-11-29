import streamlit as st
import pandas as pd
import numpy as np
import re
from dataclasses import dataclass
from typing import List
from scipy.stats import yeojohnson
from sklearn.preprocessing import QuantileTransformer

from components import inject_custom_css, render_header, render_navigation, render_footer, COLORS

st.set_page_config(
    page_title="Data Upload | Thermo Fisher Scientific",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
render_header()


# -----------------------
# Data models
# -----------------------
@dataclass
class TransformsCache:
    log2: pd.DataFrame
    log10: pd.DataFrame
    sqrt: pd.DataFrame
    cbrt: pd.DataFrame
    yeo_johnson: pd.DataFrame
    quantile: pd.DataFrame
    condition_wise_cvs: pd.DataFrame


@dataclass
class MSData:
    raw: pd.DataFrame
    raw_filled: pd.DataFrame
    missing_count: int
    numeric_cols: List[str]
    transforms: TransformsCache


# -----------------------
# Utilities
# -----------------------
@st.cache_data
def auto_rename_columns(columns: List[str]) -> dict:
    """Generate A1, A2, A3, B1, B2, B3... naming scheme."""
    rename_map = {}
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, col in enumerate(columns):
        cond_idx = i // 3
        rep = (i % 3) + 1
        new_name = f"{letters[cond_idx]}{rep}" if cond_idx < len(letters) else f"C{cond_idx+1}_{rep}"
        rename_map[col] = new_name
    return rename_map


def filter_by_species(df: pd.DataFrame, col: str, species_tags: list[str]) -> pd.DataFrame:
    """Vectorized species filtering using regex mask."""
    if not species_tags or not col or col not in df.columns:
        return df
    pattern = "|".join(re.escape(tag) for tag in species_tags)
    mask = df[col].astype(str).str.contains(pattern, case=False, na=False)
    return df[mask]


# -----------------------
# Transform computation
# -----------------------
def _compute_yeo_johnson_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized Yeo-Johnson transform."""
    result = df.copy()
    for col in df.columns:
        clean = df[col].dropna()
        if len(clean) > 0:
            try:
                transformed, _ = yeojohnson(clean)
                result.loc[clean.index, col] = transformed
            except Exception:
                pass
    return result


def _compute_quantile_norm(df: pd.DataFrame) -> pd.DataFrame:
    """Quantile normalization: equalize distributions across samples."""
    try:
        qt = QuantileTransformer(output_distribution="normal", random_state=42, n_quantiles=min(1000, len(df)))
        normalized = qt.fit_transform(df.values)
        return pd.DataFrame(normalized, columns=df.columns, index=df.index)
    except Exception:
        return df.copy()


def _compute_condition_cvs(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CV% per condition (inferred from first letter of column name)."""
    cvs = {}
    for col in df.columns:
        condition = col[0] if col and col[0].isalpha() else "X"
        if condition not in cvs:
            cvs[condition] = []
        cvs[condition].append(df[col])

    result = {}
    for condition, cols_data in cvs.items():
        stacked = pd.concat(cols_data, axis=1)
        mean_val = stacked.mean(axis=1)
        std_val = stacked.std(axis=1)
        result[condition] = (std_val / mean_val * 100).fillna(0)

    return pd.DataFrame(result) if result else df.copy()


def compute_transforms(raw_filled: pd.DataFrame, numeric_cols: List[str]) -> TransformsCache:
    """Precompute all transformations at once."""
    data = raw_filled[numeric_cols]

    return TransformsCache(
        log2=np.log2(data).astype("float32"),
        log10=np.log10(np.maximum(data, 1)).astype("float32"),
        sqrt=np.sqrt(data).astype("float32"),
        cbrt=np.cbrt(data).astype("float32"),
        yeo_johnson=_compute_yeo_johnson_transform(data).astype("float32"),
        quantile=_compute_quantile_norm(data).astype("float32"),
        condition_wise_cvs=_compute_condition_cvs(data),
    )


def build_msdata(processed_df: pd.DataFrame, numeric_cols_renamed: List[str]) -> MSData:
    """Build complete MSData with all precomputed transforms."""
    raw = processed_df.copy()

    # Force numeric conversion on all quant columns
    raw[numeric_cols_renamed] = (
        raw[numeric_cols_renamed]
        .apply(pd.to_numeric, errors="coerce")
        .astype("float64")
    )

    # Fill: NaN, 0, 1 → 1
    raw_filled = raw.copy()
    for col in numeric_cols_renamed:
        raw_filled[col] = raw_filled[col].fillna(1.0).where(~raw_filled[col].isin([0.0, 1.0]), 1.0)

    missing_count = (raw_filled[numeric_cols_renamed] == 1.0).to_numpy().sum()

    # Precompute all transforms
    transforms = compute_transforms(raw_filled, numeric_cols_renamed)

    return MSData(
        raw=raw,
        raw_filled=raw_filled,
        missing_count=missing_count,
        numeric_cols=numeric_cols_renamed,
        transforms=transforms,
    )


@st.cache_data
def get_msdata(df: pd.DataFrame, quant_cols: List[str]) -> MSData:
    """Cached wrapper for build_msdata."""
    return build_msdata(df, quant_cols)


# -----------------------
# Session state
# -----------------------
DEFAULTS = {
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

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_upload_state():
    """Consolidated reset function."""
    st.cache_data.clear()
    st.session_state.raw_df = None
    st.session_state.column_renames = {}
    st.session_state.selected_quant_cols = []
    st.session_state.upload_key += 1


# -----------------------
# UI
# -----------------------
st.markdown("## Data upload")
st.caption("Import and configure mass spectrometry output matrices")

uploaded_file = st.file_uploader(
    "Upload proteomics CSV",
    type=["csv"],
    key=f"uploader_{st.session_state.upload_key}",
)

if not uploaded_file:
    st.info("Upload a CSV file to begin")
    render_navigation(back_page="app.py", next_page="pages/2_EDA.py")
    render_footer()
    st.stop()

# Load or retrieve cached dataframe
if st.session_state.raw_df is None:
    raw_df = pd.read_csv(uploaded_file)
    st.session_state.raw_df = raw_df
else:
    raw_df = st.session_state.raw_df

st.success(f"Loaded {len(raw_df):,} rows, {len(raw_df.columns)} columns")
st.dataframe(raw_df.head(5), width='stretch', height=250)

all_cols = list(raw_df.columns)

# Step 1: metadata columns
st.markdown("### Select metadata columns")

col1, col2, col3 = st.columns(3)
with col1:
    pg_col = st.selectbox("Protein group / ID column", options=["None"] + all_cols, index=1 if all_cols else 0, key="pg_col")
    pg_col = None if pg_col == "None" else pg_col

with col2:
    species_col = st.selectbox("Species annotation column (optional)", options=["None"] + all_cols, index=0, key="species_col_select")
    species_col = None if species_col == "None" else species_col

with col3:
    peptide_seq_col = st.selectbox("Peptide sequence column (optional)", options=["None"] + all_cols, index=0, key="peptide_seq_col_select")
    peptide_seq_col = None if peptide_seq_col == "None" else peptide_seq_col

other_metadata = st.multiselect(
    "Additional metadata columns (optional)",
    options=[c for c in all_cols if c not in {pg_col, species_col, peptide_seq_col} if c is not None],
    key="other_metadata",
)

meta_cols = [c for c in [pg_col, species_col, peptide_seq_col, *other_metadata] if c is not None]

# Step 2: quantitative columns
candidate_quant = [c for c in all_cols if c not in meta_cols]

if not candidate_quant:
    st.error("No candidate quant columns left after metadata selection.")
    st.stop()

st.markdown("### Select quantitative columns (in groups of 3)")
st.caption("Labels show the last 25 characters of each header.")

def last25(s: str) -> str:
    s = str(s)
    return s[-25:] if len(s) > 25 else s

if not st.session_state.selected_quant_cols:
    st.session_state.selected_quant_cols = candidate_quant.copy()

selected_quant = []
for i in range(0, len(candidate_quant), 3):
    group = candidate_quant[i : i + 3]
    label = " | ".join(last25(c) for c in group)
    checked = st.checkbox(
        label,
        key=f"quant_group_{i}",
        value=all(c in st.session_state.selected_quant_cols for c in group),
    )
    if checked:
        selected_quant.extend(group)

if not selected_quant:
    st.error("Select at least one group of quantitative columns to continue.")
    st.stop()

st.session_state.selected_quant_cols = selected_quant
numeric_cols_orig = selected_quant
st.session_state.column_renames = auto_rename_columns(numeric_cols_orig)

# Fragment: filtering & renaming
@st.fragment
def config_fragment():
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
                species_tags.append(custom_species)

        with st.expander("Edit quant column names (A1,A2,A3,...)", expanded=False):
            edited = {}
            for i in range(0, len(numeric_cols_orig), 6):
                cols = st.columns(6)
                for j, orig in enumerate(numeric_cols_orig[i : i + 6]):
                    with cols[j]:
                        edited[orig] = st.text_input(
                            f"Col {i + j + 1}",
                            value=st.session_state.column_renames.get(orig, orig),
                            key=f"rename_{i+j}",
                        )
            if st.button("Apply renames"):
                st.session_state.column_renames.update(edited)
                st.rerun(scope="fragment")

    rename_map = {k: v for k, v in st.session_state.column_renames.items() if k in raw_df.columns}
    working_df = raw_df.rename(columns=rename_map)

    # Force numeric conversion early
    numeric_cols_renamed = [st.session_state.column_renames.get(c, c) for c in numeric_cols_orig]
    working_df[numeric_cols_renamed] = (
        working_df[numeric_cols_renamed]
        .apply(pd.to_numeric, errors="coerce")
        .astype("float64")
    )

    # Species filter
    processed_df = filter_by_species(working_df, species_col, species_tags)
    if species_col and species_tags:
        st.info(f"Filtered to {len(processed_df):,} rows matching: {', '.join(species_tags)}")

    st.markdown("### Data preview")
    st.dataframe(processed_df.head(10), width='stretch', height=300)

    # Metrics
    conditions = {col[0] for col in numeric_cols_renamed if col and col[0].isalpha()}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(processed_df):,}")
    c2.metric("Samples", len(numeric_cols_renamed))
    c3.metric("Conditions", len(conditions))
    c4.metric("Data level", "Peptide" if peptide_seq_col else "Protein")

    # Cache buttons
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
            missing_mask = processed_df[numeric_cols_renamed].isna() | (processed_df[numeric_cols_renamed] <= 1.0)

            st.session_state[existing_key] = model
            st.session_state[index_key] = pg_col
            st.session_state[seq_key] = peptide_seq_col
            st.session_state[mask_key] = missing_mask

            reset_upload_state()
            st.rerun()

    with col_b2:
        if st.button("Cancel"):
            reset_upload_state()
            st.rerun()

config_fragment()

render_navigation(back_page="app.py", next_page="pages/2_EDA.py")
render_footer()
