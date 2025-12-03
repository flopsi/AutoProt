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
class FilteredSubgroups:
    """Pre-filtered species subsets for fast filtering."""
    human: pd.DataFrame
    yeast: pd.DataFrame
    ecoli: pd.DataFrame
    mouse: pd.DataFrame
    all_species: pd.DataFrame
    
    def get(self, species_list: List[str]) -> pd.DataFrame:
        """Get combined dataframe for selected species."""
        if not species_list:
            return self.all_species
        
        # If all species selected, return full dataset
        if set(species_list) == {"HUMAN", "YEAST", "ECOLI", "MOUSE"}:
            return self.all_species
        
        # Combine requested species
        dfs = []
        for sp in species_list:
            if sp == "HUMAN" and not self.human.empty:
                dfs.append(self.human)
            elif sp == "YEAST" and not self.yeast.empty:
                dfs.append(self.yeast)
            elif sp == "ECOLI" and not self.ecoli.empty:
                dfs.append(self.ecoli)
            elif sp == "MOUSE" and not self.mouse.empty:
                dfs.append(self.mouse)
        
        return pd.concat(dfs, axis=0) if dfs else pd.DataFrame()


@dataclass
class MSData:
    raw: pd.DataFrame
    raw_filled: pd.DataFrame
    missing_count: int
    numeric_cols: List[str]
    transforms: TransformsCache
    species_subgroups: 'FilteredSubgroups'  # Add this line
    species_col: str


def build_species_subgroups(df: pd.DataFrame, species_col: str, numeric_cols: List[str]) -> FilteredSubgroups:
    """Pre-filter data by species for fast access."""
    if species_col not in df.columns:
        return FilteredSubgroups(
            human=pd.DataFrame(),
            yeast=pd.DataFrame(),
            ecoli=pd.DataFrame(),
            mouse=pd.DataFrame(),
            all_species=df,
        )
    
    species_series = df[species_col]
    
    return FilteredSubgroups(
        human=df[species_series == "HUMAN"].copy(),
        yeast=df[species_series == "YEAST"].copy(),
        ecoli=df[species_series == "ECOLI"].copy(),
        mouse=df[species_series == "MOUSE"].copy(),
        all_species=df.copy(),
    )


@dataclass
class MSData:
    raw: pd.DataFrame
    raw_filled: pd.DataFrame
    missing_count: int
    numeric_cols: List[str]
    transforms: TransformsCache


@st.cache_data
def auto_rename_columns(columns: List[str]) -> dict:
    rename_map = {}
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, col in enumerate(columns):
        cond_idx = i // 3
        rep = (i % 3) + 1
        new_name = f"{letters[cond_idx]}{rep}" if cond_idx < len(letters) else f"C{cond_idx+1}_{rep}"
        rename_map[col] = new_name
    return rename_map


def extract_species_from_protein_id(protein_id: str) -> str:
    """Extract species from protein ID or return as-is if already a species name."""
    if pd.isna(protein_id):
        return "UNKNOWN"
    
    protein_str = str(protein_id).upper().strip()
    
    # If it's already a plain species name, return it
    if protein_str in ["HUMAN", "YEAST", "ECOLI", "MOUSE"]:
        return protein_str
    
    # Otherwise, extract from protein ID format
    if "_HUMAN" in protein_str or "HUMAN_" in protein_str:
        return "HUMAN"
    elif "_YEAST" in protein_str or "YEAST_" in protein_str:
        return "YEAST"
    elif "_ECOLI" in protein_str or "ECOLI_" in protein_str:
        return "ECOLI"
    elif "_MOUSE" in protein_str or "MOUSE_" in protein_str:
        return "MOUSE"
    else:
        return "UNKNOWN"


def filter_by_species(df: pd.DataFrame, species_col: str | None, species_tags: list[str]) -> pd.DataFrame:
    """Filter by species tags."""
    if not species_tags or not species_col or species_col not in df.columns:
        return df
    mask = df[species_col].isin(species_tags)
    return df[mask]


def _compute_yeo_johnson_transform(df: pd.DataFrame) -> pd.DataFrame:
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
    try:
        qt = QuantileTransformer(output_distribution="normal", random_state=42, n_quantiles=min(1000, len(df)))
        normalized = qt.fit_transform(df.values)
        return pd.DataFrame(normalized, columns=df.columns, index=df.index)
    except Exception:
        return df.copy()


def _compute_condition_cvs(df: pd.DataFrame) -> pd.DataFrame:
    cvs = {}
    for col in df.columns:
        condition = col[0] if col and col[0].isalpha() else "X"
        cvs.setdefault(condition, []).append(df[col])

    result = {}
    for condition, cols_data in cvs.items():
        stacked = pd.concat(cols_data, axis=1)
        mean_val = stacked.mean(axis=1)
        std_val = stacked.std(axis=1)
        result[condition] = (std_val / mean_val * 100).fillna(0)

    return pd.DataFrame(result) if result else df.copy()


def compute_transforms(raw_filled: pd.DataFrame, numeric_cols: List[str]) -> TransformsCache:
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


def build_msdata(
    processed_df: pd.DataFrame, 
    numeric_cols_renamed: List[str], 
    species_col: str = "_extracted_species",
    _version: int = 2  # Increment this to break cache
) -> MSData:
    """Build MSData with pre-cached species subgroups."""
    raw = processed_df.copy()

    raw[numeric_cols_renamed] = (
        raw[numeric_cols_renamed]
        .apply(pd.to_numeric, errors="coerce")
        .astype("float64")
    )

    raw_filled = raw.copy()
    vals = raw_filled[numeric_cols_renamed]
    vals = vals.fillna(1.0)
    vals = vals.where(~vals.isin([0.0, 1.0]), 1.0)
    raw_filled[numeric_cols_renamed] = vals

    missing_count = (raw_filled[numeric_cols_renamed] == 1.0).to_numpy().sum()

    transforms = compute_transforms(raw_filled, numeric_cols_renamed)
    species_subgroups = build_species_subgroups(raw_filled, species_col, numeric_cols_renamed)

    return MSData(
        raw=raw,
        raw_filled=raw_filled,
        missing_count=missing_count,
        numeric_cols=numeric_cols_renamed,
        transforms=transforms,
        species_subgroups=species_subgroups,
        species_col=species_col,
    )


@st.cache_data
def get_msdata(df: pd.DataFrame, quant_cols: List[str]) -> MSData:
    return build_msdata(df, quant_cols)


DEFAULTS = {
    "protein_model": None,
    "peptide_model": None,
    "protein_index_col": None,
    "peptide_index_col": None,
    "protein_species_col": None,
    "peptide_species_col": None,
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
    st.cache_data.clear()
    st.session_state.raw_df = None
    st.session_state.column_renames = {}
    st.session_state.selected_quant_cols = []
    st.session_state.upload_key = st.session_state.get("upload_key", 0) + 1


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

if st.session_state.raw_df is None:
    raw_df = pd.read_csv(uploaded_file)
    st.session_state.raw_df = raw_df
else:
    raw_df = st.session_state.raw_df

st.success(f"Loaded {len(raw_df):,} rows, {len(raw_df.columns)} columns")
st.dataframe(raw_df.head(5), width="stretch", height=250)

all_cols = list(raw_df.columns)

st.markdown("### Select metadata columns")

col1, col2, col3 = st.columns(3)
with col1:
    pg_col = st.selectbox(
        "Protein group / ID column",
        options=["None"] + all_cols,
        index=1 if all_cols else 0,
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

# NEW: Drop columns selector
drop_cols = st.multiselect(
    "⚠️ Drop columns (these will be permanently removed)",
    options=[c for c in all_cols if c not in {pg_col, species_col, peptide_seq_col} if c is not None],
    default=[],
    key="drop_columns",
    help="Select columns to remove from the dataset. Useful for cleaning up unwanted metadata."
)

# Keep only essential metadata
meta_cols = [c for c in [pg_col, species_col, peptide_seq_col] if c is not None]

# Candidate quant columns = everything except metadata and dropped columns
candidate_quant = [c for c in all_cols if c not in meta_cols and c not in drop_cols]

if not candidate_quant:
    st.error("No candidate quant columns left after metadata selection and drops.")
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


@st.fragment
def config_fragment():
    with st.expander("Species filter and column renaming", expanded=True):
        col_sp1, col_sp2 = st.columns([3, 1])
        with col_sp1:
            species_tags = st.multiselect(
                "Species filter tags",
                options=["HUMAN", "YEAST", "ECOLI", "MOUSE"],
                default=["HUMAN"],
                key="species_tags",
            )
        with col_sp2:
            custom_species = st.text_input("Custom species tag", key="custom_sp_tag")
            if custom_species and custom_species not in species_tags:
                species_tags.append(custom_species.upper())

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

    # Apply column renames
    rename_map = {k: v for k, v in st.session_state.column_renames.items() if k in raw_df.columns}
    working_df = raw_df.rename(columns=rename_map)
    
    # DROP SELECTED COLUMNS - NEW
    if drop_cols:
        working_df = working_df.drop(columns=drop_cols)
        st.info(f"🗑️ Dropped {len(drop_cols)} columns: {', '.join(drop_cols[:3])}{'...' if len(drop_cols) > 3 else ''}")

    numeric_cols_renamed = [st.session_state.column_renames.get(c, c) for c in numeric_cols_orig]
    working_df[numeric_cols_renamed] = (
        working_df[numeric_cols_renamed]
        .apply(pd.to_numeric, errors="coerce")
        .astype("float64")
    )

    # Clean metadata columns: remove text after semicolon
    for meta_col in meta_cols:
        if meta_col in working_df.columns:
            working_df[meta_col] = working_df[meta_col].astype(str).str.split(';').str[0].str.strip()

    # Use user-selected species column if available
    species_col_to_use = species_col if species_col else None

    # Filter by species
    processed_df = working_df.copy()
    if species_col_to_use and species_tags:
        if species_col_to_use in processed_df.columns:
            # Check if column already contains plain species names
            sample_values = processed_df[species_col_to_use].dropna().astype(str).str.upper().str.strip().head(20)
            is_plain_species = sample_values.isin(["HUMAN", "YEAST", "ECOLI", "MOUSE", "UNKNOWN"]).mean() > 0.8
            
            if is_plain_species:
                # Column already has species names, use directly
                processed_df["_extracted_species"] = processed_df[species_col_to_use].astype(str).str.upper().str.strip()
            else:
                # Extract from protein ID format
                processed_df["_extracted_species"] = processed_df[species_col_to_use].apply(extract_species_from_protein_id)
            
            # Filter by species
            before_count = len(processed_df)
            processed_df = filter_by_species(processed_df, "_extracted_species", species_tags)
            after_count = len(processed_df)
            
            st.info(f"Filtered from {before_count:,} to {after_count:,} rows matching: {', '.join(species_tags)}")

    st.markdown("### Data preview")
    st.dataframe(processed_df.head(10), width="stretch", height=300)

    conditions = {col[0] for col in numeric_cols_renamed if col and col[0].isalpha()}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(processed_df):,}")
    c2.metric("Samples", len(numeric_cols_renamed))
    c3.metric("Conditions", len(conditions))
    c4.metric("Data level", "Peptide" if peptide_seq_col else "Protein")

    st.markdown("---")
    data_type = "peptide" if peptide_seq_col else "protein"
    existing_key = f"{data_type}_model"
    index_key = f"{data_type}_index_col"
    species_key = f"{data_type}_species_col"
    seq_key = f"{data_type}_seq_col"
    mask_key = f"{data_type}_missing_mask"

    if st.session_state.get(existing_key) is not None:
        st.warning(f"{data_type.capitalize()} data already cached. Confirming will overwrite.")

    col_b1, col_b2, _ = st.columns([1, 1, 3])
    with col_b1:
        if st.button("Confirm & cache", type="primary"):
            # Build model with species subgroups cached
            model = build_msdata(
                processed_df, 
                numeric_cols_renamed, 
                species_col="_extracted_species",
                _version=2
            )

            st.session_state[existing_key] = model
            st.session_state[index_key] = pg_col
            st.session_state[species_key] = "_extracted_species"
            st.session_state[seq_key] = peptide_seq_col
            st.session_state[mask_key] = missing_mask

            reset_upload_state()
            
            # Show cache summary
            st.success(f"✅ Cached {data_type} data")
            
            # Auto-redirect only if both loaded
            if st.session_state.get("protein_model") and st.session_state.get("peptide_model"):
                st.switch_page("pages/2_EDA.py")
            else:
                st.rerun()

    with col_b2:
        if st.button("Cancel"):
            reset_upload_state()
            st.rerun()


config_fragment()

render_navigation(back_page="app.py", next_page="pages/2_EDA.py")
render_footer()
