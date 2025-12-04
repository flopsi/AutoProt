import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Dict
from scipy.stats import shapiro
from scipy.spatial.distance import pdist, squareform

from components import inject_custom_css, render_header, render_navigation, render_footer, COLORS

st.set_page_config(
    page_title="Filtering | Thermo Fisher Scientific",
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
        if not species_list:
            return self.all_species
        if set(species_list) == {"HUMAN", "YEAST", "ECOLI", "MOUSE"}:
            return self.all_species
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
    species_subgroups: FilteredSubgroups
    species_col: str


SPECIES_COLORS = {
    "HUMAN": "#87CEEB",
    "ECOLI": "#008B8B",
    "YEAST": "#FF8C00",
    "MOUSE": "#9370DB",
    "UNKNOWN": "#808080",
}

SPECIES_ORDER = ["HUMAN", "ECOLI", "YEAST", "MOUSE", "UNKNOWN"]

TRANSFORMS = {
    "log2": "log2",
    "log10": "log10",
    "sqrt": "sqrt",
    "cbrt": "cbrt",
    "yeo_johnson": "Yeo-Johnson",
    "quantile": "Quantile Norm",
}


def extract_conditions(cols: List[str]) -> Dict[str, str]:
    """Map each column to a condition code based on its first character."""
    return {col: (col[0] if col and col[0].isalpha() else "X") for col in cols}


def build_condition_groups(numeric_cols: List[str]) -> Dict[str, List[str]]:
    """Group numeric columns by condition letter."""
    condition_map = extract_conditions(numeric_cols)
    groups: Dict[str, List[str]] = {}
    for col in numeric_cols:
        groups.setdefault(condition_map[col], []).append(col)
    return groups


def get_transform_data(model: MSData, transform_key: str) -> pd.DataFrame:
    """Get transformed data by key, defaulting to log2."""
    return getattr(model.transforms, transform_key, model.transforms.log2)


def compute_cv_per_condition(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Compute CV% for each protein within each condition."""
    condition_groups = build_condition_groups(numeric_cols)
    cv_results = {}
    for cond, cols in condition_groups.items():
        if len(cols) < 2:
            continue
        mean_vals = df[cols].mean(axis=1)
        std_vals = df[cols].std(axis=1)
        cv = (std_vals / mean_vals * 100).replace([np.inf, -np.inf], np.nan)
        cv_results[f"CV_{cond}"] = cv
    return pd.DataFrame(cv_results, index=df.index)


def apply_filters(
    df: pd.DataFrame,
    model: MSData,
    numeric_cols: List[str],
    selected_species: List[str],
    min_peptides: int,
    cv_cutoff: float,
    max_missing_ratio: float,
    sd_range: tuple[float, float] | None,
    apply_sd: bool,
    transform_key: str,
    apply_species: bool = True,
    apply_min_pep: bool = True,
    apply_cv: bool = True,
    apply_missing: bool = True,
) -> pd.DataFrame:
    """Apply filters sequentially with enable/disable flags."""
    
    filtered = df.copy()

    # Filter 1: Species (if enabled)
    if apply_species and selected_species:
        filtered = model.species_subgroups.get(selected_species)
    elif apply_species:
        filtered = model.species_subgroups.all_species
    
    if filtered.empty:
        return filtered

    # Filter 2: Min peptides (if enabled)
    if apply_min_pep and "Peptide_Count" in filtered.columns and min_peptides > 1:
        filtered = filtered[filtered["Peptide_Count"] >= min_peptides]

    if filtered.empty:
        return filtered

    # Filter 3: CV cutoff (if enabled)
    if apply_cv and cv_cutoff < 1000:
        cv_data = compute_cv_per_condition(filtered[numeric_cols], numeric_cols)
        if not cv_data.empty:
            cv_mask = cv_data.min(axis=1) <= cv_cutoff
            filtered = filtered[cv_mask]

    if filtered.empty:
        return filtered

    # Filter 4: Max missing per condition (if enabled)
    if apply_missing and max_missing_ratio < 1.0:
        condition_groups = build_condition_groups(numeric_cols)
        max_missing_allowed = {
            cond: int(np.ceil(len(cols) * max_missing_ratio))
            for cond, cols in condition_groups.items()
        }

        valid_idx = []
        for idx, row in filtered.iterrows():
            keep = True
            for cond, cols in condition_groups.items():
                missing_count = (row[cols].isna() | (row[cols] <= 1.0)).sum()
                if missing_count > max_missing_allowed[cond]:
                    keep = False
                    break
            if keep:
                valid_idx.append(idx)

        filtered = filtered.loc[valid_idx] if valid_idx else pd.DataFrame(index=df.index, columns=df.columns)

    if filtered.empty:
        return filtered

    # Filter 5: SD filter (if enabled)
    if apply_sd and sd_range is not None:
        transform_data = get_transform_data(model, transform_key).loc[filtered.index, numeric_cols]
        
        sd_mask = pd.Series(True, index=filtered.index)
        for col in numeric_cols:
            col_data = transform_data[col].dropna()
            if len(col_data) > 0:
                mean_val = col_data.mean()
                std_val = col_data.std()
                lower_bound = mean_val - (sd_range[0] * std_val)
                upper_bound = mean_val + (sd_range[1] * std_val)
                
                col_mask = (transform_data[col] >= lower_bound) & (transform_data[col] <= upper_bound)
                sd_mask &= col_mask
        
        filtered = filtered[sd_mask]

    return filtered


def compute_stats(df: pd.DataFrame, model: MSData, numeric_cols: List[str], species_col: str | None) -> dict:
    """Compute quality metrics."""
    if df.empty:
        return {
            "n_proteins": 0,
            "species_counts": {},
            "cv_mean": np.nan,
            "cv_median": np.nan,
        }

    n_proteins = len(df)
    
    if species_col and species_col in model.raw.columns:
        species_counts = model.raw.loc[df.index, species_col].value_counts().to_dict()
    else:
        species_counts = {}

    cv_data = compute_cv_per_condition(df[numeric_cols], numeric_cols)
    if not cv_data.empty:
        cv_clean = cv_data.to_numpy().ravel()
        cv_clean = cv_clean[~np.isnan(cv_clean)]
        cv_mean = cv_clean.mean() if cv_clean.size else np.nan
        cv_median = np.median(cv_clean) if cv_clean.size else np.nan
    else:
        cv_mean = cv_median = np.nan

    return {
        "n_proteins": n_proteins,
        "species_counts": species_counts,
        "cv_mean": cv_mean,
        "cv_median": cv_median,
    }


st.markdown("## Protein-level Filtering & QC")

protein_model: MSData | None = st.session_state.get("protein_model")
protein_idx = st.session_state.get("protein_index_col")
protein_species_col = st.session_state.get("protein_species_col")

if protein_model is None:
    st.warning("No protein data cached. Please upload data on the Data Upload page first.")
    render_navigation(back_page="pages/3_Preprocessing.py", next_page=None)
    render_footer()
    st.stop()

numeric_cols = protein_model.numeric_cols
df_raw = protein_model.raw_filled[numeric_cols].copy()

# ========== FILTER CONTROLS (MAIN PAGE) ==========
st.markdown("### 🎚️ Filter Configuration")

# Initialize filter enable states
if "filter_enable_species" not in st.session_state:
    st.session_state.filter_enable_species = True
if "filter_enable_min_pep" not in st.session_state:
    st.session_state.filter_enable_min_pep = True
if "filter_enable_cv" not in st.session_state:
    st.session_state.filter_enable_cv = False
if "filter_enable_missing" not in st.session_state:
    st.session_state.filter_enable_missing = False
if "filter_enable_sd" not in st.session_state:
    st.session_state.filter_enable_sd = False

# ========== FILTER 1: SPECIES ==========
st.markdown("#### 1️⃣ Species Selection")
col_f1_toggle, col_f1_space = st.columns([1, 4])

with col_f1_toggle:
    enable_species = st.checkbox(
        "Enable",
        value=st.session_state.filter_enable_species,
        key="filter_enable_species_cb",
        label_visibility="collapsed"
    )
    st.session_state.filter_enable_species = enable_species

if enable_species:
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
    with col_s1:
        species_human = st.checkbox("Human", value=True, key="filter_species_human")
    with col_s2:
        species_ecoli = st.checkbox("E.coli", value=True, key="filter_species_ecoli")
    with col_s3:
        species_yeast = st.checkbox("Yeast", value=True, key="filter_species_yeast")
    with col_s4:
        species_mouse = st.checkbox("Mouse", value=False, key="filter_species_mouse")
    
    selected_species = []
    if species_human:
        selected_species.append("HUMAN")
    if species_ecoli:
        selected_species.append("ECOLI")
    if species_yeast:
        selected_species.append("YEAST")
    if species_mouse:
        selected_species.append("MOUSE")
else:
    selected_species = ["HUMAN", "ECOLI", "YEAST", "MOUSE"]
    st.caption("ℹ️ Species filter disabled - all species included")

st.markdown("---")

# ========== FILTER 2: MIN PEPTIDES ==========
st.markdown("#### 2️⃣ Minimum Peptides per Protein")
col_f2_toggle, col_f2_input = st.columns([1, 2])

with col_f2_toggle:
    enable_min_pep = st.checkbox(
        "Enable",
        value=st.session_state.filter_enable_min_pep,
        key="filter_enable_min_pep_cb",
        label_visibility="collapsed"
    )
    st.session_state.filter_enable_min_pep = enable_min_pep

with col_f2_input:
    min_peptides = st.number_input(
        "Min peptides",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        key="filter_min_peptides",
        disabled=not enable_min_pep,
    )

if not enable_min_pep:
    st.caption("ℹ️ Min peptides filter disabled - no minimum applied")
else:
    st.caption(f"Only keep proteins with ≥{min_peptides} peptides")

st.markdown("---")

# ========== FILTER 3: CV CUTOFF ==========
st.markdown("#### 3️⃣ CV% (Coefficient of Variation) Cutoff")
col_f3_toggle, col_f3_input = st.columns([1, 2])

with col_f3_toggle:
    enable_cv = st.checkbox(
        "Enable",
        value=st.session_state.filter_enable_cv,
        key="filter_enable_cv_cb",
        label_visibility="collapsed"
    )
    st.session_state.filter_enable_cv = enable_cv

with col_f3_input:
    cv_cutoff = st.number_input(
        "Max CV%",
        min_value=0,
        max_value=100,
        value=30,
        step=5,
        key="filter_cv",
        disabled=not enable_cv,
    )

if not enable_cv:
    st.caption("ℹ️ CV filter disabled - no cutoff applied")
else:
    st.caption(f"Only keep proteins with min(CV) ≤ {cv_cutoff}% in any condition")

st.markdown("---")

# ========== FILTER 4: MISSING DATA ==========
st.markdown("#### 4️⃣ Maximum Missing Data per Condition")
col_f4_toggle, col_f4_input = st.columns([1, 2])

with col_f4_toggle:
    enable_missing = st.checkbox(
        "Enable",
        value=st.session_state.filter_enable_missing,
        key="filter_enable_missing_cb",
        label_visibility="collapsed"
    )
    st.session_state.filter_enable_missing = enable_missing

with col_f4_input:
    max_missing_pct = st.number_input(
        "Max missing %",
        min_value=0,
        max_value=100,
        value=34,
        step=5,
        key="filter_missing",
        disabled=not enable_missing,
    )

if not enable_missing:
    st.caption("ℹ️ Missing data filter disabled - no limit applied")
else:
    st.caption(f"Only keep proteins with ≤{max_missing_pct}% missing per condition")

st.markdown("---")

# ========== FILTER 5: SD RANGE ==========
st.markdown("#### 5️⃣ Standard Deviation Range Filter")
col_f5_toggle, col_f5_space = st.columns([1, 4])

with col_f5_toggle:
    enable_sd = st.checkbox(
        "Enable",
        value=st.session_state.filter_enable_sd,
        key="filter_enable_sd_cb",
        label_visibility="collapsed"
    )
    st.session_state.filter_enable_sd = enable_sd

if enable_sd:
    col_sd1, col_sd2 = st.columns(2)
    with col_sd1:
        sd_min = st.number_input(
            "Min SD (σ below mean)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
            key="sd_min_input",
        )
    with col_sd2:
        sd_max = st.number_input(
            "Max SD (σ above mean)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
            key="sd_max_input",
        )
    st.caption(f"Keep values within μ - {sd_min}σ to μ + {sd_max}σ")
else:
    sd_min = 2.0
    sd_max = 2.0
    st.caption("ℹ️ SD range filter disabled - no filtering applied")

st.markdown("---")

# ========== TRANSFORMATION SELECTION ==========
st.markdown("#### 🔄 Transformation")
transform_key = st.selectbox(
    "Select transformation",
    options=list(TRANSFORMS.keys()),
    format_func=lambda x: TRANSFORMS[x],
    index=0,
    key="filter_transform"
)
st.caption("Used for CV and SD range calculations")

st.markdown("---")

# ========== DERIVE EFFECTIVE FILTER VALUES ==========
cv_cutoff_val = cv_cutoff if enable_cv else 1000.0
max_missing_ratio = max_missing_pct / 100.0 if enable_missing else 1.0
sd_range = (sd_min, sd_max) if enable_sd else None

# ========== APPLY FILTERS AND COMPUTE STATS ==========
initial_stats = compute_stats(df_raw, protein_model, numeric_cols, protein_species_col)

filtered_df = apply_filters(
    df_raw,
    protein_model,
    numeric_cols,
    selected_species,
    min_peptides,
    cv_cutoff_val,
    max_missing_ratio,
    sd_range,
    enable_sd,
    transform_key,
    apply_species=enable_species,
    apply_min_pep=enable_min_pep,
    apply_cv=enable_cv,
    apply_missing=enable_missing,
)

filtered_stats = compute_stats(filtered_df, protein_model, numeric_cols, protein_species_col)

st.markdown("---")

# ========== SUMMARY STATISTICS ==========
st.markdown("### 📊 Filter Summary")

col_summ1, col_summ2, col_summ3, col_summ4 = st.columns(4)

with col_summ1:
    st.metric(
        "Before",
        f"{initial_stats['n_proteins']:,}",
        delta=f"{filtered_stats['n_proteins'] - initial_stats['n_proteins']:+,}"
    )

with col_summ2:
    st.metric(
        "After",
        f"{filtered_stats['n_proteins']:,}",
        delta=f"{filtered_stats['n_proteins'] / initial_stats['n_proteins'] * 100:.1f}%"
    )

with col_summ3:
    st.metric(
        "Mean CV% (Before)",
        f"{initial_stats['cv_mean']:.1f}" if not np.isnan(initial_stats["cv_mean"]) else "N/A",
        delta=f"{filtered_stats['cv_mean'] - initial_stats['cv_mean']:.1f}" if not (np.isnan(filtered_stats['cv_mean']) or np.isnan(initial_stats['cv_mean'])) else None
    )

with col_summ4:
    st.metric(
        "Mean CV% (After)",
        f"{filtered_stats['cv_mean']:.1f}" if not np.isnan(filtered_stats["cv_mean"]) else "N/A",
    )

st.markdown("---")

# ========== ACTIVE FILTERS BADGE ==========
st.markdown("### ✓ Active Filters")

active_filters = []
if enable_species and selected_species:
    active_filters.append(f"Species: {', '.join(selected_species)}")
if enable_min_pep:
    active_filters.append(f"Min peptides: {min_peptides}")
if enable_cv:
    active_filters.append(f"CV <{cv_cutoff}%")
if enable_missing:
    active_filters.append(f"Missing <{max_missing_pct}%")
if enable_sd:
    active_filters.append(f"SD: μ±{sd_min:.1f}σ to μ±{sd_max:.1f}σ")

if active_filters:
    for i, f in enumerate(active_filters, 1):
        st.caption(f"✓ {f}")
else:
    st.caption("No filters active - using all proteins")

st.markdown("---")

# ========== STORE FILTERED DATASET ==========
col_store1, col_store2 = st.columns([1, 3])

with col_store1:
    if st.session_state.filter_state["configured"]:
        filtered_df = st.session_state.filter_state["filtered_data"]
        filter_params = st.session_state.filter_state["filter_params"]
        filtered_stats = st.session_state.filter_state["filtered_stats"]

        if st.button("💾 Store for Analysis", type="primary", key="store_filtered_btn"):
            if not filtered_df.empty:
                st.session_state.last_filtered_data = filtered_df.copy()
                st.session_state.last_filtered_params = filter_params
                st.success(f"✅ Stored {len(filtered_df):,} proteins for analysis!")
            else:
                st.error("❌ No proteins after filtering.")
    else:
        st.button("💾 Store for Analysis", type="primary",
                  key="store_filtered_placeholder", disabled=True)

with col_store2:
    if st.session_state.filter_state["configured"]:
        filtered_stats = st.session_state.filter_state["filtered_stats"]
        st.metric("Proteins Ready", f"{filtered_stats['n_proteins']:,}")

st.markdown("---")

# ========== EXPORT FILTERED DATA ==========
if st.session_state.filter_state["configured"]:
    filtered_df = st.session_state.filter_state["filtered_data"]
    if not filtered_df.empty:
        col_exp1, col_exp2 = st.columns([1, 1])

        with col_exp1:
            csv = filtered_df.to_csv(index=True)
            st.download_button(
                label="💾 Export Filtered Data",
                data=csv,
                file_name="filtered_proteins.csv",
                mime="text/csv",
                key="download_filtered_csv",
            )

        with col_exp2:
            st.caption("Export current filtered dataset as CSV")

render_navigation(back_page="pages/3_Preprocessing.py", next_page=None)
render_footer()
