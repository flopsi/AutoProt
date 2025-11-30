import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List
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
class MSData:
    raw: pd.DataFrame
    raw_filled: pd.DataFrame
    missing_count: int
    numeric_cols: List[str]
    transforms: TransformsCache


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


def extract_conditions(cols: list[str]) -> dict:
    condition_map = {}
    for col in cols:
        if col and col[0].isalpha():
            condition_map[col] = col[0]
        else:
            condition_map[col] = "X"
    return condition_map


def get_transform_data(model: MSData, transform_key: str) -> pd.DataFrame:
    """Get transformed data by key."""
    if transform_key == "log2":
        return model.transforms.log2
    elif transform_key == "log10":
        return model.transforms.log10
    elif transform_key == "sqrt":
        return model.transforms.sqrt
    elif transform_key == "cbrt":
        return model.transforms.cbrt
    elif transform_key == "yeo_johnson":
        return model.transforms.yeo_johnson
    elif transform_key == "quantile":
        return model.transforms.quantile
    else:
        return model.transforms.log2


def compute_cv_per_condition(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Compute CV% for each protein within each condition."""
    condition_map = extract_conditions(numeric_cols)
    
    conditions = {}
    for col in numeric_cols:
        cond = condition_map[col]
        conditions.setdefault(cond, []).append(col)
    
    cv_results = {}
    for cond, cols in conditions.items():
        if len(cols) >= 2:
            mean_vals = df[cols].mean(axis=1)
            std_vals = df[cols].std(axis=1)
            cv_results[f"CV_{cond}"] = (std_vals / mean_vals * 100).replace([np.inf, -np.inf], np.nan)
    
    return pd.DataFrame(cv_results, index=df.index)


def apply_filters(
    df: pd.DataFrame,
    model: MSData,
    numeric_cols: list[str],
    species_col: str,
    selected_species: list[str],
    min_peptides: int,
    cv_cutoff: float,
    max_missing_ratio: float,
    intensity_range: tuple,
    transform_key: str,
) -> pd.DataFrame:
    """Apply all filters sequentially."""
    filtered = df.copy()
    
    # Filter 1: Species
    if species_col and species_col in model.raw.columns and selected_species:
        species_mask = model.raw.loc[filtered.index, species_col].isin(selected_species)
        filtered = filtered[species_mask]
    
    # Filter 2: Min peptides (if column exists)
    if "Peptide_Count" in filtered.columns:
        filtered = filtered[filtered["Peptide_Count"] >= min_peptides]
    
    # Filter 3: CV cutoff
    if len(filtered) > 0:
        cv_data = compute_cv_per_condition(filtered[numeric_cols], numeric_cols)
        if not cv_data.empty:
            cv_mask = cv_data.min(axis=1) <= cv_cutoff
            filtered = filtered[cv_mask]
    
    # Filter 4: Max missing per condition
    if len(filtered) > 0:
        condition_map = extract_conditions(numeric_cols)
        conditions = {}
        for col in numeric_cols:
            cond = condition_map[col]
            conditions.setdefault(cond, []).append(col)
        
        max_missing_allowed = {}
        for cond, cols in conditions.items():
            max_missing_allowed[cond] = int(np.ceil(len(cols) * max_missing_ratio))
        
        valid_rows = []
        for idx, row in filtered.iterrows():
            keep = True
            for cond, cols in conditions.items():
                missing_count = (row[cols].isna() | (row[cols] <= 1.0)).sum()
                if missing_count > max_missing_allowed[cond]:
                    keep = False
                    break
            if keep:
                valid_rows.append(idx)
        
        filtered = filtered.loc[valid_rows] if valid_rows else pd.DataFrame()
    
    # Filter 5: Intensity range (on transformed data)
    if len(filtered) > 0 and intensity_range and intensity_range[0] < intensity_range[1]:
        transform_df = get_transform_data(model, transform_key)
        transform_df = transform_df.loc[filtered.index, numeric_cols]
        
        mean_intensity = transform_df.mean(axis=1)
        intensity_mask = (mean_intensity >= intensity_range[0]) & (mean_intensity <= intensity_range[1])
        filtered = filtered[intensity_mask]
    
    return filtered


def compute_stats(df: pd.DataFrame, model: MSData, numeric_cols: list[str], species_col: str) -> dict:
    """Compute quality metrics."""
    if df.empty:
        return {
            "n_proteins": 0,
            "species_counts": {},
            "cv_mean": np.nan,
            "cv_median": np.nan,
            "permanova_f": np.nan,
            "permanova_p": np.nan,
            "shapiro_w": np.nan,
            "shapiro_p": np.nan,
        }
    
    n_proteins = len(df)
    
    species_counts = {}
    if species_col and species_col in model.raw.columns:
        species_counts = model.raw.loc[df.index, species_col].value_counts().to_dict()
    
    # CV stats
    cv_data = compute_cv_per_condition(df[numeric_cols], numeric_cols)
    if not cv_data.empty:
        cv_flat = cv_data.values.flatten()
        cv_clean = cv_flat[~np.isnan(cv_flat)]
        cv_mean = cv_clean.mean() if len(cv_clean) > 0 else np.nan
        cv_median = np.median(cv_clean) if len(cv_clean) > 0 else np.nan
    else:
        cv_mean = np.nan
        cv_median = np.nan
    
    # PERMANOVA
    condition_map = extract_conditions(numeric_cols)
    conditions = np.array([condition_map[c] for c in numeric_cols])
    
    permanova_f = np.nan
    permanova_p = np.nan
    
    if len(np.unique(conditions)) >= 2 and len(df) >= 3:
        try:
            data = df[numeric_cols].T.values
            dist_matrix = squareform(pdist(data, metric="euclidean"))
            n = len(conditions)
            
            ss_total = np.sum(dist_matrix**2) / (2 * n)
            ss_within = 0
            for g in np.unique(conditions):
                mask = conditions == g
                if np.sum(mask) > 1:
                    ss_within += np.sum(dist_matrix[np.ix_(mask, mask)]**2) / (2 * np.sum(mask))
            
            ss_between = ss_total - ss_within
            df_between = len(np.unique(conditions)) - 1
            df_within = n - len(np.unique(conditions))
            
            if df_within > 0 and ss_within > 0:
                F = (ss_between / df_between) / (ss_within / df_within)
                f_perms = []
                for _ in range(999):
                    perm_cond = np.random.permutation(conditions)
                    ss_within_perm = 0
                    for g in np.unique(perm_cond):
                        mask = perm_cond == g
                        if np.sum(mask) > 1:
                            ss_within_perm += np.sum(dist_matrix[np.ix_(mask, mask)]**2) / (2 * np.sum(mask))
                    ss_between_perm = ss_total - ss_within_perm
                    if ss_within_perm > 0:
                        F_perm = (ss_between_perm / df_between) / (ss_within_perm / df_within)
                        f_perms.append(F_perm)
                
                p_val = (np.sum(np.array(f_perms) >= F) + 1) / (999 + 1) if len(f_perms) > 0 else np.nan
                permanova_f = F
                permanova_p = p_val
        except Exception:
            pass
    
    # Shapiro-Wilk
    shapiro_w = np.nan
    shapiro_p = np.nan
    
    try:
        mean_vals = df[numeric_cols].mean(axis=1).dropna()
        if len(mean_vals) >= 3:
            sample = np.random.choice(mean_vals, min(5000, len(mean_vals)), replace=False)
            W, p = shapiro(sample)
            shapiro_w = W
            shapiro_p = p
    except Exception:
        pass
    
    return {
        "n_proteins": n_proteins,
        "species_counts": species_counts,
        "cv_mean": cv_mean,
        "cv_median": cv_median,
        "permanova_f": permanova_f,
        "permanova_p": permanova_p,
        "shapiro_w": shapiro_w,
        "shapiro_p": shapiro_p,
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

# Initialize session state for filters
if "filter_min_peptides" not in st.session_state:
    st.session_state.filter_min_peptides = 1
if "filter_cv_cutoff" not in st.session_state:
    st.session_state.filter_cv_cutoff = 30.0
if "filter_max_missing_ratio" not in st.session_state:
    st.session_state.filter_max_missing_ratio = 0.34
if "filter_intensity_range" not in st.session_state:
    st.session_state.filter_intensity_range = None
if "filter_transform" not in st.session_state:
    st.session_state.filter_transform = "log2"
if "filter_species" not in st.session_state:
    st.session_state.filter_species = ["HUMAN", "ECOLI", "YEAST"]

# Compute initial stats
initial_stats = compute_stats(df_raw, protein_model, numeric_cols, protein_species_col)

# CONTAINER 1: Summary Stats (Before Filtering)
st.markdown("### Before Filtering")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Total Proteins", f"{initial_stats['n_proteins']:,}")

with col2:
    species_str = ", ".join([f"{s}:{initial_stats['species_counts'].get(s, 0)}" for s in SPECIES_ORDER if s in initial_stats['species_counts']])
    st.metric("Species Count", species_str if species_str else "N/A")

with col3:
    st.metric("Mean CV%", f"{initial_stats['cv_mean']:.1f}" if not np.isnan(initial_stats['cv_mean']) else "N/A")

with col4:
    st.metric("Median CV%", f"{initial_stats['cv_median']:.1f}" if not np.isnan(initial_stats['cv_median']) else "N/A")

with col5:
    st.metric("PERMANOVA F", f"{initial_stats['permanova_f']:.2f}" if not np.isnan(initial_stats['permanova_f']) else "N/A")

with col6:
    st.metric("Shapiro W", f"{initial_stats['shapiro_w']:.4f}" if not np.isnan(initial_stats['shapiro_w']) else "N/A")

st.markdown("---")

# CONTAINER 2: Species Selection
st.markdown("### Species Selection")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.checkbox("All species", value=True, key="all_species_cb"):
        st.session_state.filter_species = ["HUMAN", "ECOLI", "YEAST", "MOUSE"]

with col2:
    human_cb = st.checkbox("HUMAN", value="HUMAN" in st.session_state.filter_species, key="human_cb")
    if human_cb and "HUMAN" not in st.session_state.filter_species:
        st.session_state.filter_species.append("HUMAN")
    elif not human_cb and "HUMAN" in st.session_state.filter_species:
        st.session_state.filter_species.remove("HUMAN")

with col3:
    yeast_cb = st.checkbox("YEAST", value="YEAST" in st.session_state.filter_species, key="yeast_cb")
    if yeast_cb and "YEAST" not in st.session_state.filter_species:
        st.session_state.filter_species.append("YEAST")
    elif not yeast_cb and "YEAST" in st.session_state.filter_species:
        st.session_state.filter_species.remove("YEAST")

with col4:
    ecoli_cb = st.checkbox("ECOLI", value="ECOLI" in st.session_state.filter_species, key="ecoli_cb")
    if ecoli_cb and "ECOLI" not in st.session_state.filter_species:
        st.session_state.filter_species.append("ECOLI")
    elif not ecoli_cb and "ECOLI" in st.session_state.filter_species:
        st.session_state.filter_species.remove("ECOLI")

with col5:
    mouse_cb = st.checkbox("MOUSE", value="MOUSE" in st.session_state.filter_species, key="mouse_cb")
    if mouse_cb and "MOUSE" not in st.session_state.filter_species:
        st.session_state.filter_species.append("MOUSE")
    elif not mouse_cb and "MOUSE" in st.session_state.filter_species:
        st.session_state.filter_species.remove("MOUSE")

st.markdown("---")

# CONTAINER 3: Filters with Toggle Switches
st.markdown("### Filter Settings")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    use_peptides = st.checkbox("Min peptides/protein", value=True, key="use_peptides_cb")
    if use_peptides:
        st.session_state.filter_min_peptides = st.slider(
            "Peptides",
            min_value=1,
            max_value=10,
            value=st.session_state.filter_min_peptides,
            key="min_pep_slider"
        )
    else:
        st.session_state.filter_min_peptides = 1

with col2:
    use_cv = st.checkbox("CV% cutoff", value=True, key="use_cv_cb")
    if use_cv:
        st.session_state.filter_cv_cutoff = st.slider(
            "CV%",
            min_value=5.0,
            max_value=100.0,
            value=st.session_state.filter_cv_cutoff,
            step=5.0,
            key="cv_slider"
        )
    else:
        st.session_state.filter_cv_cutoff = 1000.0

with col3:
    use_missing = st.checkbox("Max missing %", value=True, key="use_missing_cb")
    if use_missing:
        ratio_pct = st.slider(
            "Missing %",
            min_value=0,
            max_value=100,
            value=int(st.session_state.filter_max_missing_ratio * 100),
            step=10,
            key="missing_slider"
        )
        st.session_state.filter_max_missing_ratio = ratio_pct / 100.0
    else:
        st.session_state.filter_max_missing_ratio = 1.0

with col4:
    st.session_state.filter_transform = st.selectbox(
        "Transformation",
        options=["log2", "log10", "sqrt", "cbrt", "yeo_johnson", "quantile"],
        format_func=lambda x: TRANSFORMS[x],
        index=0,
        key="transform_select"
    )

with col5:
    use_intensity = st.checkbox("Intensity range", value=False, key="use_intensity_cb")

with col6:
    st.write("")

st.markdown("---")

# CONTAINER 4: Intensity Histograms
st.markdown("### Intensity Distribution by Sample")

# Get transformed data
transform_data = get_transform_data(protein_model, st.session_state.filter_transform)

# Set intensity range slider (only if toggled on)
if use_intensity:
    min_intensity = transform_data[numeric_cols].min().min()
    max_intensity = transform_data[numeric_cols].max().max()
    
    st.session_state.filter_intensity_range = st.slider(
        "Select intensity range",
        min_value=float(min_intensity),
        max_value=float(max_intensity),
        value=(float(min_intensity), float(max_intensity)) if st.session_state.filter_intensity_range is None else st.session_state.filter_intensity_range,
        key="intensity_slider"
    )
else:
    st.session_state.filter_intensity_range = None

# Apply filters based on toggles
filtered_df = apply_filters(
    df_raw,
    protein_model,
    numeric_cols,
    protein_species_col,
    st.session_state.filter_species,
    st.session_state.filter_min_peptides if use_peptides else 1,
    st.session_state.filter_cv_cutoff if use_cv else 1000.0,
    st.session_state.filter_max_missing_ratio if use_missing else 1.0,
    st.session_state.filter_intensity_range if use_intensity else None,
    st.session_state.filter_transform,
)

# Get transformed data for filtered
if not filtered_df.empty:
    transform_data_filtered = get_transform_data(protein_model, st.session_state.filter_transform).loc[filtered_df.index, numeric_cols]
else:
    transform_data_filtered = pd.DataFrame()

# Create histograms (one per sample)
cols = st.columns(len(numeric_cols))

for i, sample in enumerate(numeric
