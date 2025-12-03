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
    initial_sidebar_state="expanded",
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
    species_col: str | None,
    selected_species: List[str],
    min_peptides: int,
    cv_cutoff: float,
    max_missing_ratio: float,
    intensity_range: tuple[float, float] | None,
    transform_key: str,
) -> pd.DataFrame:
    """Apply all filters sequentially."""
    filtered = df.copy()

    # Filter 1: Species
    if species_col and species_col in model.raw.columns and selected_species:
        species_mask = model.raw.loc[filtered.index, species_col].isin(selected_species)
        filtered = filtered[species_mask]

    if filtered.empty:
        return filtered

    # Filter 2: Min peptides (if column exists)
    if "Peptide_Count" in filtered.columns and min_peptides > 1:
        filtered = filtered[filtered["Peptide_Count"] >= min_peptides]

    if filtered.empty:
        return filtered

    # Filter 3: CV cutoff
    cv_data = compute_cv_per_condition(filtered[numeric_cols], numeric_cols)
    if not cv_data.empty and cv_cutoff < 1000:
        cv_mask = cv_data.min(axis=1) <= cv_cutoff
        filtered = filtered[cv_mask]

    if filtered.empty:
        return filtered

    # Filter 4: Max missing per condition
    if max_missing_ratio < 1.0:
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

    # Filter 5: Intensity range (on transformed data)
    if intensity_range is not None and intensity_range[0] < intensity_range[1]:
        transform_df = get_transform_data(model, transform_key).loc[filtered.index, numeric_cols]
        mean_intensity = transform_df.mean(axis=1)
        intensity_mask = (mean_intensity >= intensity_range[0]) & (mean_intensity <= intensity_range[1])
        filtered = filtered[intensity_mask]

    return filtered


def compute_stats(
    df: pd.DataFrame,
    model: MSData,
    numeric_cols: List[str],
    species_col: str | None,
) -> dict:
    """Compute quality metrics for the given subset of proteins."""
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

    # Species counts
    if species_col and species_col in model.raw.columns:
        species_counts = model.raw.loc[df.index, species_col].value_counts().to_dict()
    else:
        species_counts = {}

    # CV stats
    cv_data = compute_cv_per_condition(df[numeric_cols], numeric_cols)
    if not cv_data.empty:
        cv_clean = cv_data.to_numpy().ravel()
        cv_clean = cv_clean[~np.isnan(cv_clean)]
        cv_mean = cv_clean.mean() if cv_clean.size else np.nan
        cv_median = np.median(cv_clean) if cv_clean.size else np.nan
    else:
        cv_mean = cv_median = np.nan

    # PERMANOVA
    condition_map = extract_conditions(numeric_cols)
    conditions = np.array([condition_map[c] for c in numeric_cols])

    permanova_f = np.nan
    permanova_p = np.nan

    unique_conds = np.unique(conditions)
    if unique_conds.size >= 2 and len(df) >= 3:
        try:
            data = df[numeric_cols].T.values
            dist_matrix = squareform(pdist(data, metric="euclidean"))
            n = len(conditions)

            ss_total = np.sum(dist_matrix**2) / (2 * n)
            ss_within = 0.0
            for g in unique_conds:
                mask = conditions == g
                if mask.sum() > 1:
                    ss_within += np.sum(dist_matrix[np.ix_(mask, mask)]**2) / (2 * mask.sum())

            ss_between = ss_total - ss_within
            df_between = unique_conds.size - 1
            df_within = n - unique_conds.size

            if df_within > 0 and ss_within > 0:
                F = (ss_between / df_between) / (ss_within / df_within)
                f_perms = []
                for _ in range(999):
                    perm_cond = np.random.permutation(conditions)
                    ss_within_perm = 0.0
                    for g in np.unique(perm_cond):
                        mask = perm_cond == g
                        if mask.sum() > 1:
                            ss_within_perm += (
                                np.sum(dist_matrix[np.ix_(mask, mask)]**2) / (2 * mask.sum())
                            )
                    ss_between_perm = ss_total - ss_within_perm
                    if ss_within_perm > 0:
                        F_perm = (ss_between_perm / df_between) / (ss_within_perm / df_within)
                        f_perms.append(F_perm)

                f_perms = np.array(f_perms)
                p_val = (np.sum(f_perms >= F) + 1) / (f_perms.size + 1) if f_perms.size else np.nan
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
            sample = np.random.choice(mean_vals, size=min(5000, len(mean_vals)), replace=False)
            W, p = shapiro(sample)
            shapiro_w, shapiro_p = W, p
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

# ========== SIDEBAR: Filter Settings ==========
with st.sidebar:
    st.markdown("## 🎛️ Filter Settings")
    
    # Species selection
    st.markdown("### Species")
    all_species = st.checkbox("All species", value=False, key="filter_all_species")
    
    if all_species:
        selected_species = ["HUMAN", "ECOLI", "YEAST", "MOUSE"]
        st.multiselect(
            "Selected species",
            options=["HUMAN", "ECOLI", "YEAST", "MOUSE"],
            default=selected_species,
            disabled=True,
            key="filter_species_display"
        )
    else:
        selected_species = st.multiselect(
            "Select species",
            options=["HUMAN", "ECOLI", "YEAST", "MOUSE"],
            default=["HUMAN", "ECOLI", "YEAST"],
            key="filter_species"
        )
    
    st.markdown("---")
    
    # Min peptides filter
    st.markdown("### Min Peptides/Protein")
    use_min_peptides = st.checkbox("Enable", value=True, key="filter_use_peptides")
    min_peptides = st.slider(
        "Min peptides",
        min_value=1,
        max_value=10,
        value=1,
        disabled=not use_min_peptides,
        key="filter_min_peptides"
    )
    
    st.markdown("---")
    
    # CV filter
    st.markdown("### CV% Cutoff")
    use_cv = st.checkbox("Enable", value=True, key="filter_use_cv")
    cv_cutoff = st.slider(
        "Max CV%",
        min_value=0,
        max_value=100,
        value=30,
        step=5,
        disabled=not use_cv,
        key="filter_cv"
    )
    
    st.markdown("---")
    
    # Missing data filter
    st.markdown("### Max Missing %")
    use_missing = st.checkbox("Enable", value=True, key="filter_use_missing")
    max_missing_pct = st.slider(
        "Max missing per condition",
        min_value=0,
        max_value=100,
        value=34,
        step=10,
        disabled=not use_missing,
        key="filter_missing"
    )
    
    st.markdown("---")
    
    # Transformation
    st.markdown("### Transformation")
    transform_key = st.selectbox(
        "Select transformation",
        options=list(TRANSFORMS.keys()),
        format_func=lambda x: TRANSFORMS[x],
        index=0,
        key="filter_transform"
    )
    
    st.markdown("---")
    
    # Intensity range
    st.markdown("### Intensity Range")
    use_intensity = st.checkbox("Enable", value=False, key="filter_use_intensity")
    
    # Derive effective values
    min_peptides = min_peptides if use_min_peptides else 1
    cv_cutoff = cv_cutoff if use_cv else 1000.0
    max_missing_ratio = max_missing_pct / 100.0 if use_missing else 1.0
    
    st.markdown("---")
    
    # Active filters summary
    st.markdown("### Active Filters")
    st.caption(f"**Species:** {', '.join(selected_species) if selected_species else 'None'}")
    
    active_filters = []
    if use_min_peptides:
        active_filters.append(f"Min peptides: {min_peptides}")
    if use_cv:
        active_filters.append(f"CV <{cv_cutoff:.0f}%")
    if use_missing:
        active_filters.append(f"Max missing: {max_missing_pct}%")
    
    if active_filters:
        for f in active_filters:
            st.caption(f"• {f}")
    else:
        st.caption("No filters active")


# ========== MAIN: Stats and Visualizations ==========

# Initial stats (unfiltered)
initial_stats = compute_stats(df_raw, protein_model, numeric_cols, protein_species_col)

# CONTAINER 1: Summary Stats (Before Filtering)
st.markdown("### Before Filtering")
c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.metric("Total Proteins", f"{initial_stats['n_proteins']:,}")

with c2:
    species_str = ", ".join(
        f"{s}:{initial_stats['species_counts'].get(s, 0)}"
        for s in SPECIES_ORDER
        if s in initial_stats["species_counts"]
    )
    st.metric("Species Count", species_str or "N/A")

with c3:
    st.metric(
        "Mean CV%",
        f"{initial_stats['cv_mean']:.1f}" if not np.isnan(initial_stats["cv_mean"]) else "N/A",
    )

with c4:
    st.metric(
        "Median CV%",
        f"{initial_stats['cv_median']:.1f}" if not np.isnan(initial_stats["cv_median"]) else "N/A",
    )

with c5:
    st.metric(
        "PERMANOVA F",
        f"{initial_stats['permanova_f']:.2f}" if not np.isnan(initial_stats["permanova_f"]) else "N/A",
    )

with c6:
    st.metric(
        "Shapiro W",
        f"{initial_stats['shapiro_w']:.4f}" if not np.isnan(initial_stats["shapiro_w"]) else "N/A",
    )

st.markdown("---")

# CONTAINER 2: Intensity Distribution
st.markdown("### Intensity Distribution by Sample")

# Get transformed data
transform_data = get_transform_data(protein_model, transform_key)

# Set intensity range slider (only if toggled on)
if use_intensity:
    min_intensity = float(transform_data[numeric_cols].min().min())
    max_intensity = float(transform_data[numeric_cols].max().max())
    
    intensity_range = st.slider(
        "Select intensity range",
        min_value=min_intensity,
        max_value=max_intensity,
        value=(min_intensity, max_intensity),
        key="intensity_slider"
    )
else:
    intensity_range = None

# Apply filters
filtered_df = apply_filters(
    df_raw,
    protein_model,
    numeric_cols,
    protein_species_col,
    selected_species,
    min_peptides,
    cv_cutoff,
    max_missing_ratio,
    intensity_range,
    transform_key,
)

# Get transformed data for filtered
if not filtered_df.empty:
    transform_data_filtered = get_transform_data(protein_model, transform_key).loc[filtered_df.index, numeric_cols]
else:
    transform_data_filtered = pd.DataFrame()

# Create histograms in 3x2 grid
n_cols = 3
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

for row_idx in range(n_rows):
    cols = st.columns(n_cols)
    for col_idx in range(n_cols):
        sample_idx = row_idx * n_cols + col_idx
        
        if sample_idx >= len(numeric_cols):
            break
        
        sample = numeric_cols[sample_idx]
        
        with cols[col_idx]:
            fig = go.Figure()
            
            sample_data = (
                transform_data_filtered[sample].dropna()
                if not transform_data_filtered.empty
                else pd.Series(dtype=float)
            )
            
            if not sample_data.empty:
                mean_val = sample_data.mean()
                std_val = sample_data.std()
                
                # Histogram
                fig.add_trace(go.Histogram(
                    x=sample_data,
                    name="Distribution",
                    nbinsx=50,
                    marker_color="rgba(135, 206, 235, 0.7)",
                    showlegend=False,
                ))
                
                # Mean line
                fig.add_vline(
                    x=mean_val,
                    line_dash="solid",
                    line_color="red",
                    line_width=2,
                    annotation_text=f"μ={mean_val:.1f}",
                    annotation_position="top",
                )
                
                # Std dev shade
                fig.add_vrect(
                    x0=mean_val - std_val,
                    x1=mean_val + std_val,
                    fillcolor="red",
                    opacity=0.1,
                    layer="below",
                    line_width=0,
                )
                
                fig.update_layout(
                    title=f"{sample} (n={len(sample_data)})",
                    xaxis_title=TRANSFORMS[transform_key],
                    yaxis_title="Count",
                    height=350,
                    plot_bgcolor="#FFFFFF",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Arial", size=10, color="#54585A"),
                    showlegend=False,
                    margin=dict(l=40, r=40, t=60, b=40),
                )
            else:
                fig.add_annotation(text="No data after filtering", showarrow=False)
                fig.update_layout(
                    height=350,
                    plot_bgcolor="#FFFFFF",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Arial", size=10, color="#54585A"),
                )
            
            st.plotly_chart(fig, use_container_width=True, key=f"hist_{sample}_{sample_idx}")

st.markdown("---")

# CONTAINER 3: After Filtering Stats
st.markdown("### After Filtering")

# Only compute stats when button clicked
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("📊 Calculate Stats", type="primary", key="calc_stats_btn"):
        st.session_state.compute_stats_now = True

with col2:
    if st.button("💾 Export Filtered Data", key="export_btn"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="filtered_proteins.csv",
            mime="text/csv",
        )

if st.session_state.get("compute_stats_now", False):
    with st.spinner("Computing stats..."):
        filtered_stats = compute_stats(filtered_df, protein_model, numeric_cols, protein_species_col)
    
    st.session_state.compute_stats_now = False
    
    def get_arrow(before, after, higher_is_better=True):
        if np.isnan(before) or np.isnan(after):
            return "→"
        diff = after - before
        if diff > 0:
            return "↑" if higher_is_better else "↓"
        if diff < 0:
            return "↓" if higher_is_better else "↑"
        return "→"

    m1, m2, m3, m4, m5, m6 = st.columns(6)

    with m1:
        arrow = get_arrow(initial_stats["n_proteins"], filtered_stats["n_proteins"], True)
        st.metric(f"Proteins {arrow}", f"{filtered_stats['n_proteins']:,}")

    with m2:
        species_str = ", ".join(
            f"{s}:{filtered_stats['species_counts'].get(s, 0)}"
            for s in SPECIES_ORDER
            if s in filtered_stats["species_counts"]
        )
        st.metric("Species Count", species_str or "N/A")

    with m3:
        arrow = get_arrow(initial_stats["cv_mean"], filtered_stats["cv_mean"], higher_is_better=False)
        st.metric(
            f"Mean CV% {arrow}",
            f"{filtered_stats['cv_mean']:.1f}" if not np.isnan(filtered_stats["cv_mean"]) else "N/A",
        )

    with m4:
        arrow = get_arrow(initial_stats["cv_median"], filtered_stats["cv_median"], higher_is_better=False)
        st.metric(
            f"Median CV% {arrow}",
            f"{filtered_stats['cv_median']:.1f}" if not np.isnan(filtered_stats["cv_median"]) else "N/A",
        )

    with m5:
        arrow = get_arrow(initial_stats["permanova_f"], filtered_stats["permanova_f"], True)
        st.metric(
            f"PERMANOVA F {arrow}",
            f"{filtered_stats['permanova_f']:.2f}" if not np.isnan(filtered_stats["permanova_f"]) else "N/A",
        )

    with m6:
        arrow = get_arrow(initial_stats["shapiro_w"], filtered_stats["shapiro_w"], True)
        st.metric(
            f"Shapiro W {arrow}",
            f"{filtered_stats['shapiro_w']:.4f}" if not np.isnan(filtered_stats["shapiro_w"]) else "N/A",
        )
else:
    st.info("Click 'Calculate Stats' to compute quality metrics for filtered data.")

render_navigation(back_page="pages/3_Preprocessing.py", next_page=None)
render_footer()
