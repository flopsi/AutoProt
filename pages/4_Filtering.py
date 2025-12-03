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
    intensity_range: tuple[float, float] | None,
    transform_key: str,
) -> pd.DataFrame:
    """Apply all filters sequentially using cached species subgroups."""
    
    # Filter 1: Species (use cached subgroups - MUCH FASTER)
    if selected_species:
        filtered = model.species_subgroups.get(selected_species)
    else:
        filtered = df.copy()

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
    st.markdown("### Species Selection")
    
    species_human = st.checkbox("HUMAN", value=True, key="filter_species_human")
    species_ecoli = st.checkbox("ECOLI", value=True, key="filter_species_ecoli")
    species_yeast = st.checkbox("YEAST", value=True, key="filter_species_yeast")
    species_mouse = st.checkbox("MOUSE", value=False, key="filter_species_mouse")
    
    # Build selected species list
    selected_species = []
    if species_human:
        selected_species.append("HUMAN")
    if species_ecoli:
        selected_species.append("ECOLI")
    if species_yeast:
        selected_species.append("YEAST")
    if species_mouse:
        selected_species.append("MOUSE")
    
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
    min_peptides_val = min_peptides if use_min_peptides else 1
    cv_cutoff_val = cv_cutoff if use_cv else 1000.0
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
st.markdown("### Summary Statistics")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Total Proteins", f"{initial_stats['n_proteins']:,}")

with c2:
    st.metric(
        "Mean CV%",
        f"{initial_stats['cv_mean']:.1f}" if not np.isnan(initial_stats["cv_mean"]) else "N/A",
    )

with c3:
    st.metric(
        "Median CV%",
        f"{initial_stats['cv_median']:.1f}" if not np.isnan(initial_stats["cv_median"]) else "N/A",
    )

with c4:
    st.metric(
        "PERMANOVA F",
        f"{initial_stats['permanova_f']:.2f}" if not np.isnan(initial_stats["permanova_f"]) else "N/A",
    )

st.markdown("---")

# CONTAINER 2: Intensity Distribution
st.markdown("### Intensity Distribution by Sample")

# Replace the intensity distribution section with this:

# CONTAINER 2: Intensity Distribution
st.markdown("### Intensity Distribution by Sample")

# Get transformed data
transform_data = get_transform_data(protein_model, transform_key)

# Standard deviation filter
st.markdown("#### Standard Deviation Filter")
col_sd1, col_sd2 = st.columns([3, 1])

with col_sd1:
    sd_range = st.slider(
        "Filter by standard deviations from mean (per sample)",
        min_value=0.5,
        max_value=5.0,
        value=(0.5, 3.0),
        step=0.5,
        key="sd_range_slider",
        help="Values outside this range will be highlighted. Only filtered when checkbox is enabled."
    )

with col_sd2:
    apply_sd_filter = st.checkbox(
        "Apply filter",
        value=False,
        key="apply_sd_filter",
        help="When enabled, actually filters the data. Otherwise just shows what would be filtered."
    )

# Apply filters (including SD filter if enabled)
if apply_sd_filter:
    # First apply other filters
    filtered_df_temp = apply_filters(
        df_raw,
        protein_model,
        numeric_cols,
        selected_species,
        min_peptides_val,
        cv_cutoff_val,
        max_missing_ratio,
        None,  # No intensity range yet
        transform_key,
    )
    
    # Then apply SD filter on transformed data
    if not filtered_df_temp.empty:
        transform_temp = get_transform_data(protein_model, transform_key).loc[filtered_df_temp.index, numeric_cols]
        
        # Filter each sample independently
        sd_mask = pd.Series(True, index=filtered_df_temp.index)
        for col in numeric_cols:
            col_data = transform_temp[col].dropna()
            if len(col_data) > 0:
                mean_val = col_data.mean()
                std_val = col_data.std()
                lower_bound = mean_val - (sd_range[0] * std_val)
                upper_bound = mean_val + (sd_range[1] * std_val)
                
                # Mark rows outside range
                col_mask = (transform_temp[col] >= lower_bound) & (transform_temp[col] <= upper_bound)
                sd_mask &= col_mask
        
        filtered_df = filtered_df_temp[sd_mask]
    else:
        filtered_df = filtered_df_temp
else:
    # Just apply other filters, not SD filter
    filtered_df = apply_filters(
        df_raw,
        protein_model,
        numeric_cols,
        selected_species,
        min_peptides_val,
        cv_cutoff_val,
        max_missing_ratio,
        None,
        transform_key,
    )

# Get transformed data for visualization
if not filtered_df.empty:
    transform_data_filtered = get_transform_data(protein_model, transform_key).loc[filtered_df.index, numeric_cols]
else:
    transform_data_filtered = pd.DataFrame()

# Create Vega-Lite histograms in 3-column grid
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
            sample_data = (
                transform_data_filtered[sample].dropna()
                if not transform_data_filtered.empty
                else pd.Series(dtype=float)
            )
            
            if not sample_data.empty:
                mean_val = sample_data.mean()
                std_val = sample_data.std()
                
                # Calculate SD bounds
                lower_bound = mean_val - (sd_range[0] * std_val)
                upper_bound = mean_val + (sd_range[1] * std_val)
                
                # Count filtered vs kept
                in_range = ((sample_data >= lower_bound) & (sample_data <= upper_bound)).sum()
                out_range = len(sample_data) - in_range
                
                # Prepare data for Vega-Lite
                chart_data = pd.DataFrame({
                    'value': sample_data.values,
                    'in_range': (sample_data >= lower_bound) & (sample_data <= upper_bound)
                })
                
                # Vega-Lite specification
                spec = {
                    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                    "width": "container",
                    "height": 300,
                    "title": {
                        "text": f"{sample} (n={len(sample_data)})",
                        "fontSize": 14
                    },
                    "data": {"values": chart_data.to_dict('records')},
                    "layer": [
                        {
                            "mark": {
                                "type": "bar",
                                "binSpacing": 1,
                                "stroke": "white",
                                "strokeWidth": 0.5
                            },
                            "encoding": {
                                "x": {
                                    "bin": {"maxbins": 50},
                                    "field": "value",
                                    "title": TRANSFORMS[transform_key],
                                    "axis": {"labelFontSize": 10, "titleFontSize": 11}
                                },
                                "y": {
                                    "aggregate": "count",
                                    "title": "Count",
                                    "axis": {"labelFontSize": 10, "titleFontSize": 11}
                                },
                                "color": {
                                    "field": "in_range",
                                    "scale": {
                                        "domain": [False, True],
                                        "range": ["#ff6b6b", "#87CEEB"]
                                    },
                                    "legend": None
                                },
                                "tooltip": [
                                    {"aggregate": "count", "title": "Count"}
                                ]
                            }
                        },
                        {
                            "mark": {"type": "rule", "color": "red", "strokeWidth": 2},
                            "encoding": {
                                "x": {"datum": mean_val}
                            }
                        },
                        {
                            "mark": {
                                "type": "rect",
                                "opacity": 0.15,
                                "color": "red"
                            },
                            "encoding": {
                                "x": {"datum": mean_val - std_val},
                                "x2": {"datum": mean_val + std_val}
                            }
                        },
                        {
                            "mark": {"type": "rule", "color": "orange", "strokeWidth": 2, "strokeDash": [5, 5]},
                            "encoding": {
                                "x": {"datum": lower_bound}
                            }
                        },
                        {
                            "mark": {"type": "rule", "color": "orange", "strokeWidth": 2, "strokeDash": [5, 5]},
                            "encoding": {
                                "x": {"datum": upper_bound}
                            }
                        }
                    ],
                    "config": {
                        "view": {"stroke": None},
                        "axis": {"grid": False}
                    }
                }
                
                st.vega_lite_chart(spec, use_container_width=True)
                
                # Show filter stats
                if apply_sd_filter:
                    st.caption(f"✅ **Kept:** {in_range} | ❌ **Filtered:** {out_range}")
                else:
                    st.caption(f"📊 **In range:** {in_range} | 🔍 **Would filter:** {out_range}")
                
                st.caption(f"μ={mean_val:.1f}, σ={std_val:.1f} | Range: [{lower_bound:.1f}, {upper_bound:.1f}]")
            else:
                st.info("No data after filtering")

# CONTAINER 3: Action Buttons
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

st.markdown("---")

# CONTAINER 4: Before/After Comparison Tables at Bottom
if st.session_state.get("compute_stats_now", False):
    with st.spinner("Computing stats..."):
        filtered_stats = compute_stats(filtered_df, protein_model, numeric_cols, protein_species_col)
    
    st.session_state.compute_stats_now = False
    
    st.markdown("### Before vs After Filtering")
    
    col_before, col_after = st.columns(2)
    
    # Helper function to style dataframe with conditional formatting
    def style_metrics_table(data_dict, is_after=False):
        df = pd.DataFrame(data_dict)
        
        def highlight_permanova(row):
            if row['Metric'] == 'PERMANOVA F':
                try:
                    val = float(row['Value'])
                    # Good: F > 5 (strong group separation)
                    if val > 5:
                        return ['background-color: #c6efce; color: #006100'] * len(row)
                    # Moderate: F > 2
                    elif val > 2:
                        return ['background-color: #ffeb9c; color: #9c6500'] * len(row)
                    # Poor: F < 2
                    else:
                        return ['background-color: #ffc7ce; color: #9c0006'] * len(row)
                except:
                    pass
            
            elif row['Metric'] == 'Shapiro W':
                try:
                    val = float(row['Value'])
                    # Good: W > 0.98 (normal distribution)
                    if val > 0.98:
                        return ['background-color: #c6efce; color: #006100'] * len(row)
                    # Moderate: W > 0.95
                    elif val > 0.95:
                        return ['background-color: #ffeb9c; color: #9c6500'] * len(row)
                    # Poor: W < 0.95 (non-normal)
                    else:
                        return ['background-color: #ffc7ce; color: #9c0006'] * len(row)
                except:
                    pass
            
            return [''] * len(row)
        
        styled_df = df.style.apply(highlight_permanova, axis=1)
        return styled_df
    
    # Before table
    with col_before:
        st.markdown("#### Before Filtering")
        
        before_data = {
            "Metric": [
                "Total Proteins",
                "Mean CV%",
                "Median CV%",
                "PERMANOVA F",
                "Shapiro W"
            ],
            "Value": [
                f"{initial_stats['n_proteins']:,}",
                f"{initial_stats['cv_mean']:.1f}" if not np.isnan(initial_stats['cv_mean']) else "N/A",
                f"{initial_stats['cv_median']:.1f}" if not np.isnan(initial_stats['cv_median']) else "N/A",
                f"{initial_stats['permanova_f']:.2f}" if not np.isnan(initial_stats['permanova_f']) else "N/A",
                f"{initial_stats['shapiro_w']:.4f}" if not np.isnan(initial_stats['shapiro_w']) else "N/A",
            ]
        }
        
        if initial_stats['species_counts']:
            for sp in SPECIES_ORDER:
                if sp in initial_stats['species_counts']:
                    before_data["Metric"].append(f"{sp}")
                    before_data["Value"].append(f"{initial_stats['species_counts'][sp]:,}")
        
        styled_before = style_metrics_table(before_data, is_after=False)
        st.dataframe(styled_before, hide_index=True, use_container_width=True, height=400)
    
    # After table
    with col_after:
        st.markdown("#### After Filtering")
        
        after_data = {
            "Metric": [
                "Total Proteins",
                "Mean CV%",
                "Median CV%",
                "PERMANOVA F",
                "Shapiro W"
            ],
            "Value": [
                f"{filtered_stats['n_proteins']:,}",
                f"{filtered_stats['cv_mean']:.1f}" if not np.isnan(filtered_stats['cv_mean']) else "N/A",
                f"{filtered_stats['cv_median']:.1f}" if not np.isnan(filtered_stats['cv_median']) else "N/A",
                f"{filtered_stats['permanova_f']:.2f}" if not np.isnan(filtered_stats['permanova_f']) else "N/A",
                f"{filtered_stats['shapiro_w']:.4f}" if not np.isnan(filtered_stats['shapiro_w']) else "N/A",
            ],
            "Change": [
                f"{filtered_stats['n_proteins'] - initial_stats['n_proteins']:+,}",
                f"{filtered_stats['cv_mean'] - initial_stats['cv_mean']:+.1f}" if not (np.isnan(filtered_stats['cv_mean']) or np.isnan(initial_stats['cv_mean'])) else "—",
                f"{filtered_stats['cv_median'] - initial_stats['cv_median']:+.1f}" if not (np.isnan(filtered_stats['cv_median']) or np.isnan(initial_stats['cv_median'])) else "—",
                f"{filtered_stats['permanova_f'] - initial_stats['permanova_f']:+.2f}" if not (np.isnan(filtered_stats['permanova_f']) or np.isnan(initial_stats['permanova_f'])) else "—",
                f"{filtered_stats['shapiro_w'] - initial_stats['shapiro_w']:+.4f}" if not (np.isnan(filtered_stats['shapiro_w']) or np.isnan(initial_stats['shapiro_w'])) else "—",
            ]
        }
        
        if filtered_stats['species_counts']:
            for sp in SPECIES_ORDER:
                if sp in filtered_stats['species_counts'] or sp in initial_stats['species_counts']:
                    after_data["Metric"].append(f"{sp}")
                    after_val = filtered_stats['species_counts'].get(sp, 0)
                    before_val = initial_stats['species_counts'].get(sp, 0)
                    after_data["Value"].append(f"{after_val:,}")
                    after_data["Change"].append(f"{after_val - before_val:+,}")
        
        styled_after = style_metrics_table(after_data, is_after=True)
        st.dataframe(styled_after, hide_index=True, use_container_width=True, height=400)
    
    # Legend
    st.markdown("---")
    st.markdown("**Statistical Quality Indicators:**")
    leg_col1, leg_col2, leg_col3 = st.columns(3)
    with leg_col1:
        st.markdown("🟢 **Good** - PERMANOVA F > 5, Shapiro W > 0.98")
    with leg_col2:
        st.markdown("🟡 **Moderate** - PERMANOVA F > 2, Shapiro W > 0.95")
    with leg_col3:
        st.markdown("🔴 **Poor** - PERMANOVA F < 2, Shapiro W < 0.95")

else:
    st.info("👆 Click 'Calculate Stats' to see before/after comparison tables")

render_navigation(back_page="pages/3_Preprocessing.py", next_page=None)
render_footer()

