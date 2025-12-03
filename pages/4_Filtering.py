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

# ========== SPECIES-SPECIFIC OPTIMIZATION MODE ==========
st.markdown("## 🎯 Species-Specific Optimization")

col_mode1, col_mode2 = st.columns([2, 1])

with col_mode1:
    optimization_mode = st.radio(
        "Filtering mode",
        options=["Global filters (same for all species)", "Species-specific optimization (build sequentially)"],
        index=0,
        key="optimization_mode",
        help="Global: Apply same filters to all species. Species-specific: Optimize filters independently for each species."
    )

use_species_optimization = optimization_mode == "Species-specific optimization (build sequentially)"

if use_species_optimization:
    st.info("📋 **Sequential Selection**: Configure and add each species separately with optimized filters. The final dataset combines all selections.")
    
    # Initialize species-specific state - store indices instead of DataFrames
    if "optimized_species_indices" not in st.session_state:
        st.session_state.optimized_species_indices = {}
    
    # Species selection for optimization
    available_species = ["HUMAN", "ECOLI", "YEAST", "MOUSE"]
    already_added = list(st.session_state.optimized_species_indices.keys())
    remaining_species = [sp for sp in available_species if sp not in already_added]
    
    if remaining_species:
        st.markdown("### Configure Next Species")
        
        col_sp1, col_sp2, col_sp3 = st.columns([2, 2, 1])
        
        with col_sp1:
            current_species = st.selectbox(
                "Select species to optimize",
                options=remaining_species,
                key="current_species_select"
            )
        
        with col_sp2:
            st.metric("Species count", len(already_added))
            st.caption(f"Added: {', '.join(already_added) if already_added else 'None'}")
        
        # Species-specific filter overrides
        st.markdown(f"#### Filters for {current_species}")
        
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            sp_use_peptides = st.checkbox(
                "Min peptides",
                value=use_min_peptides,
                key=f"sp_peptides_{current_species}"
            )
            sp_min_peptides = st.slider(
                "Count",
                min_value=1,
                max_value=10,
                value=min_peptides,
                disabled=not sp_use_peptides,
                key=f"sp_min_pep_val_{current_species}"
            )
        
        with col_f2:
            sp_use_cv = st.checkbox(
                "CV% cutoff",
                value=use_cv,
                key=f"sp_cv_{current_species}"
            )
            sp_cv_cutoff = st.slider(
                "Max CV%",
                min_value=0,
                max_value=100,
                value=cv_cutoff,
                step=5,
                disabled=not sp_use_cv,
                key=f"sp_cv_val_{current_species}"
            )
        
        with col_f3:
            sp_use_missing = st.checkbox(
                "Max missing",
                value=use_missing,
                key=f"sp_missing_{current_species}"
            )
            sp_missing_pct = st.slider(
                "Max %",
                min_value=0,
                max_value=100,
                value=max_missing_pct,
                step=10,
                disabled=not sp_use_missing,
                key=f"sp_missing_val_{current_species}"
            )
        
        # Preview stats for this species
        sp_min_pep = sp_min_peptides if sp_use_peptides else 1
        sp_cv = sp_cv_cutoff if sp_use_cv else 1000.0
        sp_missing_ratio = sp_missing_pct / 100.0 if sp_use_missing else 1.0
        
        # Get species subset
        species_subset = protein_model.species_subgroups.get([current_species])
        
        if not species_subset.empty:
            # Apply filters to preview
            preview_filtered = apply_filters(
                species_subset[numeric_cols],
                protein_model,
                numeric_cols,
                [current_species],
                sp_min_pep,
                sp_cv,
                sp_missing_ratio,
                None,
                transform_key,
            )
            
            preview_stats = compute_stats(preview_filtered, protein_model, numeric_cols, protein_species_col)
            
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            col_p1.metric("Before", f"{len(species_subset):,}")
            col_p2.metric("After", f"{preview_stats['n_proteins']:,}")
            col_p3.metric("Kept", f"{preview_stats['n_proteins'] / len(species_subset) * 100:.1f}%")
            col_p4.metric("Mean CV%", f"{preview_stats['cv_mean']:.1f}" if not np.isnan(preview_stats['cv_mean']) else "N/A")
            
            col_add1, col_add2, _ = st.columns([1, 1, 2])
            
            with col_add1:
                if st.button(f"➕ Add {current_species} with these filters", type="primary", key=f"add_{current_species}"):
                    # Store only indices and filter settings - NOT DataFrames
                    st.session_state.optimized_species_indices[current_species] = {
                        "indices": preview_filtered.index.tolist(),  # Store as list
                        "filters": {
                            "min_peptides": sp_min_pep,
                            "cv_cutoff": sp_cv,
                            "missing_ratio": sp_missing_ratio,
                        },
                        "n_proteins": preview_stats['n_proteins']
                    }
                    st.rerun()
            
            with col_add2:
                if st.button("⏭️ Skip this species", key=f"skip_{current_species}"):
                    st.session_state.optimized_species_indices[current_species] = {
                        "indices": [],
                        "filters": {"skipped": True},
                        "n_proteins": 0
                    }
                    st.rerun()
        else:
            st.warning(f"No {current_species} proteins found in dataset")
    
    else:
        st.success("✅ All species configured!")
    
    # Show summary of added species
    if already_added:
        st.markdown("### Current Selection Summary")
        
        summary_data = []
        for sp in already_added:
            sp_info = st.session_state.optimized_species_indices[sp]
            if sp_info["filters"].get("skipped"):
                summary_data.append({
                    "Species": sp,
                    "Proteins": 0,
                    "Status": "Skipped",
                    "Min Peptides": "—",
                    "Max CV%": "—",
                    "Max Missing%": "—"
                })
            else:
                filters = sp_info["filters"]
                summary_data.append({
                    "Species": sp,
                    "Proteins": sp_info["n_proteins"],
                    "Status": "Added",
                    "Min Peptides": filters["min_peptides"],
                    "Max CV%": f"{filters['cv_cutoff']:.0f}" if filters['cv_cutoff'] < 1000 else "None",
                    "Max Missing%": f"{filters['missing_ratio']*100:.0f}"
                })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, hide_index=True, use_container_width=True)
        
        col_action1, col_action2, col_action3 = st.columns([1, 1, 2])
        
        with col_action1:
            if st.button("🔄 Reset Selection", key="reset_optimization"):
                st.session_state.optimized_species_indices = {}
                st.session_state.pop("optimization_finalized", None)
                st.rerun()
        
        with col_action2:
            if len(already_added) == len(available_species):
                if st.button("✅ Finalize & Continue", type="primary", key="finalize_optimization"):
                    st.session_state.optimization_finalized = True
                    st.rerun()
        
        # Show combined stats if finalized
        if st.session_state.get("optimization_finalized"):
            # Reconstruct combined dataframe from stored indices
            all_indices = []
            for sp_info in st.session_state.optimized_species_indices.values():
                if not sp_info["filters"].get("skipped") and sp_info["indices"]:
                    all_indices.extend(sp_info["indices"])
            
            if all_indices:
                filtered_df = protein_model.raw_filled.loc[all_indices, numeric_cols]
                
                st.markdown("### Combined Dataset Statistics")
                combined_stats = compute_stats(filtered_df, protein_model, numeric_cols, protein_species_col)
                
                col_c1, col_c2, col_c3, col_c4 = st.columns(4)
                col_c1.metric("Total Proteins", f"{combined_stats['n_proteins']:,}")
                col_c2.metric("Mean CV%", f"{combined_stats['cv_mean']:.1f}" if not np.isnan(combined_stats['cv_mean']) else "N/A")
                col_c3.metric("PERMANOVA F", f"{combined_stats['permanova_f']:.2f}" if not np.isnan(combined_stats['permanova_f']) else "N/A")
                col_c4.metric("Shapiro W", f"{combined_stats['shapiro_w']:.4f}" if not np.isnan(combined_stats['shapiro_w']) else "N/A")

st.markdown("---")

# ========== MAIN FILTERING ==========

# Initial stats (unfiltered)
initial_stats = compute_stats(df_raw, protein_model, numeric_cols, protein_species_col)

# CONTAINER 1: Summary Stats
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

# Determine which filtered_df to use
if use_species_optimization and st.session_state.get("optimization_finalized"):
    # Use optimized combined dataset
    all_indices = []
    for sp_info in st.session_state.optimized_species_indices.values():
        if not sp_info["filters"].get("skipped") and sp_info["indices"]:
            all_indices.extend(sp_info["indices"])
    
    filtered_df = protein_model.raw_filled.loc[all_indices, numeric_cols] if all_indices else pd.DataFrame()
else:
    # Use global filters
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

# CONTAINER 2: Intensity Distribution
st.markdown("### Intensity Distribution by Sample")

# Create histograms in 3-column grid
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
                
                fig.add_trace(go.Histogram(
                    x=sample_data,
                    name="Distribution",
                    nbinsx=50,
                    marker_color="rgba(135, 206, 235, 0.7)",
                    showlegend=False,
                ))
                
                fig.add_vline(
                    x=mean_val,
                    line_dash="solid",
                    line_color="red",
                    line_width=2,
                    annotation_text=f"μ={mean_val:.1f}",
                    annotation_position="top",
                )
                
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

# CONTAINER 4: Before/After Comparison Tables
if st.session_state.get("compute_stats_now", False):
    with st.spinner("Computing stats..."):
        filtered_stats = compute_stats(filtered_df, protein_model, numeric_cols, protein_species_col)
    
    st.session_state.compute_stats_now = False
    
    # Helper function to style dataframe
    def style_metrics_table(data_dict):
        df = pd.DataFrame(data_dict)
        
        def highlight_permanova(row):
            if row['Metric'] == 'PERMANOVA F':
                try:
                    val = float(row['Value'])
                    if val > 5:
                        return ['background-color: #c6efce; color: #006100'] * len(row)
                    elif val > 2:
                        return ['background-color: #ffeb9c; color: #9c6500'] * len(row)
                    else:
                        return ['background-color: #ffc7ce; color: #9c0006'] * len(row)
                except:
                    pass
            
            elif row['Metric'] == 'Shapiro W':
                try:
                    val = float(row['Value'])
                    if val > 0.98:
                        return ['background-color: #c6efce; color: #006100'] * len(row)
                    elif val > 0.95:
                        return ['background-color: #ffeb9c; color: #9c6500'] * len(row)
                    else:
                        return ['background-color: #ffc7ce; color: #9c0006'] * len(row)
                except:
                    pass
            
            return [''] * len(row)
        
        styled_df = df.style.apply(highlight_permanova, axis=1)
        return styled_df
    
    st.markdown("### Before vs After Filtering")
    
    col_before, col_after = st.columns(2)
    
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
        
        styled_before = style_metrics_table(before_data)
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
        
        styled_after = style_metrics_table(after_data)
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
