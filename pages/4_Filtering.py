import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Dict

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

    if apply_species and selected_species:
        filtered = model.species_subgroups.get(selected_species)
    elif apply_species:
        filtered = model.species_subgroups.all_species
    
    if filtered.empty:
        return filtered

    if apply_min_pep and "Peptide_Count" in filtered.columns and min_peptides > 1:
        filtered = filtered[filtered["Peptide_Count"] >= min_peptides]

    if filtered.empty:
        return filtered

    if apply_cv and cv_cutoff < 1000:
        cv_data = compute_cv_per_condition(filtered[numeric_cols], numeric_cols)
        if not cv_data.empty:
            cv_mask = cv_data.min(axis=1) <= cv_cutoff
            filtered = filtered[cv_mask]

    if filtered.empty:
        return filtered

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


# ========== INITIALIZE SESSION STATE ==========
if "filter_state" not in st.session_state:
    st.session_state.filter_state = {
        "configured": False,
        "filter_params": None,
        "initial_stats": None,
        "filtered_data": None,
        "filtered_stats": None,
    }

# ========== PAGE SETUP ==========
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

# ========== SIDEBAR: ALL FILTER CONTROLS (ALWAYS VISIBLE, INACTIVE) ==========
with st.sidebar:
    st.markdown("## 🎛️ Filter Configuration")
    st.caption("Adjust filters below. Red squares on plots show active filters as visual guides.")
    
    st.markdown("---")
    
    # ========== FILTER 1: SPECIES ==========
    st.markdown("### 1️⃣ Species")
    
    col_sp1, col_sp2, col_sp3, col_sp4 = st.columns(4)
    with col_sp1:
        species_human = st.checkbox("Human", value=True, key="filter_species_human")
    with col_sp2:
        species_ecoli = st.checkbox("E.coli", value=True, key="filter_species_ecoli")
    with col_sp3:
        species_yeast = st.checkbox("Yeast", value=True, key="filter_species_yeast")
    with col_sp4:
        species_mouse = st.checkbox("Mouse", value=False, key="filter_species_mouse")
    
    apply_species = st.checkbox(
        "Apply filter",
        value=False,
        key="checkbox_apply_species"
    )
    
    st.markdown("---")
    
    # ========== FILTER 2: MIN PEPTIDES ==========
    st.markdown("### 2️⃣ Min Peptides")
    
    min_peptides = st.number_input(
        "Min peptides",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        key="filter_min_peptides",
    )
    
    apply_min_pep = st.checkbox(
        "Apply filter",
        value=False,
        key="checkbox_apply_min_pep"
    )
    
    st.markdown("---")
    
    # ========== FILTER 3: CV CUTOFF ==========
    st.markdown("### 3️⃣ CV% Cutoff")
    
    cv_cutoff = st.number_input(
        "Max CV%",
        min_value=0,
        max_value=100,
        value=30,
        step=5,
        key="filter_cv",
    )
    
    apply_cv = st.checkbox(
        "Apply filter",
        value=False,
        key="checkbox_apply_cv"
    )
    
    st.markdown("---")
    
    # ========== FILTER 4: MISSING DATA ==========
    st.markdown("### 4️⃣ Max Missing %")
    
    max_missing_pct = st.number_input(
        "Max missing %",
        min_value=0,
        max_value=100,
        value=34,
        step=5,
        key="filter_missing",
    )
    
    apply_missing = st.checkbox(
        "Apply filter",
        value=False,
        key="checkbox_apply_missing"
    )
    
    st.markdown("---")
    
    # ========== FILTER 5: SD RANGE ==========
    st.markdown("### 5️⃣ SD Range")
    
    sd_min = st.number_input(
        "Min SD (σ below)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.5,
        key="sd_min_input",
    )
    
    sd_max = st.number_input(
        "Max SD (σ above)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.5,
        key="sd_max_input",
    )
    
    apply_sd = st.checkbox(
        "Apply filter",
        value=False,
        key="checkbox_apply_sd"
    )
    
    st.markdown("---")
    
    # ========== TRANSFORMATION SELECTION ==========
    st.markdown("### 🔄 Transformation")
    transform_key = st.selectbox(
        "Select transformation",
        options=list(TRANSFORMS.keys()),
        format_func=lambda x: TRANSFORMS[x],
        index=0,
        key="filter_transform"
    )
    
    st.markdown("---")
    
    # ========== APPLY ALL FILTERS BUTTON (Expensive Process) ==========
    st.markdown("### ⚙️ Apply Filters")
    
    if st.button("🔄 Apply All Filters", type="primary", key="apply_all_filters", use_container_width=True):
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
        
        if not selected_species:
            selected_species = ["HUMAN", "ECOLI", "YEAST", "MOUSE"]
        
        # Derive effective filter values
        cv_cutoff_val = cv_cutoff if apply_cv else 1000.0
        max_missing_ratio = max_missing_pct / 100.0 if apply_missing else 1.0
        sd_range = (sd_min, sd_max) if apply_sd else None
        
        # Expensive process: compute initial stats and apply filters
        with st.spinner("Computing statistics and applying filters..."):
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
                apply_sd,
                transform_key,
                apply_species=apply_species,
                apply_min_pep=apply_min_pep,
                apply_cv=apply_cv,
                apply_missing=apply_missing,
            )
            
            filtered_stats = compute_stats(filtered_df, protein_model, numeric_cols, protein_species_col)
        
        # Build active filters list
        active_filters = []
        if apply_species and selected_species:
            active_filters.append(f"Species: {', '.join(selected_species)}")
        if apply_min_pep:
            active_filters.append(f"Min peptides: {min_peptides}")
        if apply_cv:
            active_filters.append(f"CV <{cv_cutoff}%")
        if apply_missing:
            active_filters.append(f"Missing <{max_missing_pct}%")
        if apply_sd:
            active_filters.append(f"SD: μ±{sd_min:.1f}σ to μ±{sd_max:.1f}σ")
        
        # Save results to session state
        st.session_state.filter_state = {
            "configured": True,
            "filter_params": {
                "selected_species": selected_species,
                "min_peptides": min_peptides,
                "cv_cutoff": cv_cutoff_val,
                "missing_ratio": max_missing_ratio,
                "use_sd_filter": apply_sd,
                "sd_range": sd_range,
                "transform_key": transform_key,
                "active_filters": active_filters,
                "apply_species": apply_species,
                "apply_min_pep": apply_min_pep,
                "apply_cv": apply_cv,
                "apply_missing": apply_missing,
                "apply_sd": apply_sd,
            },
            "initial_stats": initial_stats,
            "filtered_data": filtered_df,
            "filtered_stats": filtered_stats,
        }
        
        st.success("✅ Filters applied! Scroll down to see results.")

# ========== MAIN PAGE: RESULTS & VISUALIZATIONS ==========
# Only display if filters have been applied
if st.session_state.filter_state["configured"]:
    
    # Extract saved results from session state
    filter_params = st.session_state.filter_state["filter_params"]
    initial_stats = st.session_state.filter_state["initial_stats"]
    filtered_df = st.session_state.filter_state["filtered_data"]
    filtered_stats = st.session_state.filter_state["filtered_stats"]
    active_filters = filter_params["active_filters"]
    transform_key = filter_params["transform_key"]
    apply_sd = filter_params["use_sd_filter"]
    sd_range = filter_params["sd_range"]
    
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
            delta=f"{filtered_stats['n_proteins'] / initial_stats['n_proteins'] * 100:.1f}%" if initial_stats['n_proteins'] > 0 else "N/A"
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
    
    if active_filters:
        for i, f in enumerate(active_filters, 1):
            st.caption(f"✓ {f}")
    else:
        st.caption("No filters active - using all proteins")
    
    st.markdown("---")
    
# ========== INTENSITY DISTRIBUTION HISTOGRAMS ==========
st.markdown("### 📊 Intensity Distribution by Sample")

# Get transformed data
transform_data = get_transform_data(protein_model, transform_key)

# Create histograms with range slider
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
                transform_data[sample].dropna()
                if sample in transform_data.columns
                else pd.Series(dtype=float)
            )
            
            if not sample_data.empty:
                mean_val = sample_data.mean()
                std_val = sample_data.std()
                
                # Calculate SD bounds for visual guide
                lower_bound = mean_val - (sd_min * std_val)
                upper_bound = mean_val + (sd_max * std_val)
                
                # Count in/out of range
                in_range = ((sample_data >= lower_bound) & (sample_data <= upper_bound)).sum()
                out_range = len(sample_data) - in_range
                
                # Create figure with range slider
                fig = go.Figure()
                
                # Single histogram trace
                fig.add_trace(go.Histogram(
                    x=sample_data,
                    nbinsx=50,
                    marker_color="rgba(135, 206, 235, 0.7)",
                    name="Intensity",
                    showlegend=False,
                ))
                
                # Add mean line
                fig.add_vline(
                    x=mean_val,
                    line_dash="solid",
                    line_color="red",
                    line_width=2,
                    annotation_text=f"μ={mean_val:.1f}",
                    annotation_position="top",
                )
                
                # Add ±1σ shaded area
                fig.add_vrect(
                    x0=mean_val - std_val,
                    x1=mean_val + std_val,
                    fillcolor="red",
                    opacity=0.1,
                    layer="below",
                    line_width=0,
                )
                
                # Add SD range bounds as VISUAL GUIDE (orange dashed lines)
                # These show the filter preview, not actual filter status
                if apply_sd:
                    # RED rectangle to show active filter
                    fig.add_vrect(
                        x0=lower_bound,
                        x1=upper_bound,
                        fillcolor="red",
                        opacity=0.15,
                        layer="below",
                        line_width=2,
                        line_color="red",
                        line_dash="dash",
                    )
                else:
                    # ORANGE dashed lines for visual reference (inactive)
                    fig.add_vline(
                        x=lower_bound,
                        line_dash="dash",
                        line_color="orange",
                        line_width=2,
                    )
                    fig.add_vline(
                        x=upper_bound,
                        line_dash="dash",
                        line_color="orange",
                        line_width=2,
                    )
                
                fig.update_layout(
                    title=f"{sample} (n={len(sample_data)})",
                    xaxis_title=TRANSFORMS[transform_key],
                    yaxis_title="Count",
                    height=400,
                    plot_bgcolor="#FFFFFF",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Arial", size=10, color="#54585A"),
                    showlegend=False,
                    margin=dict(l=40, r=40, t=60, b=80),
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"hist_{sample}_{sample_idx}")
                
                # Show stats
                if apply_sd:
                    st.caption(f"🔴 **Filter active** | ✅ Kept: {in_range} ({in_range/len(sample_data)*100:.1f}%) | ❌ Filtered: {out_range} ({out_range/len(sample_data)*100:.1f}%)")
                else:
                    st.caption(f"📊 In range: {in_range} | Preview: {out_range} would be filtered")
                
                st.caption(f"μ={mean_val:.1f}, σ={std_val:.1f} | Range: [{lower_bound:.1f}, {upper_bound:.1f}]")
            else:
                st.info("No data")

    st.markdown("---")
    
    # ========== STORE FILTERED DATASET ==========
    col_store1, col_store2 = st.columns([1, 3])
    
    with col_store1:
        if st.button("💾 Store for Analysis", type="primary", key="store_filtered"):
            if not filtered_df.empty:
                st.session_state.last_filtered_data = filtered_df.copy()
                st.session_state.last_filtered_params = filter_params
                st.success(f"✅ Stored {len(filtered_df):,} proteins for analysis!")
            else:
                st.error("❌ No proteins after filtering.")
    
    with col_store2:
        if not filtered_df.empty:
            st.metric("Proteins Ready", f"{filtered_stats['n_proteins']:,}")
    
    st.markdown("---")
    
    # ========== EXPORT FILTERED DATA ==========
    if not filtered_df.empty:
        col_exp1, col_exp2 = st.columns([1, 1])
        
        with col_exp1:
            if st.button("💾 Export Filtered Data", key="export_filtered"):
                csv = filtered_df.to_csv(index=True)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="filtered_proteins.csv",
                    mime="text/csv",
                )
        
        with col_exp2:
            st.caption("Export current filtered dataset as CSV")

else:
    st.info("👆 Configure filters in the sidebar and click **Apply All Filters** to see results")

render_navigation(back_page="pages/3_Preprocessing.py", next_page=None)
render_footer()
