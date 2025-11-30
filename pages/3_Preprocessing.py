import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass
from typing import List
from scipy.stats import f_oneway, kruskal

from components import inject_custom_css, render_header, render_navigation, render_footer, COLORS

st.set_page_config(
    page_title="Preprocessing | Thermo Fisher Scientific",
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


TF_CHART_COLORS = ["#262262", "#A6192E", "#EA7600", "#F1B434", "#B5BD00", "#9BD3DD"]

SPECIES_COLORS = {
    "HUMAN": "#87CEEB",  # Sky blue
    "ECOLI": "#008B8B",  # Teal
    "YEAST": "#FF8C00",  # Orange
    "MOUSE": "#9370DB",  # Purple
    "UNKNOWN": "#808080",  # Gray
}

SPECIES_ORDER = ["HUMAN", "ECOLI", "YEAST", "MOUSE", "UNKNOWN"]


def extract_conditions(cols: list[str]) -> dict:
    """Extract condition assignments from column names (A1→A, A2→A, B1→B, etc.)"""
    condition_map = {}
    for col in cols:
        if col and col[0].isalpha():
            condition_map[col] = col[0]
        else:
            condition_map[col] = "X"
    return condition_map


def compute_cv_per_condition(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Compute CV% for each protein within each condition (using replicates only)."""
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


def compute_species_cv_per_condition(df: pd.DataFrame, numeric_cols: list[str], species_col: str) -> pd.DataFrame:
    """Compute CV per condition for each species separately (within replicates only)."""
    if species_col not in df.columns:
        return pd.DataFrame()
    
    condition_map = extract_conditions(numeric_cols)
    conditions = {}
    for col in numeric_cols:
        cond = condition_map[col]
        conditions.setdefault(cond, []).append(col)
    
    species_list = df[species_col].dropna().unique()
    
    results = []
    for species in species_list:
        species_df = df[df[species_col] == species]
        for cond, cols in conditions.items():
            if len(cols) >= 2:
                mean_vals = species_df[cols].mean(axis=1)
                std_vals = species_df[cols].std(axis=1)
                cvs = (std_vals / mean_vals * 100).replace([np.inf, -np.inf], np.nan)
                valid_cvs = cvs.dropna()
                if len(valid_cvs) > 0:
                    results.append({
                        "Species": species,
                        "Condition": cond,
                        "Mean_CV": valid_cvs.mean(),
                        "Median_CV": valid_cvs.median(),
                        "Count": len(valid_cvs),
                    })
    
    return pd.DataFrame(results)


@st.cache_data
def create_stacked_species_barplot(df: pd.DataFrame, numeric_cols: list[str], species_col: str) -> go.Figure:
    """Create stacked bar plot showing protein count per species for each sample."""
    if species_col not in df.columns:
        return None
    
    all_species = df[species_col].dropna().unique()
    species_list = [s for s in SPECIES_ORDER if s in all_species]
    
    data = []
    for species in species_list:
        counts = []
        species_df = df[df[species_col] == species]
        for col in numeric_cols:
            non_missing = species_df[col].notna() & (species_df[col] > 1.0)
            counts.append(non_missing.sum())
        
        data.append(go.Bar(
            name=species,
            x=numeric_cols,
            y=counts,
            marker_color=SPECIES_COLORS.get(species, "#808080"),
        ))
    
    fig = go.Figure(data=data)
    fig.update_layout(
        barmode='stack',
        title="Protein count per species across samples",
        xaxis_title="Sample",
        yaxis_title="Protein count",
        height=500,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=12, color="#54585A"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        margin=dict(l=60, r=40, t=80, b=60),
    )
    
    return fig


@st.cache_data
def compute_peptides_per_protein(
    protein_df: pd.DataFrame,
    peptide_df: pd.DataFrame,
    protein_idx: str,
    peptide_idx: str,
    protein_species_col: str,
    numeric_cols: list[str]
) -> pd.DataFrame:
    """Count peptides per protein using protein group counts."""
    if protein_idx not in protein_df.columns or peptide_idx not in peptide_df.columns:
        return pd.DataFrame()
    
    if protein_species_col not in protein_df.columns:
        return pd.DataFrame()
    
    results = []
    
    for sample in numeric_cols:
        if sample not in peptide_df.columns:
            continue
        
        detected_peptides = peptide_df[peptide_df[sample] > 1.0]
        peptide_counts = detected_peptides[peptide_idx].value_counts()
        
        for protein_id, count in peptide_counts.items():
            protein_match = protein_df[protein_df[protein_idx] == protein_id]
            if len(protein_match) > 0:
                species = protein_match[protein_species_col].iloc[0]
                results.append({
                    "Sample": sample,
                    "Species": species,
                    "Peptide_Count": count,
                })
    
    return pd.DataFrame(results)


@st.cache_data
def create_peptides_per_protein_plot(ppp_df: pd.DataFrame) -> go.Figure:
    """Create box plot of peptides per protein by species and sample."""
    if ppp_df.empty:
        return None
    
    all_species = ppp_df["Species"].unique()
    species_list = [s for s in SPECIES_ORDER if s in all_species]
    samples = sorted(ppp_df["Sample"].unique())
    
    fig = go.Figure()
    
    for species in species_list:
        species_data = ppp_df[ppp_df["Species"] == species]
        x_data = []
        y_data = []
        
        for sample in samples:
            sample_data = species_data[species_data["Sample"] == sample]["Peptide_Count"]
            x_data.extend([sample] * len(sample_data))
            y_data.extend(sample_data)
        
        fig.add_trace(go.Box(
            y=y_data,
            x=x_data,
            name=species,
            marker_color=SPECIES_COLORS.get(species, "#808080"),
            boxmean='sd',
            line=dict(width=2),
        ))
    
    fig.update_layout(
        title="Peptides per protein by species and sample",
        xaxis_title="Sample",
        yaxis_title="Peptides per protein",
        height=500,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=12, color="#54585A"),
        boxmode='group',
        boxgap=0.2,
        boxgroupgap=0.1,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        margin=dict(l=60, r=40, t=80, b=60),
    )
    
    return fig


st.markdown("## Preprocessing & Quality Control")

protein_model: MSData | None = st.session_state.get("protein_model")
peptide_model: MSData | None = st.session_state.get("peptide_model")
protein_idx = st.session_state.get("protein_index_col")
peptide_idx = st.session_state.get("peptide_index_col")
peptide_seq_col = st.session_state.get("peptide_seq_col")
protein_species_col = st.session_state.get("protein_species_col")
peptide_species_col = st.session_state.get("peptide_species_col")

if protein_model is None and peptide_model is None:
    st.warning("No data cached. Please upload data on the Data Upload page first.")
    render_navigation(back_page="pages/2_EDA.py", next_page=None)
    render_footer()
    st.stop()

tab_protein, tab_peptide = st.tabs(["Protein-level QC", "Peptide-level QC"])


def render_preprocessing(model: MSData | None, species_col: str | None, label: str):
    if model is None:
        st.info(f"No {label} data uploaded yet")
        return

    numeric_cols = model.numeric_cols
    df_raw = model.raw_filled[numeric_cols]
    
    st.caption(f"**{len(df_raw):,} {label}s** × **{len(numeric_cols)} samples**")
    
    # Section 1: CV per Condition Analysis
    st.markdown("### Coefficient of Variation (CV%) per Condition")
    st.caption("CV within biological replicates only. Thresholds: <15% excellent, <20% acceptable")
    
    cv_per_condition = compute_cv_per_condition(df_raw, numeric_cols)
    
    if cv_per_condition.empty:
        st.warning("No conditions with ≥2 replicates found. Cannot compute CV.")
        return
    
    # Violin plot for each condition (cap at 100% for visualization)
    st.markdown("#### CV Distribution by Condition")
    
    fig = go.Figure()
    for i, cond in enumerate(cv_per_condition.columns):
        cond_data = cv_per_condition[cond].dropna()
        cond_data_capped = cond_data[cond_data <= 100]
        
        if len(cond_data_capped) > 0:
            fig.add_trace(go.Violin(
                y=cond_data_capped,
                name=cond.replace("CV_", ""),
                box_visible=True,
                meanline_visible=True,
                fillcolor=TF_CHART_COLORS[i % len(TF_CHART_COLORS)],
                opacity=0.7,
                line=dict(width=2),
                points=False,
            ))
    
    fig.update_layout(
        title="CV% distribution per condition (capped at 100% for display)",
        yaxis_title="CV%",
        xaxis_title="Condition",
        height=500,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=12, color="#54585A"),
        showlegend=False,
        violingap=0.2,
        violinmode='group',
        margin=dict(l=60, r=40, t=80, b=60),
    )
    
    fig.add_hline(y=20, line_dash="dash", line_color="orange", line_width=2, annotation_text="20% threshold", annotation_position="right")
    fig.add_hline(y=15, line_dash="dash", line_color="green", line_width=2, annotation_text="15% threshold", annotation_position="right")
    
    st.plotly_chart(fig, width="stretch", key=f"cv_violin_{label}")
    
    # Show count of proteins >100% CV
    total_proteins = cv_per_condition.notna().any(axis=1).sum()
    above_100_count = (cv_per_condition > 100).any(axis=1).sum()
    if above_100_count > 0:
        st.caption(f"ℹ️ {above_100_count} {label}s ({above_100_count/total_proteins*100:.1f}%) have CV >100% (excluded from plot)")
    
    st.markdown("---")
    
    # Section 2: CV Summary Table
    st.markdown("### CV Summary per Condition")
    
    cv_summary = {}
    for col in cv_per_condition.columns:
        clean = cv_per_condition[col].dropna()
        if len(clean) > 0:
            cv_summary[col.replace("CV_", "Condition ")] = {
                "Mean CV%": f"{clean.mean():.1f}",
                "Median CV%": f"{clean.median():.1f}",
                "% <15%": f"{(clean < 15).sum() / len(clean) * 100:.1f}",
                "% <20%": f"{(clean < 20).sum() / len(clean) * 100:.1f}",
                f"N {label}s": len(clean),
            }
    
    summary_df = pd.DataFrame(cv_summary).T
    st.dataframe(summary_df, width="stretch")
    
    # Quality assessment
    all_cvs = cv_per_condition.values.flatten()
    all_cvs_clean = all_cvs[~np.isnan(all_cvs)]
    above_20_pct = (all_cvs_clean >= 20).sum() / len(all_cvs_clean) * 100
    
    if above_20_pct > 20:
        st.warning(f"⚠️ {above_20_pct:.1f}% of {label}s have CV ≥20% within conditions. Consider preprocessing.")
    else:
        st.success(f"✓ Good technical reproducibility: {100-above_20_pct:.1f}% of {label}s <20% CV")
    
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
        st.session_state.filter_cv_cutoff = 1000.0  # No filter

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
        st.session_state.filter_max_missing_ratio = 1.0  # No filter

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
transform_data_filtered = get_transform_data(protein_model, st.session_state.filter_transform).loc[filtered_df.index, numeric_cols]

# Create histograms (one per sample)
cols = st.columns(len(numeric_cols))

for i, sample in enumerate(numeric_cols):
    with cols[i]:
        fig = go.Figure()
        
        sample_data = transform_data_filtered[sample].dropna()
        
        if len(sample_data) > 0:
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
                xaxis_title=f"{TRANSFORMS[st.session_state.filter_transform]}",
                yaxis_title="Count",
                height=350,
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Arial", size=10, color="#54585A"),
                showlegend=False,
                margin=dict(l=40, r=40, t=60, b=40),
            )
            
            st.plotly_chart(fig, use_container_width=True, key=f"hist_{sample}")

st.markdown("---")

# CONTAINER 5: Updated Stats with Arrows
st.markdown("### After Filtering")

# Show active filters
active_filters = []
if use_peptides:
    active_filters.append(f"Min peptides: {st.session_state.filter_min_peptides}")
if use_cv:
    active_filters.append(f"CV <{st.session_state.filter_cv_cutoff:.0f}%")
if use_missing:
    active_filters.append(f"Max missing: {int(st.session_state.filter_max_missing_ratio * 100)}%")
if use_intensity:
    active_filters.append(f"Intensity range: {st.session_state.filter_intensity_range[0]:.1f}-{st.session_state.filter_intensity_range[1]:.1f}")

filter_str = "**Active filters:** " + " | ".join(active_filters) if active_filters else "**No filters active** (showing all proteins)"
st.caption(filter_str)

# Compute filtered stats
filtered_stats = compute_stats(filtered_df, protein_model, numeric_cols, protein_species_col)

# Display with arrows
def get_arrow(before, after, higher_is_better=True):
    if np.isnan(before) or np.isnan(after):
        return "→"
    change = after - before
    if change > 0:
        return "↑" if higher_is_better else "↓"
    elif change < 0:
        return "↓" if higher_is_better else "↑"
    else:
        return "→"

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    arrow = get_arrow(initial_stats['n_proteins'], filtered_stats['n_proteins'], higher_is_better=True)
    st.metric(f"Proteins {arrow}", f"{filtered_stats['n_proteins']:,}")

with col2:
    species_str = ", ".join([f"{s}:{filtered_stats['species_counts'].get(s, 0)}" for s in SPECIES_ORDER if s in filtered_stats['species_counts']])
    st.metric("Species Count", species_str)

with col3:
    arrow = get_arrow(initial_stats['cv_mean'], filtered_stats['cv_mean'], higher_is_better=False)
    st.metric(f"Mean CV% {arrow}", f"{filtered_stats['cv_mean']:.1f}" if not np.isnan(filtered_stats['cv_mean']) else "N/A")

with col4:
    arrow = get_arrow(initial_stats['cv_median'], filtered_stats['cv_median'], higher_is_better=False)
    st.metric(f"Median CV% {arrow}", f"{filtered_stats['cv_median']:.1f}" if not np.isnan(filtered_stats['cv_median']) else "N/A")

with col5:
    arrow = get_arrow(initial_stats['permanova_f'], filtered_stats['permanova_f'], higher_is_better=True)
    st.metric(f"PERMANOVA F {arrow}", f"{filtered_stats['permanova_f']:.2f}" if not np.isnan(filtered_stats['permanova_f']) else "N/A")

with col6:
    arrow = get_arrow(initial_stats['shapiro_w'], filtered_stats['shapiro_w'], higher_is_better=True)
    st.metric(f"Shapiro W {arrow}", f"{filtered_stats['shapiro_w']:.4f}" if not np.isnan(filtered_stats['shapiro_w']) else "N/A")

col1, col2 = st.columns([2, 3])
with col1:
    if st.button("📊 Recalculate Stats", key="recalc_stats_btn"):
        st.rerun()

with col2:
    if st.button("💾 Export Filtered Data", key="export_btn"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="filtered_proteins.csv",
            mime="text/csv",
        )

    
    # Section 4: Species distribution per sample
    if species_col and species_col in model.raw.columns:
        st.markdown(f"### {label.capitalize()} Distribution by Species")
        st.caption(f"Number of detected {label}s per species in each sample")
        
        df_with_species = model.raw_filled[numeric_cols].copy()
        df_with_species[species_col] = model.raw[species_col]
        
        fig_species = create_stacked_species_barplot(df_with_species, numeric_cols, species_col)
        if fig_species:
            st.plotly_chart(fig_species, width="stretch", key=f"species_stack_{label}")
    
    st.markdown("---")
    
    # Section 5: Peptides per protein
    if label == "protein" and peptide_model is not None and protein_idx and peptide_idx:
        st.markdown("### Peptides per Protein Analysis")
        st.caption("Number of peptides detected per protein across samples and species")
        
        ppp_df = compute_peptides_per_protein(
            model.raw_filled,
            peptide_model.raw_filled,
            protein_idx,
            peptide_idx,
            species_col,
            numeric_cols
        )
        
        if not ppp_df.empty:
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean peptides/protein", f"{ppp_df['Peptide_Count'].mean():.1f}")
            with col2:
                st.metric("Median peptides/protein", f"{ppp_df['Peptide_Count'].median():.1f}")
            with col3:
                single_peptide_pct = (ppp_df['Peptide_Count'] == 1).sum() / len(ppp_df) * 100
                st.metric("Single-peptide proteins", f"{single_peptide_pct:.1f}%")
            with col4:
                above_15 = (ppp_df['Peptide_Count'] > 15).sum()
                st.metric("Proteins >15 peptides", f"{above_15}")
            
            # Box plot (capped at 15)
            ppp_df_capped = ppp_df[ppp_df['Peptide_Count'] <= 15].copy()
            
            if not ppp_df_capped.empty:
                fig_ppp = create_peptides_per_protein_plot(ppp_df_capped)
                if fig_ppp:
                    fig_ppp.update_layout(title="Peptides per protein by species and sample (capped at 15)")
                    st.plotly_chart(fig_ppp, width="stretch", key=f"ppp_{label}")
                
                excluded_pct = (len(ppp_df) - len(ppp_df_capped)) / len(ppp_df) * 100
                if excluded_pct > 0:
                    st.caption(f"ℹ️ {excluded_pct:.1f}% of proteins have >15 peptides (excluded from plot)")
            
            st.markdown("---")
            
            # Species comparison analysis
            st.markdown("#### Peptides per Protein: Species Comparison")
            
            species_stats = ppp_df.groupby('Species')['Peptide_Count'].agg([
                ('Mean', 'mean'),
                ('Median', 'median'),
                ('Std', 'std'),
                ('Min', 'min'),
                ('Max', 'max'),
                ('Count', 'count'),
            ]).round(2)
            
            # Add percentage columns
            total = species_stats['Count'].sum()
            species_stats['% of Total'] = (species_stats['Count'] / total * 100).round(1)
            
            # Single peptide proteins per species
            single_pep = ppp_df[ppp_df['Peptide_Count'] == 1].groupby('Species').size()
            species_stats['Single-peptide %'] = (single_pep / species_stats['Count'] * 100).round(1).fillna(0)
            
            # Reorder columns
            species_stats = species_stats[['Count', '% of Total', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Single-peptide %']]
            
            # Sort by species order
            species_stats = species_stats.reindex([s for s in SPECIES_ORDER if s in species_stats.index])
            
            st.dataframe(
                species_stats.style.background_gradient(subset=['Mean', 'Median'], cmap='RdYlGn', vmin=1, vmax=10),
                width="stretch"
            )
            
            # Statistical comparison
            st.markdown("#### Statistical Comparison (ANOVA)")
            
            if len(ppp_df['Species'].unique()) >= 2:
                species_groups = [group['Peptide_Count'].values for name, group in ppp_df.groupby('Species')]
                
                # ANOVA (parametric)
                f_stat, p_anova = f_oneway(*species_groups)
                
                # Kruskal-Wallis (non-parametric)
                h_stat, p_kruskal = kruskal(*species_groups)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("ANOVA F-statistic", f"{f_stat:.2f}")
                    st.metric("ANOVA p-value", f"{p_anova:.4f}")
                    if p_anova < 0.05:
                        st.success("✓ Significant difference between species (p < 0.05)")
                    else:
                        st.info("No significant difference between species")
                
                with col2:
                    st.metric("Kruskal-Wallis H", f"{h_stat:.2f}")
                    st.metric("Kruskal p-value", f"{p_kruskal:.4f}")
                    if p_kruskal < 0.05:
                        st.success("✓ Significant difference (non-parametric test)")
                    else:
                        st.info("No significant difference (non-parametric test)")
                
                if p_anova < 0.05 or p_kruskal < 0.05:
                    st.caption("💡 Significant differences suggest species-specific protein coverage or identification rates.")
        else:
            st.info("No peptide-to-protein mapping found. Check that protein IDs match between datasets.")


with tab_protein:
    render_preprocessing(protein_model, protein_species_col, "protein")

with tab_peptide:
    render_preprocessing(peptide_model, peptide_species_col, "peptide")

render_navigation(back_page="pages/2_EDA.py", next_page="pages/4_Filtering.py")
render_footer()
