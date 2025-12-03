import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Dict
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


SPECIES_COLORS = {
    "HUMAN": "#87CEEB",
    "ECOLI": "#008B8B",
    "YEAST": "#FF8C00",
    "MOUSE": "#9370DB",
    "UNKNOWN": "#808080",
}

SPECIES_ORDER = ["HUMAN", "ECOLI", "YEAST", "MOUSE", "UNKNOWN"]


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


def compute_cv_per_condition(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Compute CV% for each protein within each condition (using replicates only)."""
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


def compute_species_cv_per_condition(
    df: pd.DataFrame, numeric_cols: List[str], species_col: str
) -> pd.DataFrame:
    """Compute CV per condition for each species separately (within replicates only)."""
    if species_col not in df.columns:
        return pd.DataFrame()
    
    condition_groups = build_condition_groups(numeric_cols)
    species_list = df[species_col].dropna().unique()
    
    results = []
    for species in species_list:
        species_df = df[df[species_col] == species]
        for cond, cols in condition_groups.items():
            if len(cols) < 2:
                continue
            mean_vals = species_df[cols].mean(axis=1)
            std_vals = species_df[cols].std(axis=1)
            cvs = (std_vals / mean_vals * 100).replace([np.inf, -np.inf], np.nan).dropna()
            
            if not cvs.empty:
                results.append({
                    "Species": species,
                    "Condition": cond,
                    "Mean_CV": cvs.mean(),
                    "Median_CV": cvs.median(),
                    "Count": len(cvs),
                })
    
    return pd.DataFrame(results)


@st.cache_data
def create_stacked_species_barplot(
    df: pd.DataFrame, numeric_cols: List[str], species_col: str
) -> go.Figure | None:
    """Create stacked bar plot showing protein count per species for each sample."""
    if species_col not in df.columns:
        return None
    
    all_species = df[species_col].dropna().unique()
    species_list = [s for s in SPECIES_ORDER if s in all_species]
    
    traces = []
    for species in species_list:
        species_df = df[df[species_col] == species]
        counts = [
            (species_df[col].notna() & (species_df[col] > 1.0)).sum()
            for col in numeric_cols
        ]
        
        traces.append(go.Bar(
            name=species,
            x=numeric_cols,
            y=counts,
            marker_color=SPECIES_COLORS.get(species, "#808080"),
        ))
    
    fig = go.Figure(data=traces)
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
    numeric_cols: List[str],
) -> pd.DataFrame:
    """Count peptides per protein using protein group counts."""
    if (protein_idx not in protein_df.columns 
        or peptide_idx not in peptide_df.columns
        or protein_species_col not in protein_df.columns):
        return pd.DataFrame()
    
    results = []
    for sample in numeric_cols:
        if sample not in peptide_df.columns:
            continue
        
        detected_peptides = peptide_df[peptide_df[sample] > 1.0]
        peptide_counts = detected_peptides[peptide_idx].value_counts()
        
        for protein_id, count in peptide_counts.items():
            protein_match = protein_df[protein_df[protein_idx] == protein_id]
            if not protein_match.empty:
                species = protein_match[protein_species_col].iloc[0]
                results.append({
                    "Sample": sample,
                    "Species": species,
                    "Peptide_Count": count,
                })
    
    return pd.DataFrame(results)


@st.cache_data
def create_peptides_per_protein_plot(ppp_df: pd.DataFrame) -> go.Figure | None:
    """Create box plot of peptides per protein by species and sample."""
    if ppp_df.empty:
        return None
    
    all_species = ppp_df["Species"].unique()
    species_list = [s for s in SPECIES_ORDER if s in all_species]
    samples = sorted(ppp_df["Sample"].unique())
    
    fig = go.Figure()
    for species in species_list:
        species_data = ppp_df[ppp_df["Species"] == species]
        x_data, y_data = [], []
        
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
protein_species_col = st.session_state.get("protein_species_col")
peptide_species_col = st.session_state.get("peptide_species_col")

if protein_model is None and peptide_model is None:
    st.warning("No data cached. Please upload data on the Data Upload page first.")
    render_navigation(back_page="pages/2_EDA.py", next_page=None)
    render_footer()
    st.stop()

tab_protein, tab_peptide = st.tabs(["Protein-level QC", "Peptide-level QC"])


def render_preprocessing(model: MSData | None, species_col: str | None, label: str):
    """Render preprocessing QC for protein or peptide level."""
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
    condition_names = sorted(cv_per_condition.columns)
    
    for cond in condition_names:
        cond_data = cv_per_condition[cond].dropna()
        cond_data_capped = cond_data[cond_data <= 100]
        
        if not cond_data_capped.empty:
            fig.add_trace(go.Violin(
                y=cond_data_capped,
                name=cond.replace("CV_", ""),
                box_visible=True,
                meanline_visible=True,
                fillcolor="rgba(135, 206, 235, 0.6)",
                line=dict(color="#008B8B", width=2),
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
    
    fig.add_hline(
        y=20, line_dash="dash", line_color="orange", line_width=2,
        annotation_text="20% threshold", annotation_position="right"
    )
    fig.add_hline(
        y=15, line_dash="dash", line_color="green", line_width=2,
        annotation_text="15% threshold", annotation_position="right"
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"cv_violin_{label}")
    
    # Show count of proteins >100% CV
    total_proteins = cv_per_condition.notna().any(axis=1).sum()
    above_100_count = (cv_per_condition > 100).any(axis=1).sum()
    if above_100_count > 0:
        pct = above_100_count / total_proteins * 100
        st.caption(f"ℹ️ {above_100_count} {label}s ({pct:.1f}%) have CV >100% (excluded from plot)")
    
    st.markdown("---")
    
    # Section 2: CV Summary Table
    st.markdown("### CV Summary per Condition")
    
    cv_summary = {}
    for col in cv_per_condition.columns:
        clean = cv_per_condition[col].dropna()
        if not clean.empty:
            cv_summary[col.replace("CV_", "Condition ")] = {
                "Mean CV%": f"{clean.mean():.1f}",
                "Median CV%": f"{clean.median():.1f}",
                "% <15%": f"{(clean < 15).sum() / len(clean) * 100:.1f}",
                "% <20%": f"{(clean < 20).sum() / len(clean) * 100:.1f}",
                f"N {label}s": len(clean),
            }
    
    summary_df = pd.DataFrame(cv_summary).T
    st.dataframe(summary_df, use_container_width=True)
    
    # Quality assessment
    all_cvs = cv_per_condition.to_numpy().ravel()
    all_cvs_clean = all_cvs[~np.isnan(all_cvs)]
    above_20_pct = (all_cvs_clean >= 20).sum() / len(all_cvs_clean) * 100 if all_cvs_clean.size else 0
    
    if above_20_pct > 20:
        st.warning(f"⚠️ {above_20_pct:.1f}% of {label}s have CV ≥20% within conditions. Consider preprocessing.")
    else:
        st.success(f"✓ Good technical reproducibility: {100 - above_20_pct:.1f}% of {label}s <20% CV")
    
    st.markdown("---")
    
    # Section 3: Species-specific CV (if species column available)
    if species_col and species_col in model.raw.columns:
        st.markdown("### CV% per Species and Condition")
        st.caption("Technical reproducibility breakdown by species")
        
        df_with_species = df_raw.copy()
        df_with_species[species_col] = model.raw[species_col]
        
        species_cv_df = compute_species_cv_per_condition(df_with_species, numeric_cols, species_col)
        
        if not species_cv_df.empty:
            pivot_mean = species_cv_df.pivot(index="Condition", columns="Species", values="Mean_CV")
            pivot_median = species_cv_df.pivot(index="Condition", columns="Species", values="Median_CV")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Mean CV% by Species & Condition")
                st.dataframe(
                    pivot_mean.style.format("{:.1f}").background_gradient(cmap="RdYlGn_r", vmin=0, vmax=30),
                    use_container_width=True
                )
            
            with col2:
                st.markdown("#### Median CV% by Species & Condition")
                st.dataframe(
                    pivot_median.style.format("{:.1f}").background_gradient(cmap="RdYlGn_r", vmin=0, vmax=30),
                    use_container_width=True
                )
    
    st.markdown("---")
    
    # Section 4: Species distribution per sample
    if species_col and species_col in model.raw.columns:
        st.markdown(f"### {label.capitalize()} Distribution by Species")
        st.caption(f"Number of detected {label}s per species in each sample")
        
        df_with_species = model.raw_filled[numeric_cols].copy()
        df_with_species[species_col] = model.raw[species_col]
        
        fig_species = create_stacked_species_barplot(df_with_species, numeric_cols, species_col)
        if fig_species:
            st.plotly_chart(fig_species, use_container_width=True, key=f"species_stack_{label}")
    
    st.markdown("---")
    
    # Section 5: Peptides per protein (protein-level only)
    if label == "protein" and peptide_model and protein_idx and peptide_idx and species_col:
        st.markdown("### Peptides per Protein Analysis")
        st.caption("Number of peptides detected per protein across samples and species")
        
        ppp_df = compute_peptides_per_protein(
            model.raw_filled, peptide_model.raw_filled,
            protein_idx, peptide_idx, species_col, numeric_cols
        )
        
        if not ppp_df.empty:
            # Summary stats
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Mean peptides/protein", f"{ppp_df['Peptide_Count'].mean():.1f}")
            with c2:
                st.metric("Median peptides/protein", f"{ppp_df['Peptide_Count'].median():.1f}")
            with c3:
                single_pep_pct = (ppp_df['Peptide_Count'] == 1).sum() / len(ppp_df) * 100
                st.metric("Single-peptide proteins", f"{single_pep_pct:.1f}%")
            with c4:
                above_15 = (ppp_df['Peptide_Count'] > 15).sum()
                st.metric("Proteins >15 peptides", f"{above_15}")
            
            # Box plot (capped at 15)
            ppp_df_capped = ppp_df[ppp_df['Peptide_Count'] <= 15].copy()
            
            if not ppp_df_capped.empty:
                fig_ppp = create_peptides_per_protein_plot(ppp_df_capped)
                if fig_ppp:
                    fig_ppp.update_layout(title="Peptides per protein by species and sample (capped at 15)")
                    st.plotly_chart(fig_ppp, use_container_width=True, key=f"ppp_{label}")
                
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
            species_stats = species_stats[
                ['Count', '% of Total', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Single-peptide %']
            ]
            
            # Sort by species order
            species_stats = species_stats.reindex(
                [s for s in SPECIES_ORDER if s in species_stats.index]
            )
            
            st.dataframe(
                species_stats.style.background_gradient(
                    subset=['Mean', 'Median'], cmap='RdYlGn', vmin=1, vmax=10
                ),
                use_container_width=True
            )
            
            # Statistical comparison
            st.markdown("#### Statistical Comparison (ANOVA)")
            
            unique_species = ppp_df['Species'].unique()
            if len(unique_species) >= 2:
                species_groups = [
                    group['Peptide_Count'].values
                    for _, group in ppp_df.groupby('Species')
                ]
                
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
                    st.caption(
                        "💡 Significant differences suggest species-specific protein coverage or identification rates."
                    )
        else:
            st.info("No peptide-to-protein mapping found. Check that protein IDs match between datasets.")


with tab_protein:
    render_preprocessing(protein_model, protein_species_col, "protein")

with tab_peptide:
    render_preprocessing(peptide_model, peptide_species_col, "peptide")

render_navigation(back_page="pages/2_EDA.py", next_page="pages/4_Filtering.py")
render_footer()
