import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass
from typing import List

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
    
    # Get species in desired order
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
        height=400,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#54585A"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return fig


def compute_peptides_per_protein(
    protein_model: MSData,
    peptide_model: MSData,
    protein_idx: str,
    peptide_idx: str,
    peptide_seq_col: str,
    protein_species_col: str,
    numeric_cols: list[str]
) -> pd.DataFrame:
    """Compute peptides per protein for each species and sample."""
    if not all([protein_model, peptide_model, protein_idx, peptide_idx, peptide_seq_col]):
        return pd.DataFrame()
    
    protein_df = protein_model.raw_filled
    peptide_df = peptide_model.raw_filled
    
    if protein_idx not in protein_df.columns or peptide_idx not in peptide_df.columns:
        return pd.DataFrame()
    
    if peptide_seq_col not in peptide_df.columns:
        return pd.DataFrame()
    
    results = []
    
    for sample in numeric_cols:
        if sample not in protein_df.columns or sample not in peptide_df.columns:
            continue
        
        # Get proteins detected in this sample
        detected_proteins = protein_df[protein_df[sample] > 1.0]
        
        for species in detected_proteins[protein_species_col].dropna().unique():
            species_proteins = detected_proteins[detected_proteins[protein_species_col] == species]
            
            for protein_id in species_proteins[protein_idx]:
                # Find matching peptides
                matching_peptides = peptide_df[peptide_df[peptide_idx].str.contains(str(protein_id), na=False)]
                detected_peptides = matching_peptides[matching_peptides[sample] > 1.0]
                
                peptide_count = len(detected_peptides)
                
                if peptide_count > 0:
                    results.append({
                        "Sample": sample,
                        "Species": species,
                        "Protein": protein_id,
                        "Peptide_Count": peptide_count,
                    })
    
    return pd.DataFrame(results)


@st.cache_data
def create_peptides_per_protein_plot(ppp_df: pd.DataFrame) -> go.Figure:
    """Create box plot of peptides per protein by species and sample."""
    if ppp_df.empty:
        return None
    
    # Get species in order
    all_species = ppp_df["Species"].unique()
    species_list = [s for s in SPECIES_ORDER if s in all_species]
    
    fig = go.Figure()
    
    for species in species_list:
        species_data = ppp_df[ppp_df["Species"] == species]
        
        for sample in sorted(species_data["Sample"].unique()):
            sample_data = species_data[species_data["Sample"] == sample]
            
            fig.add_trace(go.Box(
                y=sample_data["Peptide_Count"],
                name=f"{species}",
                x=[sample] * len(sample_data),
                marker_color=SPECIES_COLORS.get(species, "#808080"),
                legendgroup=species,
                showlegend=(sample == sorted(species_data["Sample"].unique())[0]),
            ))
    
    fig.update_layout(
        title="Peptides per protein by species and sample",
        xaxis_title="Sample",
        yaxis_title="Peptides per protein",
        height=400,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#54585A"),
        boxmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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
                opacity=0.6,
            ))
    
    fig.update_layout(
        title="CV% distribution per condition (capped at 100% for display)",
        yaxis_title="CV%",
        xaxis_title="Condition",
        height=400,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#54585A"),
        showlegend=False,
    )
    
    fig.add_hline(y=20, line_dash="dash", line_color="orange", annotation_text="20% threshold")
    fig.add_hline(y=15, line_dash="dash", line_color="green", annotation_text="15% threshold")
    
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
                st.dataframe(pivot_mean.style.format("{:.1f}").background_gradient(cmap="RdYlGn_r", vmin=0, vmax=30), width="stretch")
            
            with col2:
                st.markdown("#### Median CV% by Species & Condition")
                st.dataframe(pivot_median.style.format("{:.1f}").background_gradient(cmap="RdYlGn_r", vmin=0, vmax=30), width="stretch")
    
    st.markdown("---")
    
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
    
    # Section 5: Peptides per protein (only for protein-level QC and if peptide data available)
    if label == "protein" and peptide_model is not None and protein_idx and peptide_idx and peptide_seq_col:
        st.markdown("### Peptides per Protein Analysis")
        st.caption("Number of peptides detected per protein across samples and species")
        
        with st.spinner("Computing peptides per protein..."):
            ppp_df = compute_peptides_per_protein(
                model,
                peptide_model,
                protein_idx,
                peptide_idx,
                peptide_seq_col,
                species_col,
                numeric_cols
            )
        
        if not ppp_df.empty:
            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean peptides/protein", f"{ppp_df['Peptide_Count'].mean():.1f}")
            with col2:
                st.metric("Median peptides/protein", f"{ppp_df['Peptide_Count'].median():.1f}")
            with col3:
                single_peptide_pct = (ppp_df['Peptide_Count'] == 1).sum() / len(ppp_df) * 100
                st.metric("Single-peptide proteins", f"{single_peptide_pct:.1f}%")
            
            # Box plot
            fig_ppp = create_peptides_per_protein_plot(ppp_df)
            if fig_ppp:
                st.plotly_chart(fig_ppp, width="stretch", key=f"ppp_{label}")
        else:
            st.info("No peptide-to-protein mapping found. Check that protein IDs match between datasets.")


with tab_protein:
    render_preprocessing(protein_model, protein_species_col, "protein")

with tab_peptide:
    render_preprocessing(peptide_model, peptide_species_col, "peptide")

render_navigation(back_page="pages/2_EDA.py", next_page=None)
render_footer()
