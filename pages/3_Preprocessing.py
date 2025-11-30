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
    """Compute CV% for each protein within each condition."""
    condition_map = extract_conditions(numeric_cols)
    
    # Group columns by condition
    conditions = {}
    for col in numeric_cols:
        cond = condition_map[col]
        conditions.setdefault(cond, []).append(col)
    
    # Compute CV for each condition
    cv_results = {}
    for cond, cols in conditions.items():
        if len(cols) > 1:
            mean_vals = df[cols].mean(axis=1)
            std_vals = df[cols].std(axis=1)
            cv_results[f"CV_{cond}"] = (std_vals / mean_vals * 100).replace([np.inf, -np.inf], np.nan)
    
    return pd.DataFrame(cv_results, index=df.index)


def compute_overall_cv(df: pd.DataFrame, numeric_cols: list[str]) -> pd.Series:
    """Compute CV% across all samples for each protein."""
    mean_vals = df[numeric_cols].mean(axis=1)
    std_vals = df[numeric_cols].std(axis=1)
    cv = (std_vals / mean_vals * 100).replace([np.inf, -np.inf], np.nan)
    return cv


def compute_species_cv_per_condition(df: pd.DataFrame, numeric_cols: list[str], species_col: str) -> pd.DataFrame:
    """Compute CV per condition for each species separately."""
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
            if len(cols) > 1:
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
def create_cv_violin_plot(cv_series: pd.Series) -> go.Figure:
    """Create violin plot of overall CVs."""
    clean_cvs = cv_series.dropna()
    
    fig = go.Figure(data=go.Violin(
        y=clean_cvs,
        box_visible=True,
        meanline_visible=True,
        fillcolor=TF_CHART_COLORS[0],
        opacity=0.6,
        x0="Overall CV%",
    ))
    
    fig.update_layout(
        title="Distribution of CV% across all proteins",
        yaxis_title="CV%",
        xaxis_title="",
        height=400,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#54585A"),
        showlegend=False,
    )
    
    # Add threshold lines
    fig.add_hline(y=20, line_dash="dash", line_color="orange", annotation_text="20% threshold")
    fig.add_hline(y=15, line_dash="dash", line_color="green", annotation_text="15% threshold")
    
    return fig


@st.cache_data
def create_stacked_species_barplot(df: pd.DataFrame, numeric_cols: list[str], species_col: str) -> go.Figure:
    """Create stacked bar plot showing protein count per species for each sample."""
    if species_col not in df.columns:
        return None
    
    species_list = sorted(df[species_col].dropna().unique())
    
    # Count proteins per species for each sample (non-missing values)
    data = []
    for species in species_list:
        counts = []
        species_df = df[df[species_col] == species]
        for col in numeric_cols:
            # Count non-missing proteins (not NaN and not filled as 1.0)
            non_missing = species_df[col].notna() & (species_df[col] > 1.0)
            counts.append(non_missing.sum())
        data.append(go.Bar(name=species, x=numeric_cols, y=counts))
    
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

tab_protein, tab_peptide = st.tabs(["Protein data", "Peptide data"])


def render_preprocessing(model: MSData | None, species_col: str | None, label: str):
    if model is None:
        st.info(f"No {label} data uploaded yet")
        return

    numeric_cols = model.numeric_cols
    df_raw = model.raw_filled[numeric_cols]
    
    st.caption(f"**{len(df_raw):,} {label}s** × **{len(numeric_cols)} samples**")
    
    # Section 1: Overall CV Analysis
    st.markdown("### Coefficient of Variation (CV%) Analysis")
    st.caption("CV thresholds: <15% excellent, <20% acceptable, >20% review recommended [web:129][web:167]")
    
    overall_cv = compute_overall_cv(df_raw, numeric_cols)
    cv_per_condition = compute_cv_per_condition(df_raw, numeric_cols)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Overall CV stats
        st.markdown("#### Overall CV Statistics")
        clean_cvs = overall_cv.dropna()
        if len(clean_cvs) > 0:
            st.metric("Mean CV%", f"{clean_cvs.mean():.1f}%")
            st.metric("Median CV%", f"{clean_cvs.median():.1f}%")
            
            below_15 = (clean_cvs < 15).sum() / len(clean_cvs) * 100
            below_20 = (clean_cvs < 20).sum() / len(clean_cvs) * 100
            above_20 = (clean_cvs >= 20).sum() / len(clean_cvs) * 100
            
            st.metric("Proteins <15% CV", f"{below_15:.1f}%")
            st.metric("Proteins <20% CV", f"{below_20:.1f}%")
            st.metric("Proteins ≥20% CV", f"{above_20:.1f}%")
            
            if above_20 > 20:
                st.warning("⚠️ >20% of proteins have CV ≥20%. Consider preprocessing.")
            else:
                st.success("✓ Good technical reproducibility")
    
    with col2:
        # Violin plot
        fig_violin = create_cv_violin_plot(overall_cv)
        st.plotly_chart(fig_violin, width="stretch", key=f"cv_violin_{label}")
    
    st.markdown("---")
    
    # Section 2: CV per condition
    st.markdown("### CV% per Condition")
    st.caption("Replicate reproducibility within each experimental condition")
    
    if not cv_per_condition.empty:
        # Summary table
        cv_summary = {}
        for col in cv_per_condition.columns:
            clean = cv_per_condition[col].dropna()
            if len(clean) > 0:
                cv_summary[col] = {
                    "Mean CV%": f"{clean.mean():.1f}",
                    "Median CV%": f"{clean.median():.1f}",
                    "% <15%": f"{(clean < 15).sum() / len(clean) * 100:.1f}",
                    "% <20%": f"{(clean < 20).sum() / len(clean) * 100:.1f}",
                }
        
        summary_df = pd.DataFrame(cv_summary).T
        st.dataframe(summary_df, width="stretch")
    
    st.markdown("---")
    
    # Section 3: Species-specific CV (if species column available)
    if species_col and species_col in model.raw.columns:
        st.markdown("### CV% per Species and Condition")
        st.caption("Technical reproducibility breakdown by species")
        
        # Add species column to df_raw for filtering
        df_with_species = df_raw.copy()
        df_with_species[species_col] = model.raw[species_col]
        
        species_cv_df = compute_species_cv_per_condition(df_with_species, numeric_cols, species_col)
        
        if not species_cv_df.empty:
            # Pivot for better display
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
        st.markdown("### Protein Distribution by Species")
        st.caption("Number of detected proteins per species in each sample")
        
        df_with_species = model.raw_filled[numeric_cols].copy()
        df_with_species[species_col] = model.raw[species_col]
        
        fig_species = create_stacked_species_barplot(df_with_species, numeric_cols, species_col)
        if fig_species:
            st.plotly_chart(fig_species, width="stretch", key=f"species_stack_{label}")


with tab_protein:
    render_preprocessing(protein_model, protein_species_col, "protein")

with tab_peptide:
    render_preprocessing(peptide_model, peptide_species_col, "peptide")

render_navigation(back_page="pages/2_EDA.py", next_page=None)
render_footer()
