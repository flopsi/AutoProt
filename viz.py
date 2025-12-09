"""
helpers/viz.py - Visualization Functions Module

Consolidated plotting functions for exploratory data analysis and statistical output.
All functions return Plotly figures for interactive visualization.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from typing import Tuple, List, Dict, Optional


# ============================================================================
# DISTRIBUTION & EXPLORATORY PLOTS
# ============================================================================

@st.cache_data(ttl=3600)
def create_density_histograms(
    df: pd.DataFrame,
    numeric_cols: List[str],
    title: str = "Intensity Distribution",
    theme: str = "plotly_white",
    nbins: int = 50
) -> go.Figure:
    """
    Create overlaid density histograms for all numeric columns.
    
    Args:
        df: Input DataFrame
        numeric_cols: Columns to plot
        title: Plot title
        theme: Plotly theme
        nbins: Number of histogram bins
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    for col in numeric_cols:
        values = df[col].dropna()
        if len(values) == 0:
            continue
        
        fig.add_trace(go.Histogram(
            x=values,
            name=col,
            opacity=0.7,
            nbinsx=nbins,
            hovertemplate='<b>%{fullData.name}</b><br>Value: %{x}<br>Count: %{y}<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Intensity",
        yaxis_title="Count",
        barmode='overlay',
        hovermode='x unified',
        template=theme,
        height=500,
        showlegend=True
    )
    
    return fig


@st.cache_data(ttl=3600)
def create_box_plot_by_condition(
    df: pd.DataFrame,
    numeric_cols: List[str],
    condition_mapping: Dict[str, str],
    title: str = "Abundance by Condition",
    theme: str = "plotly_white"
) -> go.Figure:
    """
    Create grouped box plots by experimental condition.
    
    Args:
        df: Input DataFrame
        numeric_cols: Sample columns
        condition_mapping: Column to condition mapping
        title: Plot title
        theme: Plotly theme
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    conditions = sorted(set(condition_mapping.values()))
    
    for condition in conditions:
        cols = [c for c in numeric_cols if condition_mapping.get(c) == condition]
        if not cols:
            continue
        
        # Combine all values for this condition
        values = []
        for col in cols:
            values.extend(df[col].dropna().values)
        
        fig.add_trace(go.Box(
            y=values,
            name=condition,
            boxmean='sd',
            hovertemplate='<b>%{fullData.name}</b><br>Value: %{y}<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        yaxis_title="Intensity",
        xaxis_title="Condition",
        template=theme,
        height=500,
        hovermode='y unified'
    )
    
    return fig


@st.cache_data(ttl=3600)
def create_violin_plot_by_condition(
    df: pd.DataFrame,
    numeric_cols: List[str],
    condition_mapping: Dict[str, str],
    title: str = "Distribution by Condition",
    theme: str = "plotly_white"
) -> go.Figure:
    """
    Create violin plots by condition.
    
    Args:
        df: Input DataFrame
        numeric_cols: Sample columns
        condition_mapping: Column to condition mapping
        title: Plot title
        theme: Plotly theme
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    conditions = sorted(set(condition_mapping.values()))
    
    for condition in conditions:
        cols = [c for c in numeric_cols if condition_mapping.get(c) == condition]
        if not cols:
            continue
        
        values = []
        for col in cols:
            values.extend(df[col].dropna().values)
        
        fig.add_trace(go.Violin(
            y=values,
            name=condition,
            box_visible=True,
            meanline_visible=True,
            hovertemplate='<b>%{fullData.name}</b><br>Value: %{y}<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        yaxis_title="Intensity",
        xaxis_title="Condition",
        template=theme,
        height=500,
        hovermode='y unified'
    )
    
    return fig


# ============================================================================
# STATISTICAL DIAGNOSTIC PLOTS
# ============================================================================

@st.cache_data(ttl=3600)
def create_qq_plot(
    data: np.ndarray,
    title: str = "Q-Q Plot",
    theme: str = "plotly_white"
) -> go.Figure:
    """
    Create Q-Q plot for normality assessment.
    
    Args:
        data: Values to plot (1D array)
        title: Plot title
        theme: Plotly theme
    
    Returns:
        Plotly figure
    """
    from scipy import stats
    
    # Remove NaN values
    data_clean = data[~np.isnan(data)]
    
    if len(data_clean) < 3:
        # Not enough data for Q-Q plot
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for Q-Q plot")
        return fig
    
    # Compute theoretical quantiles
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(data_clean)))
    sample_quantiles = np.sort(data_clean)
    
    fig = go.Figure()
    
    # Add Q-Q plot points
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=sample_quantiles,
        mode='markers',
        name='Data',
        marker=dict(size=6, color='steelblue'),
        hovertemplate='Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>'
    ))
    
    # Add reference line (y=x)
    min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
    max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Normal Reference',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Theoretical Quantiles (Normal Distribution)",
        yaxis_title="Sample Quantiles",
        template=theme,
        height=500,
        hovermode='closest',
        showlegend=True
    )
    
    return fig


# ============================================================================
# DIMENSIONALITY REDUCTION PLOTS
# ============================================================================

@st.cache_data(ttl=3600)
def create_pca_plot(
    df: pd.DataFrame,
    numeric_cols: List[str],
    condition_mapping: Optional[Dict[str, str]] = None,
    n_components: int = 2,
    title: str = "PCA Plot"
) -> go.Figure:
    """
    Create PCA plot with optional sample grouping.
    
    Args:
        df: Input DataFrame (proteins as rows, samples as columns)
        numeric_cols: Columns for PCA
        condition_mapping: Column to group mapping (color by group)
        n_components: 2 or 3
        title: Plot title
    
    Returns:
        Plotly figure
    """
    # Prepare data
    data_matrix = df[numeric_cols].dropna().values
    
    if data_matrix.shape[0] < 3:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient samples for PCA")
        return fig
    
    # Compute PCA
    pca = PCA(n_components=min(n_components, len(numeric_cols)))
    components = pca.fit_transform(data_matrix.T)  # Transpose: samples as rows
    
    # Create color mapping if conditions provided
    colors = None
    if condition_mapping:
        color_map = {v: i for i, v in enumerate(set(condition_mapping.values()))}
        colors = [color_map.get(condition_mapping.get(col), 0) for col in numeric_cols]
    
    if n_components == 2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=components[:, 0],
            y=components[:, 1],
            mode='markers+text',
            text=numeric_cols,
            textposition='top center',
            marker=dict(
                size=8,
                color=colors,
                colorscale='Viridis' if colors else None,
                showscale=True if colors else False,
                colorbar=dict(title="Condition") if colors else None
            ),
            hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
            height=500,
            hovermode='closest'
        )
    
    elif n_components == 3:
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=components[:, 0],
            y=components[:, 1],
            z=components[:, 2],
            mode='markers+text',
            text=numeric_cols,
            marker=dict(
                size=5,
                color=colors,
                colorscale='Viridis' if colors else None,
                showscale=True if colors else False
            ),
            hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
                zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]:.1%})"
            ),
            height=600
        )
    
    return fig


# ============================================================================
# HEATMAPS
# ============================================================================

@st.cache_data(ttl=3600)
def create_heatmap(
    df: pd.DataFrame,
    numeric_cols: List[str],
    title: str = "Abundance Heatmap",
    n_top: int = 50
) -> go.Figure:
    """
    Create heatmap of top variable proteins/peptides.
    
    Args:
        df: Input DataFrame
        numeric_cols: Sample columns
        title: Plot title
        n_top: Top N rows by variance to display
    
    Returns:
        Plotly figure
    """
    # Select top variable features
    subset = df[numeric_cols].copy()
    variances = subset.var(axis=1)
    top_idx = variances.nlargest(n_top).index
    subset = subset.loc[top_idx]
    
    # Log transform for better visualization
    subset_log = np.log2(subset.clip(lower=1))
    
    # Normalize by row
    subset_norm = (subset_log - subset_log.mean(axis=1).values.reshape(-1, 1)) / (subset_log.std(axis=1).values.reshape(-1, 1) + 1e-6)
    
    fig = go.Figure(data=go.Heatmap(
        z=subset_norm.values,
        x=numeric_cols,
        y=subset.index.astype(str)[:20] + ['...'] if len(subset) > 20 else subset.index,
        colorscale='RdBu_r',
        zmid=0,
        hovertemplate='Sample: %{x}<br>Protein: %{y}<br>Z-score: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Sample",
        yaxis_title="Protein/Peptide",
        height=600,
        width=900
    )
    
    return fig


# ============================================================================
# DIFFERENTIAL EXPRESSION PLOTS
# ============================================================================

@st.cache_data(ttl=3600)
def create_volcano_plot(
    log2fc: np.ndarray,
    neg_log10_pval: np.ndarray,
    regulation: List[str],
    fc_threshold: float = 1.0,
    pval_threshold: float = 0.05,
    protein_ids: Optional[List[str]] = None,
    title: str = "Volcano Plot",
    theme: str = "plotly_white"
) -> go.Figure:
    """
    Create volcano plot (log2 fold change vs -log10 p-value).
    
    Args:
        log2fc: Log2 fold change values
        neg_log10_pval: -Log10 p-values
        regulation: Classification (upregulated/downregulated/not_significant/not_tested)
        fc_threshold: Fold change threshold lines
        pval_threshold: P-value threshold
        protein_ids: Optional protein IDs for hover text
        title: Plot title
        theme: Plotly theme
    
    Returns:
        Plotly figure
    """
    # Color mapping
    color_map = {
        'upregulated': '#E74C3C',       # Red
        'downregulated': '#3498DB',    # Blue
        'not_significant': '#95A5A6',  # Gray
        'not_tested': '#BDC3C7'        # Light gray
    }
    
    colors = [color_map.get(reg, '#000000') for reg in regulation]
    
    fig = go.Figure()
    
    # Add points by regulation status
    for reg_status in ['downregulated', 'upregulated', 'not_significant', 'not_tested']:
        mask = np.array(regulation) == reg_status
        
        if not mask.any():
            continue
        
        fig.add_trace(go.Scatter(
            x=log2fc[mask],
            y=neg_log10_pval[mask],
            mode='markers',
            name=reg_status.replace('_', ' ').title(),
            marker=dict(
                color=color_map[reg_status],
                size=6,
                opacity=0.7
            ),
            text=protein_ids[mask] if protein_ids is not None else None,
            hovertemplate='<b>%{text}</b><br>log2FC: %{x:.2f}<br>-log10(p): %{y:.2f}<extra></extra>' if protein_ids is not None else None
        ))
    
    # Add threshold lines
    pval_line_y = -np.log10(pval_threshold)
    
    fig.add_hline(y=pval_line_y, line_dash='dash', line_color='gray', annotation_text='p=0.05')
    fig.add_vline(x=fc_threshold, line_dash='dash', line_color='gray')
    fig.add_vline(x=-fc_threshold, line_dash='dash', line_color='gray')
    
    fig.update_layout(
        title=title,
        xaxis_title="log2(Fold Change)",
        yaxis_title="-log10(p-value)",
        template=theme,
        height=600,
        hovermode='closest',
        showlegend=True
    )
    
    return fig


@st.cache_data(ttl=3600)
def create_ma_plot(
    mean_expr: np.ndarray,
    log2fc: np.ndarray,
    regulation: List[str],
    fc_threshold: float = 1.0,
    protein_ids: Optional[List[str]] = None,
    title: str = "MA Plot",
    theme: str = "plotly_white"
) -> go.Figure:
    """
    Create MA plot (Mean vs log2 fold change).
    
    Args:
        mean_expr: Mean expression values
        log2fc: Log2 fold changes
        regulation: Classification
        fc_threshold: Threshold lines
        protein_ids: Optional protein IDs
        title: Plot title
        theme: Plotly theme
    
    Returns:
        Plotly figure
    """
    color_map = {
        'upregulated': '#E74C3C',
        'downregulated': '#3498DB',
        'not_significant': '#95A5A6',
        'not_tested': '#BDC3C7'
    }
    
    fig = go.Figure()
    
    for reg_status in ['downregulated', 'upregulated', 'not_significant', 'not_tested']:
        mask = np.array(regulation) == reg_status
        
        if not mask.any():
            continue
        
        fig.add_trace(go.Scatter(
            x=mean_expr[mask],
            y=log2fc[mask],
            mode='markers',
            name=reg_status.replace('_', ' ').title(),
            marker=dict(
                color=color_map[reg_status],
                size=5,
                opacity=0.7
            ),
            text=protein_ids[mask] if protein_ids is not None else None,
            hovertemplate='<b>%{text}</b><br>Mean: %{x:.2f}<br>log2FC: %{y:.2f}<extra></extra>' if protein_ids is not None else None
        ))
    
    # Add fold change threshold lines
    fig.add_hline(y=fc_threshold, line_dash='dash', line_color='gray')
    fig.add_hline(y=-fc_threshold, line_dash='dash', line_color='gray')
    
    fig.update_layout(
        title=title,
        xaxis_title="Mean Expression (log2)",
        yaxis_title="log2(Fold Change)",
        xaxis_type='log',
        template=theme,
        height=600,
        hovermode='closest',
        showlegend=True
    )
    
    return fig


# ============================================================================
# QUALITY CONTROL PLOTS
# ============================================================================

@st.cache_data(ttl=3600)
def create_missing_data_heatmap(
    df: pd.DataFrame,
    numeric_cols: List[str],
    n_top: int = 50
) -> go.Figure:
    """
    Visualize missing data pattern (1 = present, 0 = missing).
    
    Args:
        df: Input DataFrame
        numeric_cols: Sample columns
        n_top: Top N rows with most missing data
    
    Returns:
        Plotly figure
    """
    # Get rows with most missing data
    missing_counts = df[numeric_cols].isna().sum(axis=1)
    top_missing_idx = missing_counts.nlargest(n_top).index
    
    # Create binary matrix: 1 = present, 0 = missing
    subset = df.loc[top_missing_idx, numeric_cols]
    missing_matrix = (~subset.isna()).astype(int)
    
    fig = go.Figure(data=go.Heatmap(
        z=missing_matrix.values,
        x=numeric_cols,
        y=missing_matrix.index.astype(str)[:20] + ['...'] if len(missing_matrix) > 20 else missing_matrix.index,
        colorscale=[[0, '#E74C3C'], [1, '#27AE60']],  # Red = missing, Green = present
        showscale=True,
        hovertemplate='Sample: %{x}<br>Protein: %{y}<br>Present: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Missing Data Pattern",
        xaxis_title="Sample",
        yaxis_title="Protein/Peptide",
        height=600,
        width=900
    )
    
    return fig


@st.cache_data(ttl=3600)
def create_valid_counts_by_sample(
    valid_counts: pd.DataFrame
) -> go.Figure:
    """
    Create bar plot of valid protein/peptide counts per sample.
    
    Args:
        valid_counts: DataFrame with 'Sample' and 'Count' columns
    
    Returns:
        Plotly figure
    """
    fig = px.bar(
        valid_counts,
        x='Sample',
        y='Count',
        title='Valid Proteins/Peptides per Sample',
        labels={'Count': 'Number of Valid Entries'},
        height=500
    )
    
    fig.update_traces(marker_color='steelblue')
    fig.update_layout(hovermode='x unified')
    
    return fig
