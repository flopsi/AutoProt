"""
helpers/viz.py - CONSOLIDATED PLOTTING MODULE (REFERENCE)
All plotting functions consolidated from 6 pages into single module
This file shows the structure and key functions to migrate

INSTRUCTIONS:
1. Copy your existing viz.py
2. Add these function signatures
3. Move all plotting code from pages to corresponding functions
4. Use @st.cache_data decorators where appropriate
5. Return plotly go.Figure objects (not st.plotly_chart)
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from plotnine import *
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional

# ============================================================================
# DISTRIBUTION & EXPLORATORY PLOTS
# ============================================================================

@st.cache_data(ttl=3600)
def create_density_histograms(
    df: pd.DataFrame,
    numeric_cols: List[str],
    title: str = "Distribution",
    theme: str = "plotly_white"
) -> go.Figure:
    """
    Create density histograms for all numeric columns.
    
    Args:
        df: Input DataFrame
        numeric_cols: Columns to plot
        title: Plot title
        theme: Plotly theme
        
    Returns:
        Plotly figure
    """
    # IMPLEMENT: Stack multiple histograms with density overlay
    # Return go.Figure
    pass


@st.cache_data(ttl=3600)
def create_box_plot_by_condition(
    df: pd.DataFrame,
    numeric_cols: List[str],
    condition_mapping: Dict[str, str],
    theme: str = "plotly_white"
) -> go.Figure:
    """
    Create grouped box plots by experimental condition.
    
    Args:
        df: Input DataFrame
        numeric_cols: Sample columns
        condition_mapping: Column to condition mapping
        theme: Plotly theme
        
    Returns:
        Plotly figure
    """
    # IMPLEMENT: Box plots grouped by condition
    # Return go.Figure
    pass


@st.cache_data(ttl=3600)
def create_violin_plot_by_condition(
    df: pd.DataFrame,
    numeric_cols: List[str],
    condition_mapping: Dict[str, str]
) -> go.Figure:
    """Create violin plots by condition."""
    # IMPLEMENT: Violin plots
    pass

# ============================================================================
# STATISTICAL DIAGNOSTIC PLOTS
# ============================================================================

@st.cache_data(ttl=3600)
def create_qq_plot(
    data: np.ndarray,
    title: str = "Q-Q Plot"
) -> go.Figure:
    """
    Create Q-Q plot for normality assessment.
    
    Args:
        data: Values to plot (1D array)
        title: Plot title
        
    Returns:
        Plotly figure
    """
    # IMPLEMENT: Q-Q plot with reference line
    # Return go.Figure
    pass


@st.cache_data(ttl=3600)
def create_mean_variance_plot(
    means: np.ndarray,
    variances: np.ndarray,
    regulation: Optional[List[str]] = None
) -> go.Figure:
    """
    Create mean-variance plot (useful for variance stabilization check).
    
    Args:
        means: Mean values per protein
        variances: Variance values per protein
        regulation: Optional regulation labels (up/down/ns)
        
    Returns:
        Plotly figure
    """
    # IMPLEMENT: Scatter plot with color by regulation
    # Return go.Figure
    pass

# ============================================================================
# DIMENSIONALITY REDUCTION PLOTS
# ============================================================================

@st.cache_data(ttl=3600)
def create_pca_plot(
    df: pd.DataFrame,
    numeric_cols: List[str],
    group_mapping: Optional[Dict[str, str]] = None,
    n_components: int = 2,
    title: str = "PCA Plot"
) -> go.Figure:
    """
    Create PCA plot with optional sample grouping.
    
    Args:
        df: Input DataFrame
        numeric_cols: Columns for PCA
        group_mapping: Column to group mapping (color by group)
        n_components: 2 or 3
        title: Plot title
        
    Returns:
        Plotly figure
    """
    # IMPLEMENT: PCA with optional 3D
    # Return go.Figure
    pass


@st.cache_data(ttl=3600)
def create_tsne_plot(
    df: pd.DataFrame,
    numeric_cols: List[str],
    group_mapping: Optional[Dict[str, str]] = None
) -> go.Figure:
    """Create t-SNE plot."""
    # IMPLEMENT: t-SNE visualization
    pass

# ============================================================================
# HEATMAPS & CLUSTERING
# ============================================================================

@st.cache_data(ttl=3600)
def create_heatmap_clustered(
    df: pd.DataFrame,
    numeric_cols: List[str],
    sample_dendrogram: bool = True,
    protein_dendrogram: bool = False,
    n_top: int = 50
) -> go.Figure:
    """
    Create hierarchically clustered heatmap.
    
    Args:
        df: Input DataFrame (proteins as rows)
        numeric_cols: Sample columns
        sample_dendrogram: Include sample clustering
        protein_dendrogram: Include protein clustering
        n_top: Top N proteins to show (by variance)
        
    Returns:
        Plotly figure
    """
    # IMPLEMENT: Clustered heatmap (seaborn or plotly)
    # Return go.Figure
    pass

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
    title: str = "Volcano Plot"
) -> go.Figure:
    """
    Create volcano plot (log2 fold change vs -log10 p-value).
    
    Args:
        log2fc: Log2 fold change values
        neg_log10_pval: -Log10 p-values
        regulation: Classification (upregulated/downregulated/not_significant/not_tested)
        fc_threshold: Fold change threshold lines
        pval_threshold: P-value threshold line
        protein_ids: Optional protein IDs for hover text
        title: Plot title
        
    Returns:
        Plotly figure
    """
    # IMPLEMENT: Volcano plot with threshold lines
    # Color by regulation
    # Return go.Figure
    pass


@st.cache_data(ttl=3600)
def create_ma_plot(
    mean_expr: np.ndarray,
    log2fc: np.ndarray,
    regulation: List[str],
    fc_threshold: float = 1.0
) -> go.Figure:
    """
    Create MA plot (Mean vs log2 fold change).
    
    Args:
        mean_expr: Mean expression values
        log2fc: Log2 fold changes
        regulation: Classification
        fc_threshold: Threshold lines
        
    Returns:
        Plotly figure
    """
    # IMPLEMENT: MA plot
    pass

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
    Visualize missing data pattern (black = missing, white = present).
    
    Args:
        df: Input DataFrame
        numeric_cols: Sample columns
        n_top: Top N rows with most missing data
        
    Returns:
        Plotly figure
    """
    # IMPLEMENT: Missing data heatmap
    pass


@st.cache_data(ttl=3600)
def create_valid_counts_by_sample(
    valid_counts: pd.DataFrame
) -> go.Figure:
    """
    Create bar plot of valid protein/peptide counts per sample.
    
    Args:
        valid_counts: DataFrame with sample and count columns
        
    Returns:
        Plotly figure
    """
    # IMPLEMENT: Bar plot
    pass

# ============================================================================
# NORMALITY & TRANSFORM ASSESSMENT
# ============================================================================

@st.cache_data(ttl=3600)
def create_transform_comparison_plot(
    transform_stats: pd.DataFrame
) -> go.Figure:
    """
    Visualize normality scores for different transformations.
    
    Args:
        transform_stats: DataFrame from compute_transform_comparison()
        
    Returns:
        Plotly figure
    """
    # IMPLEMENT: Bar plot comparing normality scores
    pass

# ============================================================================
# UTILITY PLOTTING FUNCTIONS
# ============================================================================

def add_threshold_lines(
    fig: go.Figure,
    fc_threshold: float,
    pval_threshold: float
) -> go.Figure:
    """Add horizontal/vertical threshold lines to figure."""
    # IMPLEMENT: Add lines for volcano plot thresholds
    return fig


def color_by_regulation(regulation: List[str]) -> List[str]:
    """Map regulation labels to colors."""
    color_map = {
        'upregulated': '#E74C3C',      # Red
        'downregulated': '#3498DB',    # Blue
        'not_significant': '#95A5A6',  # Gray
        'not_tested': '#BDC3C7'        # Light gray
    }
    return [color_map.get(r, '#000000') for r in regulation]


# ============================================================================
# COMPOSITE/DASHBOARD PLOTS
# ============================================================================

@st.cache_data(ttl=3600)
def create_qc_dashboard(
    df: pd.DataFrame,
    numeric_cols: List[str],
    condition_mapping: Dict[str, str],
    valid_counts: pd.DataFrame
) -> go.Figure:
    """
    Create composite QC dashboard with multiple subplots.
    
    Returns:
        Plotly figure with subplots
    """
    # IMPLEMENT: Subplot figure with:
    # - Box plots
    # - Missing data
    # - Valid counts
    # - Distribution
    pass

# ============================================================================
# NOTES FOR IMPLEMENTATION
# ============================================================================

"""
MIGRATION CHECKLIST:

1. Move all plotting code from pages into these functions
2. Each function returns go.Figure (not st.plotly_chart)
3. Call st.plotly_chart(fig) from pages
4. Add @st.cache_data(ttl=3600) to expensive plots
5. Use consistent color schemes
6. Add proper docstrings
7. Test each plot function independently

PAGES THAT HAVE PLOTTING CODE:
- 2_Visual_EDA_proteins.py: box, PCA, distributions
- 3_Statistical_EDA.py: Q-Q, distributions, transform comparison
- 4_Quality_Overview.py: Missing data, valid counts, distributions
- 5_Differential_Expression.py: Volcano, MA, results table

Move all of these into helpers/viz.py functions and call from pages.
"""
