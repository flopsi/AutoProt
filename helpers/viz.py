"""
helpers/viz.py

Visualization functions for proteomics data analysis
Consolidates all plotting: distributions, PCA, heatmaps, volcano, QC dashboards
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde, probplot
from helpers.core import get_theme, FONT_FAMILY
import streamlit as st

# ============================================================================
# DISTRIBUTION PLOTS
# Histograms with KDE overlays and statistical annotations
# ============================================================================

@st.cache_data(ttl=1800)
def create_density_histograms(
    df: pd.DataFrame,
    numeric_cols: list,
    transform_name: str,
    theme: dict,
    max_plots: int = 6
) -> go.Figure:
    """
    Create density histograms (histogram + KDE overlay) for each sample.
    Shows mean line and ±2σ shaded region.
    
    Args:
        df: Data to plot (already transformed)
        numeric_cols: List of numeric columns to plot
        transform_name: Name of transformation applied (for title)
        theme: Theme colors dictionary
        max_plots: Maximum number of plots (default 6 for 2×3 grid)
    
    Returns:
        Plotly figure with subplots
    """
    # Limit to max_plots
    cols_to_plot = numeric_cols[:max_plots]
    n_samples = len(cols_to_plot)
    
    # Determine grid layout
    if n_samples <= 3:
        n_rows, n_cols = 1, n_samples
    else:
        n_rows, n_cols = 2, 3
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"{col}" for col in cols_to_plot],
        vertical_spacing=0.15,
        horizontal_spacing=0.10
    )
    
    for idx, col in enumerate(cols_to_plot):
        row = (idx // n_cols) + 1
        col_pos = (idx % n_cols) + 1
        
        # Get data (filter > 1.0 and drop NaN)
        values = df[col][df[col] > 1.0].dropna().values
        
        if len(values) < 10:
            # Add "No data" annotation
            fig.add_annotation(
                text="Insufficient data",
                xref=f"x{idx+1}" if idx > 0 else "x",
                yref=f"y{idx+1}" if idx > 0 else "y",
                x=0.5,
                y=0.5,
                showarrow=False,
                row=row,
                col=col_pos
            )
            continue
        
        # Calculate statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Create histogram
        hist, bin_edges = np.histogram(values, bins=40)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        
        # Add histogram bars
        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=hist,
                width=bin_width * 0.9,
                marker=dict(
                    color=theme['accent'],
                    opacity=0.6,
                    line=dict(color=theme['text_primary'], width=0.5)
                ),
                showlegend=False,
                name="Histogram",
                hovertemplate="Intensity: %{x:.2f}<br>Count: %{y}<extra></extra>"
            ),
            row=row, col=col_pos
        )
        
        # Add KDE (kernel density estimate) overlay
        try:
            kde = gaussian_kde(values)
            x_range = np.linspace(values.min(), values.max(), 200)
            kde_values = kde(x_range)
            # Scale KDE to match histogram height
            kde_scaled = kde_values * len(values) * bin_width
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=kde_scaled,
                    mode='lines',
                    line=dict(color='red', width=2),
                    showlegend=False,
                    name="Density",
                    hovertemplate="Intensity: %{x:.2f}<br>Density: %{y:.2f}<extra></extra>"
                ),
                row=row, col=col_pos
            )
        except:
            pass  # Skip KDE if it fails
        
        # Add mean line
        y_max = max(hist) if len(hist) > 0 else 1
        fig.add_trace(
            go.Scatter(
                x=[mean_val, mean_val],
                y=[0, y_max],
                mode='lines',
                line=dict(dash='dash', color='darkred', width=1.5),
                showlegend=False,
                name="Mean",
                hovertemplate=f"Mean: {mean_val:.2f}<extra></extra>"
            ),
            row=row, col=col_pos
        )
        
        # Update axes
        fig.update_xaxes(
            title_text="Intensity",
            showgrid=True,
            gridcolor=theme['grid'],
            row=row, col=col_pos
        )
        
        fig.update_yaxes(
            title_text="Count",
            showgrid=True,
            gridcolor=theme['grid'],
            row=row, col=col_pos
        )
    
    # Update layout
    fig.update_layout(
        height=400 * n_rows,
        title_text=f"Sample Distributions - {transform_name}",
        title_font_size=16,
        showlegend=False,
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['paper_bg'],
        font=dict(
            family=FONT_FAMILY,
            size=11,
            color=theme['text_primary']
        ),
        hovermode="closest"
    )
    
    return fig

# ============================================================================
# TRANSFORMATION EVALUATION
# Before/after diagnostics for assessing transformation quality
# ============================================================================

def create_raw_row_figure(
    df_raw: pd.DataFrame,
    raw_cols: list,
    title: str = "Raw Data Diagnostics",
) -> go.Figure:
    """
    Single row (1×3) diagnostic panel for raw data:
    - Col 1: Distribution histogram with mean and ±2σ
    - Col 2: Q-Q plot for normality assessment
    - Col 3: Mean-variance relationship
    
    Args:
        df_raw: Raw untransformed data
        raw_cols: Column names to analyze
        title: Plot title
    
    Returns:
        Plotly figure with 3 subplots
    """
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Raw Intensities",
            "Q-Q Plot (Raw)",
            "Mean-Variance (Raw)",
        ],
        horizontal_spacing=0.08,
    )
    
    # --- Col 1: Raw distribution + mean + ±2σ ---
    raw_vals = df_raw[raw_cols].to_numpy().ravel()
    raw_vals = raw_vals[np.isfinite(raw_vals)]
    
    if len(raw_vals) > 0:
        mu = float(np.mean(raw_vals))
        sigma = float(np.std(raw_vals))
        x0, x1 = mu - 3 * sigma, mu + 3 * sigma
        
        # Histogram
        hist, bin_edges = np.histogram(raw_vals, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        
        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=hist,
                width=bin_width * 0.9,
                marker=dict(color="#1f77b4", opacity=0.7),
                showlegend=False,
            ),
            row=1, col=1,
        )
        
        # Shaded ±2σ region
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor="#1f77b4",
            opacity=0.15,
            line_width=0,
            row=1, col=1,
        )
        
        # Mean line
        fig.add_vline(
            x=mu,
            line_color="red",
            line_width=2,
            line_dash="dash",
            row=1, col=1,
        )
        
        # Annotations
        y_max = max(hist) if len(hist) > 0 else 1
        fig.add_annotation(
            x=mu, y=y_max * 1.05,
            xref="x1", yref="y1",
            text=f"μ={mu:.2f}",
            showarrow=False,
            font=dict(color="red", size=10),
        )
        
        fig.update_xaxes(title_text="Intensity", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
    
    # --- Col 2: Q-Q plot for raw data ---
    if len(raw_vals) >= 10:
        from scipy.stats import probplot
        osm_raw, osr_raw = probplot(raw_vals, dist="norm")[:2]
        theo_q_raw = osm_raw[0]
        ordered_raw = osm_raw[1]
        
        fig.add_trace(
            go.Scatter(
                x=theo_q_raw,
                y=ordered_raw,
                mode="markers",
                marker=dict(color="#1f77b4", size=3),
                showlegend=False,
            ),
            row=1, col=2,
        )
        
        # Reference line
        min_q, max_q = theo_q_raw.min(), theo_q_raw.max()
        fig.add_trace(
            go.Scatter(
                x=[min_q, max_q],
                y=[min_q, max_q],
                mode="lines",
                line=dict(color="red", dash="dash"),
                showlegend=False,
            ),
            row=1, col=2,
        )
        
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
    
    # --- Col 3: Mean-variance relationship ---
    means_raw = df_raw[raw_cols].mean(axis=1)
    vars_raw = df_raw[raw_cols].var(axis=1)
    
    fig.add_trace(
        go.Scatter(
            x=means_raw,
            y=vars_raw,
            mode="markers",
            marker=dict(color="#1f77b4", size=4, opacity=0.4),
            showlegend=False,
        ),
        row=1, col=3,
    )
    
    fig.update_xaxes(title_text="Mean", row=1, col=3)
    fig.update_yaxes(title_text="Variance", row=1, col=3)
    
    fig.update_layout(
        height=350,
        title=title,
        font=dict(family=FONT_FAMILY, size=11),
    )
    
    return fig

def create_transformed_row_figure(
    df_transformed: pd.DataFrame,
    trans_cols: list,
    title: str,
) -> go.Figure:
    """
    Single row (1×3) diagnostic panel for transformed data.
    Same layout as create_raw_row_figure but for transformed values.
    
    Args:
        df_transformed: Transformed data
        trans_cols: Column names to analyze
        title: Plot title (e.g., "Log2 Transform")
    
    Returns:
        Plotly figure with 3 subplots
    """
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            f"{title} Intensities",
            "Q-Q Plot (Transformed)",
            "Mean-Variance (Transformed)",
        ],
        horizontal_spacing=0.08,
    )
    
    # --- Col 1: Transformed distribution ---
    trans_vals = df_transformed[trans_cols].to_numpy().ravel()
    trans_vals = trans_vals[np.isfinite(trans_vals)]
    
    if len(trans_vals) > 0:
        mu = float(np.mean(trans_vals))
        sigma = float(np.std(trans_vals))
        x0, x1 = mu - 3 * sigma, mu + 3 * sigma
        
        hist, bin_edges = np.histogram(trans_vals, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        
        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=hist,
                width=bin_width * 0.9,
                marker=dict(color="#ff7f0e", opacity=0.7),
                showlegend=False,
            ),
            row=1, col=1,
        )
        
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor="#ff7f0e",
            opacity=0.15,
            line_width=0,
            row=1, col=1,
        )
        
        fig.add_vline(
            x=mu,
            line_color="darkred",
            line_width=2,
            line_dash="dash",
            row=1, col=1,
        )
        
        y_max = max(hist) if len(hist) > 0 else 1
        fig.add_annotation(
            x=mu, y=y_max * 1.05,
            xref="x1", yref="y1",
            text=f"μ={mu:.2f}",
            showarrow=False,
            font=dict(color="darkred", size=10),
        )
        
        fig.update_xaxes(title_text="Transformed Intensity", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
    
    # --- Col 2: Q-Q plot ---
    if len(trans_vals) >= 10:
        from scipy.stats import probplot
        osm_t, osr_t = probplot(trans_vals, dist="norm")[:2]
        theo_q_t = osm_t[0]
        ordered_t = osm_t[1]
        
        fig.add_trace(
            go.Scatter(
                x=theo_q_t,
                y=ordered_t,
                mode="markers",
                marker=dict(color="#ff7f0e", size=3),
                showlegend=False,
            ),
            row=1, col=2,
        )
        
        min_qt, max_qt = theo_q_t.min(), theo_q_t.max()
        fig.add_trace(
            go.Scatter(
                x=[min_qt, max_qt],
                y=[min_qt, max_qt],
                mode="lines",
                line=dict(color="red", dash="dash"),
                showlegend=False,
            ),
            row=1, col=2,
        )
        
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
    
    # --- Col 3: Mean-variance ---
    means_trans = df_transformed[trans_cols].mean(axis=1)
    vars_trans = df_transformed[trans_cols].var(axis=1)
    
    fig.add_trace(
        go.Scatter(
            x=means_trans,
            y=vars_trans,
            mode="markers",
            marker=dict(color="#ffb74d", size=4, opacity=0.4),
            showlegend=False,
        ),
        row=1, col=3,
    )
    
    fig.update_xaxes(title_text="Mean", row=1, col=3)
    fig.update_yaxes(title_text="Variance", row=1, col=3)
    
    fig.update_layout(
        height=350,
        title=f"Transformation: {title}",
        font=dict(family=FONT_FAMILY, size=11),
    )
    
    return fig

# ============================================================================
# VOLCANO PLOT
# log2FC vs -log10(p-value) for differential expression
# ============================================================================

@st.cache_data(ttl=1800)
def create_volcano_plot(
    log2fc: pd.Series,
    neg_log10_pval: pd.Series,
    regulation: pd.Series,
    fc_threshold: float = 1.0,
    pval_threshold: float = 0.05,
    theme_name: str = "light",
) -> go.Figure:
    """
    Create volcano plot for differential expression visualization.
    X-axis: log2 fold change
    Y-axis: -log10(p-value)
    
    Args:
        log2fc: log2 fold changes
        neg_log10_pval: -log10(p-values)
        regulation: Regulation classification (up/down/ns/nt)
        fc_threshold: Vertical threshold lines
        pval_threshold: Horizontal threshold line
        theme_name: Theme for colors
    
    Returns:
        Plotly scatter plot
    """
    theme = get_theme(theme_name)
    
    # Create figure
    fig = go.Figure()
    
    # Color mapping
    color_map = {
        "up": theme["color_up"],
        "down": theme["color_down"],
        "not_significant": theme["color_ns"],
        "not_tested": theme["color_nt"],
    }
    
    # Plot by category for proper legend
    for cat in ["not_tested", "not_significant", "down", "up"]:
        mask = regulation == cat
        
        if mask.sum() == 0:
            continue
        
        opacity = 0.7 if cat in ["up", "down"] else 0.3
        size = 6 if cat in ["up", "down"] else 4
        
        fig.add_trace(go.Scatter(
            x=log2fc[mask],
            y=neg_log10_pval[mask],
            mode="markers",
            name=cat.replace("_", " ").title(),
            marker=dict(
                color=color_map[cat],
                size=size,
                opacity=opacity,
                line=dict
