"""
helpers/plots.py
All plotting functions with unified color scheme and layout
Theme parameter passed to every function for consistency
Supports: light, dark, colorblind, journal themes
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from helpers.constants import get_theme, FONT_FAMILY

# ============================================================================
# PLOT HELPER: Apply theme styling
# ============================================================================

def apply_theme(fig: go.Figure, theme: dict) -> go.Figure:
    """Apply theme colors and fonts to any figure."""
    fig.update_layout(
        plot_bgcolor=theme["bg_secondary"],
        paper_bgcolor=theme["paper_bg"],
        font=dict(family=FONT_FAMILY, color=theme["text_primary"], size=12),
        title_font_size=16,
    )
    return fig


# ============================================================================
# VOLCANO PLOT
# ============================================================================

def create_volcano_plot(
    results_df: pd.DataFrame,
    fc_threshold: float = 1.0,
    pval_threshold: float = 0.05,
    theme_name: str = "light",
) -> go.Figure:
    """
    Create volcano plot: log2FC vs -log10(p-value).
    
    Args:
        results_df: Results with 'log2fc', 'pvalue', 'regulation' columns
        fc_threshold: Fold change threshold for lines
        pval_threshold: P-value threshold for lines
        theme_name: Theme to apply
        
    Returns:
        Plotly figure
    """
    theme = get_theme(theme_name)
    results_df = results_df.copy()
    
    # Ensure regulation column exists
    if "regulation" not in results_df.columns:
        from helpers.statistics import classify_regulation
        results_df["regulation"] = results_df.apply(
            lambda row: classify_regulation(
                row.get("log2fc", np.nan),
                row.get("pvalue", np.nan),
                fc_threshold,
                pval_threshold,
            ),
            axis=1,
        )
    
    results_df["neg_log10_pval"] = -np.log10(
        results_df["pvalue"].replace(0, 1e-300)
    )
    
    # Color map by regulation
    color_map = {
        "up": theme["color_up"],
        "down": theme["color_down"],
        "not_significant": theme["color_ns"],
        "not_tested": theme["color_nt"],
    }
    
    fig = go.Figure()
    
    # ---- Add traces in order (NS first, so they're behind) ----
    for reg_type in ["not_tested", "not_significant", "down", "up"]:
        subset = results_df[results_df["regulation"] == reg_type]
        if len(subset) > 0:
            opacity = 0.7 if reg_type in ["up", "down"] else 0.3
            fig.add_trace(go.Scatter(
                x=subset["log2fc"],
                y=subset["neg_log10_pval"],
                mode="markers",
                name=reg_type.replace("_", " ").title(),
                marker=dict(
                    color=color_map[reg_type],
                    size=6,
                    opacity=opacity,
                ),
                text=subset.index,
                hovertemplate="<b>%{text}</b><br>log2FC: %{x:.2f}<br>-log10(p): %{y:.2f}<extra></extra>",
            ))
    
    # Add threshold lines
    fig.add_hline(
        y=-np.log10(pval_threshold),
        line_dash="dash",
        line_color=theme["text_secondary"],
        line_width=1,
        opacity=0.5,
    )
    fig.add_vline(x=fc_threshold, line_dash="dash", line_color=theme["text_secondary"], line_width=1, opacity=0.5)
    fig.add_vline(x=-fc_threshold, line_dash="dash", line_color=theme["text_secondary"], line_width=1, opacity=0.5)
    
    fig.update_layout(
        title=f"Volcano Plot (FC threshold: {fc_threshold:.1f}, p < {pval_threshold})",
        xaxis_title="log2 Fold Change",
        yaxis_title="-log10(p-value)",
        height=600,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    
    return apply_theme(fig, theme)


# ============================================================================
# SPECIES DISTRIBUTION PLOT
# ============================================================================

def create_species_distribution(
    results_df: pd.DataFrame,
    species_mapping: dict,
    fc_threshold: float = 1.0,
    pval_threshold: float = 0.05,
    theme_name: str = "light",
) -> go.Figure:
    """
    Create species-colored distribution with histogram, KDE, and boxplot.
    """
    theme = get_theme(theme_name)
    results_df = results_df.copy()
    
    # Ensure regulation column
    if "regulation" not in results_df.columns:
        from helpers.statistics import classify_regulation
        results_df["regulation"] = results_df.apply(
            lambda row: classify_regulation(
                row.get("log2fc", np.nan),
                row.get("pvalue", np.nan),
                fc_threshold,
                pval_threshold,
            ),
            axis=1,
        )
    
    # Group data by species
    species_data = {}
    species_colors = {
        "HUMAN": theme["color_human"],
        "YEAST": theme["color_yeast"],
        "ECOLI": theme["color_ecoli"],
    }
    
    # ✓ FIX: Normalize species names by removing leading underscores
    for species in ["HUMAN", "YEAST", "ECOLI"]:
        # Match both "HUMAN" and "_HUMAN"
        proteins = [
            pid for pid, sp in species_mapping.items() 
            if sp.lstrip('_').upper() == species  # Remove leading _ and compare
        ]
        if proteins:
            data = results_df.loc[results_df.index.intersection(proteins), "log2fc"].dropna()
            if len(data) > 0:
                species_data[species] = data.values

    
    if not species_data:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        return fig
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        specs=[[{"type": "histogram"}, {"type": "box"}]],
        horizontal_spacing=0.12,
    )
    
    # ---- Add traces for each species ----
    for i, species in enumerate(["HUMAN", "YEAST", "ECOLI"]):
        if species not in species_data:
            continue
        
        data = species_data[species]
        color = species_colors[species]
        
        # Histogram (left)
        fig.add_trace(
            go.Histogram(
                x=data,
                name=species,
                marker_color=color,
                opacity=0.6,
                nbinsx=40,
                legendgroup=species,
                showlegend=True,
            ),
            row=1, col=1
        )
        
        # KDE curve (left)
        hist, bin_edges = np.histogram(data, bins=60, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        smoothed = gaussian_filter1d(hist, sigma=2)
        
        fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=smoothed,
                name=f"{species} KDE",
                line=dict(color=color, width=3),
                legendgroup=species,
                showlegend=False,
            ),
            row=1, col=1
        )
        
        # Boxplot (right)
        fig.add_trace(
            go.Box(
                y=data,
                name=species,
                marker_color=color,
                legendgroup=species,
                showlegend=False,
                boxmean="sd",
                opacity=0.7,
            ),
            row=1, col=2
        )
    
    # Add threshold lines
    fig.add_vline(x=fc_threshold, line_dash="dash", line_color="red", line_width=1.5, opacity=0.5, row=1, col=1)
    fig.add_vline(x=-fc_threshold, line_dash="dash", line_color="red", line_width=1.5, opacity=0.5, row=1, col=1)
    fig.add_vline(x=0, line_dash="solid", line_color=theme["text_primary"], line_width=1, row=1, col=1)
    
    fig.add_hline(y=fc_threshold, line_dash="dash", line_color="red", line_width=1.5, opacity=0.5, row=1, col=2)
    fig.add_hline(y=-fc_threshold, line_dash="dash", line_color="red", line_width=1.5, opacity=0.5, row=1, col=2)
    fig.add_hline(y=0, line_dash="solid", line_color=theme["text_primary"], line_width=1, row=1, col=2)
    
    fig.update_xaxes(title_text="log2 Fold Change", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_yaxes(title_text="log2 Fold Change", row=1, col=2)
    
    fig.update_layout(
        title="Species Distribution of log2 Fold Changes",
        height=600,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        barmode="overlay",
    )
    
    return apply_theme(fig, theme)


# ============================================================================
# ROC CURVE
# ============================================================================

def create_roc_curve(
    fpr_list: list,
    tpr_list: list,
    theme_name: str = "light",
) -> go.Figure:
    """
    Create ROC curve plot.
    
    Args:
        fpr_list: False positive rates
        tpr_list: True positive rates
        theme_name: Theme
        
    Returns:
        Plotly figure
    """
    theme = get_theme(theme_name)
    
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr_list,
        y=tpr_list,
        mode="lines",
        name="ROC Curve",
        line=dict(color=theme["accent"], width=2),
        fill="tozeroy",
        fillcolor=f"rgba({int(theme['accent'][1:3], 16)}, {int(theme['accent'][3:5], 16)}, {int(theme['accent'][5:7], 16)}, 0.2)",
    ))
    
    # Diagonal (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random",
        line=dict(color=theme["color_ns"], width=1, dash="dash"),
        showlegend=True,
    ))
    
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    
    return apply_theme(fig, theme)


# ============================================================================
# PRECISION-RECALL CURVE
# ============================================================================

def create_precision_recall_curve(
    recall_list: list,
    precision_list: list,
    theme_name: str = "light",
) -> go.Figure:
    """
    Create precision-recall curve plot.
    
    Args:
        recall_list: Recall values
        precision_list: Precision values
        theme_name: Theme
        
    Returns:
        Plotly figure
    """
    theme = get_theme(theme_name)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recall_list,
        y=precision_list,
        mode="lines",
        name="PR Curve",
        line=dict(color=theme["accent"], width=2),
        fill="tozeroy",
        fillcolor=f"rgba({int(theme['accent'][1:3], 16)}, {int(theme['accent'][3:5], 16)}, {int(theme['accent'][5:7], 16)}, 0.2)",
    ))
    
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=500,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    
    return apply_theme(fig, theme)


# ============================================================================
# DENSITY PLOT
# ============================================================================

def create_density_plot(
    results_df: pd.DataFrame,
    fc_threshold: float = 1.0,
    theme_name: str = "light",
) -> go.Figure:
    """
    Create simple density plot of log2FC distribution.
    
    Args:
        results_df: Results with 'log2fc'
        fc_threshold: For threshold lines
        theme_name: Theme
        
    Returns:
        Plotly figure
    """
    theme = get_theme(theme_name)
    
    data = results_df["log2fc"].dropna().values
    
    if len(data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False)
        return fig
    
    # Compute KDE
    hist, bin_edges = np.histogram(data, bins=80, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    smoothed = gaussian_filter1d(hist, sigma=2)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=bin_centers,
        y=smoothed,
        fill="tozeroy",
        name="Density",
        line=dict(color=theme["accent"], width=2),
        fillcolor=f"rgba({int(theme['accent'][1:3], 16)}, {int(theme['accent'][3:5], 16)}, {int(theme['accent'][5:7], 16)}, 0.3)",
    ))
    
    # Threshold lines
    fig.add_vline(x=fc_threshold, line_dash="dash", line_color="red", line_width=1.5, opacity=0.5)
    fig.add_vline(x=-fc_threshold, line_dash="dash", line_color="red", line_width=1.5, opacity=0.5)
    fig.add_vline(x=0, line_dash="solid", line_color=theme["text_primary"], line_width=1)
    
    fig.update_layout(
        title="Distribution of log2 Fold Changes",
        xaxis_title="log2 Fold Change",
        yaxis_title="Density",
        height=400,
        showlegend=False,
    )
    
    return apply_theme(fig, theme)
