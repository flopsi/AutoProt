"""
helpers/plots_advanced.py
Advanced visualization functions (heatmaps, PCA, QC dashboard)
Extends Phase 1 plots with sophisticated analysis visualizations
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from helpers.constants import get_theme, FONT_FAMILY

# ============================================================================
# HEATMAP PLOTS
# ============================================================================

def create_heatmap_clustered(
    df: pd.DataFrame,
    numeric_cols: list,
    species_mapping: dict = None,
    theme_name: str = "light",
) -> go.Figure:
    """
    Create hierarchical clustered heatmap.
    Rows = proteins, Columns = samples, colored by intensity.
    
    Args:
        df: Data matrix (proteins × samples)
        numeric_cols: Column names to include
        species_mapping: Optional protein → species mapping
        theme_name: Theme for styling
        
    Returns:
        Plotly figure with heatmap + dendrograms
    """
    theme = get_theme(theme_name)
    
    # Extract data
    data = df[numeric_cols].fillna(0).astype(float)
    
    # Row clustering
    row_linkage = linkage(data, method="ward")
    row_dendro = dendrogram(row_linkage, no_plot=True)
    row_order = row_dendro["leaves"]
    
    # Column clustering
    col_linkage = linkage(data.T, method="ward")
    col_dendro = dendrogram(col_linkage, no_plot=True)
    col_order = col_dendro["leaves"]
    
    # Reorder data
    clustered_data = data.iloc[row_order, col_order]
    
    # Create heatmap
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.15, 0.85],
        specs=[[{"type": "scatter"}, {"type": "heatmap"}]],
        horizontal_spacing=0.05,
    )
    
    # Dendrogram (left)
    dendro_x = row_dendro["icoord"]
    dendro_y = row_dendro["dcoord"]
    
    for x, y in zip(dendro_x, dendro_y):
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="lines", line=dict(color=theme["accent"], width=1),
                       showlegend=False, hoverinfo="skip"),
            row=1, col=1
        )
    
    # Heatmap (right)
    fig.add_trace(
        go.Heatmap(
            z=clustered_data.values,
            x=clustered_data.columns,
            y=clustered_data.index,
            colorscale="Viridis",
            name="",
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Clustered Heatmap",
        height=800,
        showlegend=False,
    )
    
    return fig


# ============================================================================
# PCA PLOTS
# ============================================================================

def create_pca_plot(
    df: pd.DataFrame,
    numeric_cols: list,
    group_mapping: dict = None,
    theme_name: str = "light",
    dim: int = 2,
) -> go.Figure:
    """
    Create PCA plot with group coloring.
    
    Args:
        df: Data matrix
        numeric_cols: Columns for analysis
        group_mapping: Column → Group mapping
        theme_name: Theme
        dim: 2 or 3 for dimensionality
        
    Returns:
        Plotly 2D or 3D scatter plot
    """
    theme = get_theme(theme_name)
    
    # Prepare data
    data = df[numeric_cols].fillna(data[numeric_cols].mean())
    
    # Run PCA
    pca = PCA(n_components=min(3, len(numeric_cols)))
    transformed = pca.fit_transform(data.T)
    
    # Color mapping
    colors = []
    if group_mapping:
        species_colors = {
            "HUMAN": theme["color_human"],
            "YEAST": theme["color_yeast"],
            "ECOLI": theme["color_ecoli"],
        }
        colors = [species_colors.get(group_mapping.get(col, "UNKNOWN"), theme["accent"]) 
                  for col in numeric_cols]
    else:
        colors = [theme["accent"]] * len(numeric_cols)
    
    # Create plot
    if dim == 3 and transformed.shape[1] >= 3:
        fig = go.Figure(data=[go.Scatter3d(
            x=transformed[:, 0],
            y=transformed[:, 1],
            z=transformed[:, 2],
            mode="markers",
            marker=dict(size=8, color=colors, opacity=0.8),
            text=numeric_cols,
            hoverinfo="text",
        )])
    else:
        fig = go.Figure(data=[go.Scatter(
            x=transformed[:, 0],
            y=transformed[:, 1],
            mode="markers+text",
            marker=dict(size=10, color=colors, opacity=0.8),
            text=numeric_cols,
            textposition="top center",
            hoverinfo="text",
        )])
    
    fig.update_layout(
        title=f"PCA (Var explained: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%})",
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
        height=600,
        plot_bgcolor=theme["bg_secondary"],
    )
    
    return fig


# ============================================================================
# MA PLOT
# ============================================================================

def create_ma_plot(
    log2fc: pd.Series,
    mean_intensity: pd.Series,
    fc_threshold: float = 1.0,
    theme_name: str = "light",
) -> go.Figure:
    """
    Create M-A plot: log2FC vs average intensity.
    Useful for identifying expression-dependent bias.
    
    Args:
        log2fc: log2 fold changes
        mean_intensity: Mean intensity for each protein
        fc_threshold: FC threshold for highlighting
        theme_name: Theme
        
    Returns:
        Plotly figure
    """
    theme = get_theme(theme_name)
    
    # Classify proteins
    classification = []
    colors = []
    for fc in log2fc:
        if pd.isna(fc):
            classification.append("not_tested")
            colors.append(theme["color_nt"])
        elif abs(fc) > fc_threshold:
            if fc > 0:
                classification.append("up")
                colors.append(theme["color_up"])
            else:
                classification.append("down")
                colors.append(theme["color_down"])
        else:
            classification.append("not_significant")
            colors.append(theme["color_ns"])
    
    fig = go.Figure()
    
    # Add points by category
    for cat in ["not_tested", "not_significant", "down", "up"]:
        mask = [c == cat for c in classification]
        indices = [i for i, m in enumerate(mask) if m]
        
        if indices:
            opacity = 0.7 if cat in ["up", "down"] else 0.3
            fig.add_trace(go.Scatter(
                x=mean_intensity.iloc[indices],
                y=log2fc.iloc[indices],
                mode="markers",
                name=cat.replace("_", " ").title(),
                marker=dict(
                    color=[colors[i] for i in indices],
                    size=5,
                    opacity=opacity,
                ),
            ))
    
    # Add threshold lines
    fig.add_hline(y=fc_threshold, line_dash="dash", line_color=theme["text_secondary"], opacity=0.5)
    fig.add_hline(y=-fc_threshold, line_dash="dash", line_color=theme["text_secondary"], opacity=0.5)
    fig.add_hline(y=0, line_dash="solid", line_color=theme["text_primary"], width=1)
    
    fig.update_xaxes(type="log", title="Average Intensity")
    fig.update_yaxes(title="log2 Fold Change")
    fig.update_layout(
        title="M-A Plot",
        height=500,
        plot_bgcolor=theme["bg_secondary"],
        hovermode="closest",
    )
    
    return fig


# ============================================================================
# Q-Q PLOT
# ============================================================================

def create_qq_plot(
    residuals: np.ndarray,
    theme_name: str = "light",
) -> go.Figure:
    """
    Q-Q plot for normality assessment.
    
    Args:
        residuals: Residual values
        theme_name: Theme
        
    Returns:
        Plotly figure
    """
    from scipy.stats import norm
    
    theme = get_theme(theme_name)
    
    # Clean data
    residuals_clean = residuals[~np.isnan(residuals)]
    
    # Standardize
    z_scores = (residuals_clean - residuals_clean.mean()) / residuals_clean.std()
    z_scores_sorted = np.sort(z_scores)
    
    # Theoretical quantiles
    q = np.arange(1, len(z_scores_sorted) + 1) / (len(z_scores_sorted) + 1)
    theoretical_q = norm.ppf(q)
    
    fig = go.Figure()
    
    # Data points
    fig.add_trace(go.Scatter(
        x=theoretical_q,
        y=z_scores_sorted,
        mode="markers",
        marker=dict(color=theme["accent"], size=6),
        name="Data",
    ))
    
    # Diagonal line
    min_q, max_q = theoretical_q.min(), theoretical_q.max()
    fig.add_trace(go.Scatter(
        x=[min_q, max_q],
        y=[min_q, max_q],
        mode="lines",
        line=dict(color=theme["color_ns"], dash="dash"),
        name="Normal",
    ))
    
    fig.update_layout(
        title="Q-Q Plot (Normality Assessment)",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        height=500,
        plot_bgcolor=theme["bg_secondary"],
        showlegend=True,
    )
    
    return fig


# ============================================================================
# QC DASHBOARD
# ============================================================================

def create_qc_dashboard(
    protein_data,
    results_df: pd.DataFrame,
    theme_name: str = "light",
) -> go.Figure:
    """
    Multi-panel QC dashboard showing:
    - Missing rate per sample
    - Distribution of intensities
    - CV distribution
    - Number of significant proteins
    
    Args:
        protein_data: ProteinData object
        results_df: Analysis results
        theme_name: Theme
        
    Returns:
        Plotly figure with 4 subplots
    """
    theme = get_theme(theme_name)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Missing Rate", "Intensity Distribution", "CV Distribution", "Significance"),
        specs=[[{"type": "bar"}, {"type": "box"}],
               [{"type": "histogram"}, {"type": "pie"}]],
    )
    
    # Panel 1: Missing rate per sample
    missing_rates = protein_data.raw[protein_data.numeric_cols].isna().mean() * 100
    
    fig.add_trace(
        go.Bar(x=protein_data.numeric_cols, y=missing_rates, 
               marker_color=theme["accent"], name="Missing %",
               showlegend=False),
        row=1, col=1
    )
    
    # Panel 2: Intensity distribution
    for col in protein_data.numeric_cols[:min(5, len(protein_data.numeric_cols))]:
        fig.add_trace(
            go.Box(y=protein_data.raw[col].dropna(), name=col, showlegend=False),
            row=1, col=2
        )
    
    # Panel 3: CV distribution (if available)
    cv_vals = []
    for i, protein in enumerate(protein_data.raw.index):
        row = protein_data.raw.iloc[i]
        vals = row[protein_data.numeric_cols].dropna()
        if len(vals) > 1:
            cv = vals.std() / vals.mean() * 100 if vals.mean() > 0 else np.nan
            cv_vals.append(cv)
    
    if cv_vals:
        fig.add_trace(
            go.Histogram(x=cv_vals, nbinsx=30, marker_color=theme["accent"],
                         name="CV %", showlegend=False),
            row=2, col=1
        )
    
    # Panel 4: Significance pie chart
    if "regulation" in results_df.columns:
        sig_counts = results_df["regulation"].value_counts()
        
        fig.add_trace(
            go.Pie(labels=sig_counts.index, values=sig_counts.values,
                   marker=dict(colors=[theme["color_up"], theme["color_down"], 
                                       theme["color_ns"], theme["color_nt"]]),
                   name="", showlegend=True),
            row=2, col=2
        )
    
    fig.update_yaxes(title_text="Missing %", row=1, col=1)
    fig.update_yaxes(title_text="log10 Intensity", row=1, col=2)
    fig.update_xaxes(title_text="CV %", row=2, col=1)
    
    fig.update_layout(
        title="Quality Control Dashboard",
        height=800,
        showlegend=True,
    )
    
    return fig
