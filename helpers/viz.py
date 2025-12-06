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
                line=dict(width=0)
            ),
            hovertemplate="log2FC: %{x:.2f}<br>-log10(p): %{y:.2f}<extra></extra>"
        ))
    
    # Add threshold lines
    fig.add_vline(
        x=fc_threshold,
        line_dash="dash",
        line_color=theme["text_secondary"],
        opacity=0.5,
        annotation_text=f"FC={fc_threshold}"
    )
    fig.add_vline(
        x=-fc_threshold,
        line_dash="dash",
        line_color=theme["text_secondary"],
        opacity=0.5,
        annotation_text=f"FC={-fc_threshold}"
    )
    fig.add_hline(
        y=-np.log10(pval_threshold),
        line_dash="dash",
        line_color=theme["text_secondary"],
        opacity=0.5,
        annotation_text=f"p={pval_threshold}"
    )
    
    fig.update_layout(
        title="Volcano Plot",
        xaxis_title="log2 Fold Change",
        yaxis_title="-log10(p-value)",
        height=600,
        plot_bgcolor=theme["bg_secondary"],
        paper_bgcolor=theme["paper_bg"],
        font=dict(family=FONT_FAMILY, color=theme["text_primary"]),
        hovermode="closest",
    )
    
    return fig

# ============================================================================
# PCA PLOT
# Principal component analysis for sample clustering
# ============================================================================

@st.cache_data(ttl=1800)
def create_pca_plot(
    df: pd.DataFrame,
    numeric_cols: list,
    group_mapping: dict = None,
    theme_name: str = "light",
    dim: int = 2,
) -> go.Figure:
    """
    Create PCA plot with optional group coloring.
    
    Args:
        df: Data matrix (proteins × samples)
        numeric_cols: Columns to include in PCA
        group_mapping: Optional dict mapping column → group name
        theme_name: Theme for colors
        dim: Dimensionality (2 or 3)
    
    Returns:
        Plotly 2D or 3D scatter plot
    """
    theme = get_theme(theme_name)
    
    # Prepare data (samples as rows)
    data = df[numeric_cols].fillna(df[numeric_cols].mean())
    
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
        colors = [
            species_colors.get(group_mapping.get(col, "UNKNOWN"), theme["accent"])
            for col in numeric_cols
        ]
    else:
        colors = [theme["accent"]] * len(numeric_cols)
    
    # Create plot
    if dim == 3 and transformed.shape[1] >= 3:
        fig = go.Figure(data=[go.Scatter3d(
            x=transformed[:, 0],
            y=transformed[:, 1],
            z=transformed[:, 2],
            mode="markers+text",
            marker=dict(size=8, color=colors, opacity=0.8),
            text=numeric_cols,
            textposition="top center",
            hoverinfo="text",
        )])
        title = f"PCA 3D (PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}, PC3={pca.explained_variance_ratio_[2]:.1%})"
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
        title = f"PCA (PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%})"
    
    fig.update_layout(
        title=title,
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
        height=600,
        plot_bgcolor=theme["bg_secondary"],
        font=dict(family=FONT_FAMILY),
    )
    
    return fig

# ============================================================================
# HEATMAP
# Hierarchical clustered heatmap
# ============================================================================

@st.cache_data(ttl=1800)
def create_heatmap_clustered(
    df: pd.DataFrame,
    numeric_cols: list,
    theme_name: str = "light",
) -> go.Figure:
    """
    Create hierarchical clustered heatmap.
    Rows = proteins, Columns = samples, colored by intensity.
    
    Args:
        df: Data matrix (proteins × samples)
        numeric_cols: Column names to include
        theme_name: Theme for styling
    
    Returns:
        Plotly figure with heatmap + row dendrogram
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
    fig = go.Figure(data=go.Heatmap(
        z=clustered_data.values,
        x=clustered_data.columns,
        y=clustered_data.index,
        colorscale="Viridis",
        colorbar=dict(title="Intensity"),
    ))
    
    fig.update_layout(
        title="Clustered Heatmap",
        height=800,
        xaxis_title="Samples",
        yaxis_title="Proteins",
        plot_bgcolor=theme["bg_secondary"],
    )
    
    return fig

# ============================================================================
# QC DASHBOARD
# Multi-panel quality control overview
# ============================================================================

@st.cache_data(ttl=1800)
def create_qc_dashboard(
    df: pd.DataFrame,
    numeric_cols: list,
    results_df: pd.DataFrame = None,
    theme_name: str = "light",
) -> go.Figure:
    """
    Multi-panel QC dashboard showing:
    - Missing rate per sample
    - Intensity distribution (box plots)
    - CV distribution
    - Significance breakdown (if results provided)
    
    Args:
        df: Raw data matrix
        numeric_cols: Columns to analyze
        results_df: Optional analysis results with 'regulation' column
        theme_name: Theme
    
    Returns:
        Plotly figure with 2×2 subplots
    """
    theme = get_theme(theme_name)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Missing Rate", "Intensity Distribution", "CV Distribution", "Significance"),
        specs=[[{"type": "bar"}, {"type": "box"}],
               [{"type": "histogram"}, {"type": "pie"}]],
    )
    
    # Panel 1: Missing rate per sample
    missing_rates = df[numeric_cols].isna().mean() * 100
    fig.add_trace(
        go.Bar(
            x=numeric_cols,
            y=missing_rates,
            marker_color=theme["accent"],
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Panel 2: Intensity distribution (first 5 samples)
    for col in numeric_cols[:min(5, len(numeric_cols))]:
        fig.add_trace(
            go.Box(y=df[col].dropna(), name=col, showlegend=False),
            row=1, col=2
        )
    
    # Panel 3: CV distribution
    cv_vals = []
    for _, row in df.iterrows():
        vals = row[numeric_cols].dropna()
        if len(vals) > 1:
            cv = vals.std() / vals.mean() * 100 if vals.mean() > 0 else np.nan
            if np.isfinite(cv):
                cv_vals.append(cv)
    
    if cv_vals:
        fig.add_trace(
            go.Histogram(x=cv_vals, nbinsx=30, marker_color=theme["accent"], showlegend=False),
            row=2, col=1
        )
    
    # Panel 4: Significance pie chart
    if results_df is not None and "regulation" in results_df.columns:
        sig_counts = results_df["regulation"].value_counts()
        fig.add_trace(
            go.Pie(
                labels=sig_counts.index,
                values=sig_counts.values,
                marker=dict(colors=[
                    theme["color_up"],
                    theme["color_down"],
                    theme["color_ns"],
                    theme["color_nt"]
                ][:len(sig_counts)]),
                showlegend=True
            ),
            row=2, col=2
        )
    
    fig.update_yaxes(title_text="Missing %", row=1, col=1)
    fig.update_yaxes(title_text="Intensity", row=1, col=2)
    fig.update_xaxes(title_text="CV %", row=2, col=1)
    
    fig.update_layout(
        title="Quality Control Dashboard",
        height=800,
        showlegend=True,
        plot_bgcolor=theme["bg_secondary"],
        font=dict(family=FONT_FAMILY),
    )
    
    return fig

# ============================================================================
# EDA-SPECIFIC PLOTS
# Protein counts and condition-based boxplots
# ============================================================================

@st.cache_data(ttl=3600, show_spinner="Computing protein counts...")
def create_protein_count_stacked_bar(
    df_log2: pd.DataFrame,
    numeric_cols: list,
    species_mapping: dict,
    theme: dict
) -> tuple[go.Figure, pd.DataFrame]:
    """
    Create stacked bar chart of protein counts per sample, grouped by species.
    
    Args:
        df_log2: Log2-transformed data
        numeric_cols: Sample column names
        species_mapping: Dict mapping protein ID → species
        theme: Theme colors dictionary
    
    Returns:
        (figure, summary_dataframe) tuple
    """
    # Count proteins per sample per species
    protein_counts = {}
    
    for sample in numeric_cols:
        valid_proteins = df_log2[df_log2[sample].notna() & (df_log2[sample] != 0.0)].index
        species_counts = {}
        for protein_id in valid_proteins:
            species = species_mapping.get(protein_id, "UNKNOWN")
            if species:
                species_counts[species] = species_counts.get(species, 0) + 1
        protein_counts[sample] = species_counts
    
    # Get all species
    all_species = sorted(set(sp for counts in protein_counts.values() for sp in counts.keys()))
    
    # Create stacked bar chart
    fig = go.Figure()
    
    colors = {
        'HUMAN': theme['color_human'],
        'YEAST': theme['color_yeast'],
        'ECOLI': theme['color_ecoli'],
        'MOUSE': '#9467bd',
        'UNKNOWN': '#999999'
    }
    
    for species in all_species:
        counts = [protein_counts[sample].get(species, 0) for sample in numeric_cols]
        total_count = sum(counts)
        
        fig.add_trace(go.Bar(
            name=f"{species} (n={total_count})",
            x=numeric_cols,
            y=counts,
            marker_color=colors.get(species, '#cccccc'),
            hovertemplate=f"<b>{species}</b><br>Sample: %{{x}}<br>Proteins: %{{y}}<extra></extra>"
        ))
    
    fig.update_layout(
        barmode='stack',
        title="Protein Counts per Sample",
        xaxis_title="Sample",
        yaxis_title="Number of Proteins",
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['paper_bg'],
        font=dict(family=FONT_FAMILY, size=14, color=theme['text_primary']),
        showlegend=True,
        legend=dict(title="Species", orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        height=500,
        hovermode='x unified'
    )
    
    fig.update_xaxes(showgrid=True, gridcolor=theme['grid'], tickangle=-45)
    fig.update_yaxes(showgrid=True, gridcolor=theme['grid'])
    
    # Summary dataframe
    summary_data = []
    for species in all_species:
        total = sum(protein_counts[sample].get(species, 0) for sample in numeric_cols)
        avg = total / len(numeric_cols)
        min_count = min(protein_counts[sample].get(species, 0) for sample in numeric_cols)
        max_count = max(protein_counts[sample].get(species, 0) for sample in numeric_cols)
        summary_data.append({
            'Species': species,
            'Total': total,
            'Avg/Sample': f"{avg:.1f}",
            'Min': min_count,
            'Max': max_count
        })
    
    return fig, pd.DataFrame(summary_data)

@st.cache_data(ttl=3600, show_spinner="Creating box plots...")
def create_boxplot_by_condition(
    df_log2: pd.DataFrame,
    condition_dict: dict,
    conditions_to_plot: list,
    theme: dict
) -> tuple[go.Figure, pd.DataFrame]:
    """
    Create boxplots of log2 intensities grouped by experimental condition.
    
    Args:
        df_log2: Log2-transformed data
        condition_dict: Dict mapping condition → list of samples
        conditions_to_plot: List of conditions to include (typically ["A", "B"])
        theme: Theme colors dictionary
    
    Returns:
        (figure, summary_stats_dataframe) tuple
    """
    fig = go.Figure()
    
    condition_colors = {
        conditions_to_plot[0]: theme['color_human'],
        conditions_to_plot[1]: theme['color_yeast'] if len(conditions_to_plot) > 1 else theme['color_ecoli']
    }
    
    for cond in conditions_to_plot:
        samples = condition_dict[cond]
        
        for sample in samples:
            values = df_log2[sample].dropna()
            values = values[values != 0.0]
            
            fig.add_trace(go.Box(
                y=values,
                name=sample,
                marker_color=condition_colors[cond],
                legendgroup=cond,
                legendgrouptitle_text=f"Condition {cond}",
                boxmean='sd',
                hovertemplate=f"<b>{sample}</b><br>Intensity: %{{y:.2f}}<extra></extra>"
            ))
    
    fig.update_layout(
        title="Log2 Intensity Distribution by Condition",
        xaxis_title="Sample",
        yaxis_title="Log2 Intensity",
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['paper_bg'],
        font=dict(family=FONT_FAMILY, size=14, color=theme['text_primary']),
        showlegend=True,
        height=600
    )
    
    fig.update_xaxes(showgrid=True, gridcolor=theme['grid'], tickangle=-45)
    fig.update_yaxes(showgrid=True, gridcolor=theme['grid'])
    
    # Summary statistics
    summary_stats = []
    for cond in conditions_to_plot:
        samples = condition_dict[cond]
        vals = np.concatenate([
            df_log2[s].dropna().values[df_log2[s].dropna() != 0.0]
            for s in samples
        ])
        
        summary_stats.append({
            'Condition': cond,
            'N Samples': len(samples),
            'Mean': f"{np.mean(vals):.2f}",
            'Median': f"{np.median(vals):.2f}",
            'Std Dev': f"{np.std(vals):.2f}",
            'Min': f"{np.min(vals):.2f}",
            'Max': f"{np.max(vals):.2f}"
        })
    
    return fig, pd.DataFrame(summary_stats)
