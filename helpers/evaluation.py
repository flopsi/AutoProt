"""
helpers/evaluation.py
Transformation evaluation with diagnostic plots (from working Colab code)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from typing import Dict, Tuple

def evaluate_transformation(
    df_raw: pd.DataFrame,
    df_transformed: pd.DataFrame,
    raw_cols: list,
    trans_cols: list,
    transform_name: str
) -> Dict[str, float]:
    """
    Evaluate transformation quality with diagnostic plots and statistics.
    
    Parameters:
    -----------
    df_raw : pd.DataFrame
        Original (raw) data
    df_transformed : pd.DataFrame  
        Transformed data
    raw_cols : list
        Raw intensity column names
    trans_cols : list
        Transformed column names (e.g., col_transformed)
    transform_name : str
        Name of transformation applied
    
    Returns:
    --------
    dict with diagnostic metrics
    """
    # Get flattened data (remove NaN)
    raw_data = df_raw[raw_cols].values.flatten()
    raw_data = raw_data[~np.isnan(raw_data)]
    
    trans_data = df_transformed[trans_cols].values.flatten()
    trans_data = trans_data[~np.isnan(trans_data)]
    
    # Limit for Shapiro-Wilk (too sensitive with large N)
    sample_size = min(5000, len(raw_data))
    
    # Shapiro-Wilk tests
    _, p_shapiro_raw = stats.shapiro(raw_data[:sample_size])
    _, p_shapiro_trans = stats.shapiro(trans_data[:sample_size])
    
    # Mean-variance relationships
    means_raw = df_raw[raw_cols].mean(axis=1)
    vars_raw = df_raw[raw_cols].var(axis=1)
    
    means_trans = df_transformed[trans_cols].mean(axis=1)
    vars_trans = df_transformed[trans_cols].var(axis=1)
    
    # Correlations (lower is better for variance stabilization)
    corr_raw = np.corrcoef(means_raw, vars_raw)[0, 1]
    corr_trans = np.corrcoef(means_trans, vars_trans)[0, 1]
    
    metrics = {
        'shapiro_raw': p_shapiro_raw,
        'shapiro_transformed': p_shapiro_trans,
        'mean_var_corr_raw': corr_raw,
        'mean_var_corr_trans': corr_trans,
        'n_raw': len(raw_data),
        'n_trans': len(trans_data)
    }
    
    return metrics


def create_evaluation_plots(
    df_raw: pd.DataFrame,
    df_transformed: pd.DataFrame,
    raw_cols: list,
    trans_cols: list,
    transform_name: str,
    theme: dict
) -> go.Figure:
    """
    Create 2x3 diagnostic plots for transformation evaluation.
    """
    # Create subplot figure (2x3 grid)
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            'Raw Distribution', 
            f'{transform_name} Distribution',
            'Q-Q Plot (Raw)', 
            'Q-Q Plot (Transformed)',
            'Mean-Variance (Raw)', 
            'Mean-Variance (Transformed)'
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Raw distribution histogram
    raw_data = df_raw[raw_cols].values.flatten()
    raw_data = raw_data[~np.isnan(raw_data)]
    raw_data = raw_data[raw_data > 0]
    
    if len(raw_data) > 0:
        fig.add_trace(
            go.Histogram(x=raw_data, nbinsx=50, opacity=0.7, 
                        marker_color=theme.get('primary', '#1f77b4'),
                        name='Raw', showlegend=False),
            row=1, col=1
        )
    
    # 2. Transformed distribution histogram  
    trans_data = df_transformed[trans_cols].values.flatten()
    trans_data = trans_data[~np.isnan(trans_data)]
    
    if len(trans_data) > 0:
        fig.add_trace(
            go.Histogram(x=trans_data, nbinsx=50, opacity=0.7,
                        marker_color=theme.get('secondary', '#ff7f0e'),
                        name=f'{transform_name}', showlegend=False),
            row=1, col=2
        )
    
    # 3. Q-Q Plot Raw
    if len(raw_data) > 100:
        qq_raw = stats.probplot(raw_data[:5000], dist="norm")
        fig.add_trace(
            go.Scatter(x=qq_raw[0][0], y=qq_raw[0][1], mode='markers',
                      marker=dict(color=theme.get('primary', '#1f77b4'), size=3),
                      name='Raw QQ', showlegend=False),
            row=1, col=3
        )
        fig.add_trace(
            go.Scatter(x=[raw_data.min(), raw_data.max()], 
                      y=[raw_data.min(), raw_data.max()], 
                      mode='lines',
                      line=dict(color='red', dash='dash'),
                      name='Perfect fit', showlegend=False),
            row=1, col=3
        )
    
    # 4. Q-Q Plot Transformed
    if len(trans_data) > 100:
        qq_trans = stats.probplot(trans_data[:5000], dist="norm")
        fig.add_trace(
            go.Scatter(x=qq_trans[0][0], y=qq_trans[0][1], mode='markers',
                      marker=dict(color=theme.get('secondary', '#ff7f0e'), size=3),
                      name=f'{transform_name} QQ', showlegend=False),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=[trans_data.min(), trans_data.max()], 
                      y=[trans_data.min(), trans_data.max()], 
                      mode='lines',
                      line=dict(color='red', dash='dash'),
                      name='Perfect fit', showlegend=False),
            row=2, col=1
        )
    
    # 5. Mean-Variance Raw (log scale)
    means_raw = df_raw[raw_cols].mean(axis=1)
    vars_raw = df_raw[raw_cols].var(axis=1)
    valid_raw = ~(np.isnan(means_raw) | np.isnan(vars_raw))
    
    if valid_raw.sum() > 0:
        fig.add_trace(
            go.Scatter(x=means_raw[valid_raw], y=vars_raw[valid_raw], 
                      mode='markers', marker=dict(color=theme.get('primary', '#1f77b4'), 
                      size=4, opacity=0.6), showlegend=False),
            row=2, col=2
        )
    
    # 6. Mean-Variance Transformed
    means_trans = df_transformed[trans_cols].mean(axis=1)
    vars_trans = df_transformed[trans_cols].var(axis=1)
    valid_trans = ~(np.isnan(means_trans) | np.isnan(vars_trans))
    
    if valid_trans.sum() > 0:
        fig.add_trace(
            go.Scatter(x=means_trans[valid_trans], y=vars_trans[valid_trans],
                      mode='markers', marker=dict(color=theme.get('secondary', '#ff7f0e'),
                      size=4, opacity=0.6), showlegend=False),
            row=2, col=3
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text=f"<b>Transformation Evaluation: {transform_name}</b>",
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=12)
    )
    
    # Update axes
    fig.update_xaxes(type='log', title_text="Log(Intensity)", row=2, col=[2,3])
    fig.update_yaxes(type='log', title_text="Log(Variance)", row=2, col=[2,3])
    
    return fig
