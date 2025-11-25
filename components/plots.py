"""
Visualization components for proteomics data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def create_volcano_plot(df: pd.DataFrame, p_value_threshold: float = 1.3, 
                       fc_threshold: float = 1.0):
    """
    Create an interactive volcano plot
    
    Args:
        df: DataFrame with log2FoldChange, negLog10PValue, significance, and gene columns
        p_value_threshold: Threshold for -log10(p-value)
        fc_threshold: Threshold for log2 fold change
        
    Returns:
        Plotly figure object
    """
    if 'significance' not in df.columns:
        st.error("Data must be processed first (missing 'significance' column)")
        return None
    
    # Define colors for each category
    color_map = {
        'UP': '#e74c3c',      # Red
        'DOWN': '#3498db',    # Blue
        'NS': '#95a5a6'       # Gray
    }
    
    fig = go.Figure()
    
    # Plot each significance category
    for sig_type in ['NS', 'DOWN', 'UP']:
        subset = df[df['significance'] == sig_type]
        
        fig.add_trace(go.Scatter(
            x=subset['log2FoldChange'],
            y=subset['negLog10PValue'],
            mode='markers',
            name=sig_type,
            marker=dict(
                color=color_map[sig_type],
                size=6,
                opacity=0.6,
                line=dict(width=0)
            ),
            text=subset.get('gene', subset.index),
            hovertemplate='<b>%{text}</b><br>' +
                         'Log2 FC: %{x:.2f}<br>' +
                         '-Log10 p-value: %{y:.2f}<br>' +
                         '<extra></extra>'
        ))
    
    # Add threshold lines
    fig.add_hline(y=p_value_threshold, line_dash="dash", line_color="gray",
                 annotation_text=f"p-value threshold ({10**(-p_value_threshold):.1e})",
                 annotation_position="right")
    fig.add_vline(x=fc_threshold, line_dash="dash", line_color="gray")
    fig.add_vline(x=-fc_threshold, line_dash="dash", line_color="gray")
    
    # Update layout
    fig.update_layout(
        title="Volcano Plot",
        xaxis_title="Log2 Fold Change",
        yaxis_title="-Log10 p-value",
        hovermode='closest',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig
