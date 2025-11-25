"""
QC Visualization Components for Proteomics Data
Includes Boxplots, CV Analysis, PCA, Heatmaps, and Rank Plots
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict
from utils.stats import (
    calculate_cv, calculate_quartiles, perform_pca, 
    calculate_missing_values, calculate_dynamic_range
)


def render_boxplots(data: pd.DataFrame, replicate_cols: List[str], 
                   condition_names: Dict[str, List[str]], log_scale: bool = False):
    """
    Render boxplots for all replicates grouped by condition
    
    Args:
        data: DataFrame with protein intensities
        replicate_cols: All replicate column names
        condition_names: Dict mapping condition name to list of column names
        log_scale: Whether data is log-transformed
    """
    st.markdown("### 📊 Replicate Intensity Distributions")
    
    # Prepare data for plotting
    plot_data = []
    for condition, cols in condition_names.items():
        for col in cols:
            values = data[col].dropna()
            for val in values:
                plot_data.append({
                    'Replicate': col,
                    'Condition': condition,
                    'Intensity': val
                })
    
    plot_df = pd.DataFrame(plot_data)
    
    if len(plot_df) == 0:
        st.warning("No data available for boxplots")
        return
    
    # Create boxplot
    fig = go.Figure()
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for idx, (condition, cols) in enumerate(condition_names.items()):
        color = colors[idx % len(colors)]
        for col in cols:
            subset = plot_df[plot_df['Replicate'] == col]
            fig.add_trace(go.Box(
                y=subset['Intensity'],
                name=col,
                marker_color=color,
                boxmean='sd',
                legendgroup=condition,
                legendgrouptitle_text=condition
            ))
    
    y_label = "Log2 Intensity" if log_scale else "Intensity"
    fig.update_layout(
        title="Intensity Distribution Across Replicates",
        yaxis_title=y_label,
        xaxis_title="Replicate",
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    with st.expander("📈 Summary Statistics"):
        summary_data = []
        for col in replicate_cols:
            values = data[col].dropna()
            if len(values) > 0:
                quartiles = calculate_quartiles(values)
                summary_data.append({
                    'Replicate': col,
                    'Count': len(values),
                    'Mean': f"{values.mean():.2f}",
                    'Median': f"{quartiles['median']:.2f}",
                    'Std Dev': f"{values.std():.2f}",
                    'Min': f"{quartiles['min']:.2f}",
                    'Max': f"{quartiles['max']:.2f}"
                })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


def render_cv_analysis(data: pd.DataFrame, condition_names: Dict[str, List[str]]):
    """
    Render CV analysis for each condition
    
    Args:
        data: DataFrame with protein intensities
        condition_names: Dict mapping condition name to list of column names
    """
    st.markdown("### 📈 Coefficient of Variation Analysis")
    
    cv_results = {}
    for condition, cols in condition_names.items():
        if len(cols) >= 2:
            cvs = calculate_cv(data, cols)
            cv_results[condition] = cvs.dropna()
    
    if not cv_results:
        st.warning("Need at least 2 replicates per condition for CV analysis")
        return
    
    # Create histogram for each condition
    fig = make_subplots(
        rows=1, cols=len(cv_results),
        subplot_titles=list(cv_results.keys())
    )
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for idx, (condition, cvs) in enumerate(cv_results.items()):
        fig.add_trace(
            go.Histogram(
                x=cvs,
                name=condition,
                marker_color=colors[idx % len(colors)],
                opacity=0.7,
                nbinsx=30
            ),
            row=1, col=idx+1
        )
        
        # Add median line
        median_cv = cvs.median()
        fig.add_vline(
            x=median_cv,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Median: {median_cv:.1f}%",
            row=1, col=idx+1
        )
    
    fig.update_xaxes(title_text="CV (%)")
    fig.update_yaxes(title_text="Count")
    fig.update_layout(
        title="CV Distribution by Condition",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Quality assessment
    cols = st.columns(len(cv_results))
    for idx, (condition, cvs) in enumerate(cv_results.items()):
        with cols[idx]:
            median_cv = cvs.median()
            
            if median_cv < 20:
                status = "🟢 Excellent"
            elif median_cv < 30:
                status = "🟠 Good"
            else:
                status = "🔴 Needs Review"
            
            st.metric(
                label=f"{condition} - Median CV",
                value=f"{median_cv:.1f}%",
                delta=status
            )


def render_pca_plot(data: pd.DataFrame, replicate_cols: List[str], 
                   condition_names: Dict[str, List[str]]):
    """
    Render PCA plot of samples
    
    Args:
        data: DataFrame with protein intensities
        replicate_cols: All replicate column names
        condition_names: Dict mapping condition name to list of column names
    """
    st.markdown("### 🎯 Principal Component Analysis")
    
    try:
        # Perform PCA
        pc_df, explained_var, pca_model = perform_pca(data, replicate_cols, n_components=2)
        
        # Map samples to conditions
        sample_conditions = []
        for sample in pc_df.index:
            for condition, cols in condition_names.items():
                if sample in cols:
                    sample_conditions.append(condition)
                    break
        
        pc_df['Condition'] = sample_conditions
        
        # Create scatter plot
        fig = px.scatter(
            pc_df,
            x='PC1',
            y='PC2',
            color='Condition',
            text=pc_df.index,
            title=f"PCA Plot (PC1: {explained_var[0]*100:.1f}%, PC2: {explained_var[1]*100:.1f}%)"
        )
        
        fig.update_traces(
            textposition='top center',
            marker=dict(size=12, line=dict(width=2, color='white'))
        )
