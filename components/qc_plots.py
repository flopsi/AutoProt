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
        
        fig.update_layout(
            xaxis_title=f"PC1 ({explained_var[0]*100:.1f}% variance)",
            yaxis_title=f"PC2 ({explained_var[1]*100:.1f}% variance)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation guide
        with st.expander("📖 How to Interpret PCA"):
            st.markdown("""
            **What to Look For:**
            - **Clustering by condition**: Samples from the same condition should group together
            - **Variance explained**: Higher is better (>60% combined is good)
            - **Outliers**: Points far from their group may indicate technical issues
            - **Separation**: Clear separation between conditions suggests biological differences
            
            **Quality Indicators:**
            - ✅ Good: Clear clustering, >60% variance explained
            - ⚠️ Moderate: Some overlap, 40-60% variance explained
            - ❌ Poor: No clustering, <40% variance explained or obvious outliers
            """)
        
        # Variance metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("PC1 Variance", f"{explained_var[0]*100:.1f}%")
        with col2:
            st.metric("PC2 Variance", f"{explained_var[1]*100:.1f}%")
        with col3:
            total_var = (explained_var[0] + explained_var[1]) * 100
            st.metric("Total Variance", f"{total_var:.1f}%")
            
    except Exception as e:
        st.error(f"PCA failed: {str(e)}")
        st.info("This usually means not enough complete proteins across all samples. Try filtering missing values first.")


def render_missing_value_heatmap(data: pd.DataFrame, replicate_cols: List[str]):
    """
    Render heatmap showing missing value patterns
    
    Args:
        data: DataFrame with protein intensities
        replicate_cols: All replicate column names
    """
    st.markdown("### 🔥 Missing Value Patterns")
    
    # Calculate missing value stats
    missing_stats = calculate_missing_values(data, replicate_cols)
    
    # Create binary matrix (1 = present, 0 = missing) for top proteins
    n_proteins_to_show = min(100, len(data))
    subset = data[replicate_cols].head(n_proteins_to_show)
    binary_matrix = (~subset.isna()).astype(int)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=binary_matrix.values,
        x=replicate_cols,
        y=[f"Protein {i+1}" for i in range(len(binary_matrix))],
        colorscale=[[0, '#e74c3c'], [1, '#2ecc71']],
        showscale=True,
        colorbar=dict(
            title="Status",
            tickvals=[0.25, 0.75],
            ticktext=["Missing", "Present"]
        )
    ))
    
    fig.update_layout(
        title=f"Data Completeness (Top {n_proteins_to_show} Proteins)",
        xaxis_title="Sample",
        yaxis_title="Protein",
        height=600,
        yaxis=dict(showticklabels=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary metrics
    total_missing = missing_stats['Missing_Count'].sum()
    total_possible = len(data) * len(replicate_cols)
    percent_missing = (total_missing / total_possible) * 100
    
    complete_proteins = (data[replicate_cols].notna().all(axis=1)).sum()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Missing", f"{total_missing:,}", 
                 delta=f"{percent_missing:.1f}% of total")
    with col2:
        st.metric("Complete Proteins", f"{complete_proteins:,}",
                 delta=f"{(complete_proteins/len(data)*100):.1f}%")
    with col3:
        worst_sample = missing_stats.loc[missing_stats['Missing_Count'].idxmax(), 'Sample']
        worst_count = missing_stats['Missing_Count'].max()
        st.metric("Most Affected Sample", worst_sample,
                 delta=f"{worst_count:,} missing")
    
    # Sample-wise statistics
    with st.expander("📊 Sample-wise Missing Value Statistics"):
        st.dataframe(missing_stats, use_container_width=True, hide_index=True)


def render_rank_plots(data: pd.DataFrame, condition_names: Dict[str, List[str]], 
                     log_scale: bool = False):
    """
    Render rank-ordered intensity plots for dynamic range visualization
    
    Args:
        data: DataFrame with protein intensities
        condition_names: Dict mapping condition name to list of column names
        log_scale: Whether data is log-transformed
    """
    st.markdown("### 📉 Dynamic Range Analysis")
    
    fig = go.Figure()
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for idx, (condition, cols) in enumerate(condition_names.items()):
        rank_df = calculate_dynamic_range(data, cols)
        
        fig.add_trace(go.Scatter(
            x=rank_df['Rank'],
            y=rank_df['Mean_Intensity'],
            mode='lines',
            name=condition,
            line=dict(color=colors[idx % len(colors)], width=2)
        ))
    
    y_label = "Log2 Mean Intensity" if log_scale else "Mean Intensity"
    fig.update_layout(
        title="Protein Rank Plot (Dynamic Range)",
        xaxis_title="Protein Rank (High to Low)",
        yaxis_title=y_label,
        yaxis_type="log" if not log_scale else "linear",
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Dynamic range summary
    cols_summary = st.columns(len(condition_names))
    for idx, (condition, cols) in enumerate(condition_names.items()):
        with cols_summary[idx]:
            rank_df = calculate_dynamic_range(data, cols)
            if len(rank_df) > 0:
                dynamic_range = rank_df['Mean_Intensity'].max() / rank_df['Mean_Intensity'].min()
                log_range = np.log10(dynamic_range) if dynamic_range > 0 else 0
                
                st.metric(
                    label=f"{condition} - Dynamic Range",
                    value=f"{log_range:.1f} orders",
                    delta=f"{dynamic_range:.1e}x"
                )


def render_qc_dashboard(data: pd.DataFrame, replicate_cols: List[str], 
                       condition_names: Dict[str, List[str]], log_transformed: bool = False):
    """
    Render complete QC dashboard with all visualizations
    
    Args:
        data: DataFrame with protein intensities
        replicate_cols: All replicate column names
        condition_names: Dict mapping condition name to list of column names
        log_transformed: Whether data has been log-transformed
    """
    st.markdown("## 🔬 Quality Control Dashboard")
    st.markdown("---")
    
    # 1. Boxplots
    render_boxplots(data, replicate_cols, condition_names, log_transformed)
    st.markdown("---")
    
    # 2. CV Analysis
    render_cv_analysis(data, condition_names)
    st.markdown("---")
    
    # 3. PCA Plot
    render_pca_plot(data, replicate_cols, condition_names)
    st.markdown("---")
    
    # 4. Missing Value Heatmap
    render_missing_value_heatmap(data, replicate_cols)
    st.markdown("---")
    
    # 5. Rank Plots
    render_rank_plots(data, condition_names, log_transformed)
    st.markdown("---")
    
    st.success("✅ Quality control analysis complete!")
