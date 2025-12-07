"""
pages/2_Visual_EDA_Proteins.py - OPTIMIZED
Exploratory Data Analysis visualizations for protein abundance data
Uses helpers for plots, naming, and analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional
import gc

# Import helpers
from helpers.core import ProteinData
from helpers.analysis import (
    detect_conditions_from_columns,
    group_columns_by_condition,
    create_condition_mapping,
    compute_sample_stats
)
from helpers.naming import (
    get_display_names,
    create_label_rotation_angle,
    standardize_condition_names
)

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Visual EDA - Proteins",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Visual EDA - Proteins")
st.markdown("Exploratory Data Analysis: Distributions, Quality, and Patterns")

# ============================================================================
# DATA VALIDATION
# ============================================================================

if 'data_ready' not in st.session_state or not st.session_state.data_ready:
    st.error("📥 Please upload protein data first on the **Data Upload** page")
    st.stop()

if st.session_state.data_type != 'protein':
    st.error("⚠️ This page is for protein data. Please upload protein data on the **Data Upload** page")
    st.stop()

# Get data from session state
df_raw = st.session_state.df_raw
numeric_cols = st.session_state.numeric_cols
id_col = st.session_state.id_col
species_col = st.session_state.species_col
data_obj: ProteinData = st.session_state.protein_data

# ============================================================================
# PAGE HEADER & METRICS
# ============================================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Proteins", f"{data_obj.n_proteins:,}", help="Number of protein identifiers")

with col2:
    st.metric("Samples", data_obj.n_samples, help="Number of experimental samples")

with col3:
    st.metric("Missing %", f"{data_obj.missing_rate:.1f}%", help="Overall missing data rate")

with col4:
    mean_intensity = df_raw[numeric_cols].mean().mean()
    st.metric("Mean Intensity", f"{mean_intensity:.0f}", help="Average abundance value")

with col5:
    valid_proteins = (df_raw[numeric_cols].notna().sum(axis=1) > 0).sum()
    st.metric("Valid Proteins", f"{valid_proteins:,}", help="Proteins with ≥1 measurement")

st.markdown("---")

# ============================================================================
# CONDITION DETECTION
# ============================================================================

st.subheader("1️⃣ Condition Analysis")

try:
    conditions = detect_conditions_from_columns(numeric_cols)
    condition_map = create_condition_mapping(numeric_cols)
    st.success(f"✅ Detected conditions: {', '.join(conditions)}")
except Exception as e:
    st.warning(f"⚠️ Could not auto-detect conditions: {str(e)}")
    conditions = None
    condition_map = None

if condition_map:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("**Condition Mapping:**")
        mapping_df = pd.DataFrame(list(condition_map.items()), columns=['Sample', 'Condition'])
        st.dataframe(mapping_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.write("**Samples per Condition:**")
        counts = pd.Series(condition_map.values()).value_counts().sort_index()
        for cond, count in counts.items():
            st.metric(f"Condition {cond}", count)

st.markdown("---")

# ============================================================================
# SAMPLE STATISTICS
# ============================================================================

st.subheader("2️⃣ Sample Statistics")

try:
    sample_stats = compute_sample_stats(df_raw, numeric_cols)
    st.dataframe(sample_stats, use_container_width=True, height=300)
except Exception as e:
    st.warning(f"Could not compute sample statistics: {str(e)}")

st.markdown("---")

# ============================================================================
# DISTRIBUTION ANALYSIS
# ============================================================================

st.subheader("3️⃣ Abundance Distributions")

col1, col2 = st.columns(2)

with col1:
    st.write("**Distribution Plots**")
    
    plot_type = st.radio(
        "Select visualization:",
        options=["Box Plot", "Violin Plot", "Histogram"],
        horizontal=True,
        key="dist_plot_type"
    )

with col2:
    st.write("**Options**")
    log_scale = st.checkbox("Log10 scale", value=True, help="Apply log10 transformation for visualization")
    show_points = st.checkbox("Show points", value=False, help="Show individual data points")

# Get display names
display_names = get_display_names(numeric_cols, max_chars=12)
rotation_angle = create_label_rotation_angle(numeric_cols)

# Prepare data for plotting
df_long = df_raw[numeric_cols].melt(
    var_name='Sample',
    value_name='Abundance'
)

if log_scale:
    df_long['Abundance'] = np.log10(df_long['Abundance'] + 1)
    y_label = 'log10(Abundance + 1)'
else:
    y_label = 'Abundance'

# Create mapping for display names
name_mapping = dict(zip(numeric_cols, display_names))
df_long['Sample_Display'] = df_long['Sample'].map(name_mapping)

# Create plot based on selection
try:
    if plot_type == "Box Plot":
        fig = px.box(
            df_long,
            x='Sample_Display',
            y='Abundance',
            title="Abundance Distribution by Sample",
            labels={'Abundance': y_label},
            points='all' if show_points else False,
            hover_name='Sample'
        )
    
    elif plot_type == "Violin Plot":
        fig = px.violin(
            df_long,
            x='Sample_Display',
            y='Abundance',
            title="Abundance Distribution by Sample (Violin)",
            labels={'Abundance': y_label},
            points='all' if show_points else False,
            hover_name='Sample'
        )
    
    else:  # Histogram
        fig = px.histogram(
            df_long,
            x='Abundance',
            nbins=50,
            title=f"Distribution of All Abundance Values",
            labels={'Abundance': y_label},
            marginal='rug'
        )
    
    fig.update_xaxes(tickangle=rotation_angle)
    fig.update_layout(height=500, hovermode='closest')
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error creating distribution plot: {str(e)}")

st.markdown("---")

# ============================================================================
# MISSING DATA ANALYSIS
# ============================================================================

st.subheader("4️⃣ Missing Data Pattern")

col1, col2 = st.columns([2, 1])

with col1:
    # Missing data heatmap (top proteins with most missing)
    missing_per_row = df_raw[numeric_cols].isna().sum(axis=1)
    top_missing_idx = missing_per_row.nlargest(20).index
    
    missing_pattern = df_raw.loc[top_missing_idx, numeric_cols].isna().astype(int)
    missing_pattern.index = df_raw.loc[top_missing_idx, id_col].values
    
    fig = go.Figure(
        data=go.Heatmap(
            z=missing_pattern.values,
            x=get_display_names(numeric_cols, max_chars=10),
            y=missing_pattern.index.astype(str),
            colorscale='Reds',
            text=['Missing' if val == 1 else 'Present' for val in missing_pattern.values.flatten()],
            textposition='auto',
            colorbar=dict(title="Missing")
        )
    )
    fig.update_layout(
        title="Missing Data Pattern (Top 20 Proteins)",
        xaxis_title="Sample",
        yaxis_title="Protein",
        height=400
    )
    fig.update_xaxes(tickangle=rotation_angle)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.write("**Missing Data Summary:**")
    
    missing_by_sample = df_raw[numeric_cols].isna().sum()
    col_display = get_display_names(numeric_cols, max_chars=10)
    
    fig_bar = px.bar(
        x=col_display,
        y=missing_by_sample.values,
        title="Missing Values per Sample",
        labels={'x': 'Sample', 'y': 'Count'}
    )
    fig_bar.update_xaxes(tickangle=rotation_angle)
    fig_bar.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

st.subheader("5️⃣ Sample Correlation")

try:
    # Calculate correlation matrix
    corr_matrix = df_raw[numeric_cols].corr()
    
    # Use display names for heatmap
    corr_display = corr_matrix.copy()
    corr_display.index = get_display_names(numeric_cols, max_chars=10)
    corr_display.columns = get_display_names(numeric_cols, max_chars=10)
    
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_display.values,
            x=corr_display.columns,
            y=corr_display.index,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation")
        )
    )
    fig.update_layout(
        title="Sample-to-Sample Correlation Matrix",
        height=500,
        xaxis_title="Sample",
        yaxis_title="Sample"
    )
    fig.update_xaxes(tickangle=rotation_angle)
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.warning(f"Could not calculate correlation: {str(e)}")

st.markdown("---")

# ============================================================================
# ADVANCED OPTIONS
# ============================================================================

st.subheader("6️⃣ Advanced Analysis")

with st.expander("📊 Additional Visualizations", expanded=False):
    
    adv_col1, adv_col2 = st.columns(2)
    
    with adv_col1:
        st.write("**Protein Intensity Range**")
        
        # Min vs Max for each protein
        protein_min = df_raw[numeric_cols].min(axis=1)
        protein_max = df_raw[numeric_cols].max(axis=1)
        protein_range = protein_max - protein_min
        
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=protein_min,
            y=protein_max,
            mode='markers',
            marker=dict(
                size=5,
                color=protein_range,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Range")
            ),
            text=df_raw[id_col].astype(str),
            hovertemplate='<b>%{text}</b><br>Min: %{x:.0f}<br>Max: %{y:.0f}<extra></extra>'
        ))
        fig_scatter.update_layout(
            title="Protein Min vs Max Abundance",
            xaxis_title="Minimum Abundance",
            yaxis_title="Maximum Abundance",
            height=400
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with adv_col2:
        st.write("**Data Completeness by Protein**")
        
        completeness = df_raw[numeric_cols].notna().sum(axis=1) / len(numeric_cols) * 100
        
        fig_hist = px.histogram(
            x=completeness,
            nbins=30,
            title="Distribution of Data Completeness",
            labels={'x': 'Completeness (%)'},
            marginal='box'
        )
        fig_hist.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# ============================================================================
# DATA EXPORT & ACTIONS
# ============================================================================

st.subheader("7️⃣ Export & Next Steps")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Go to Statistical EDA", use_container_width=True):
        st.switch_page("pages/3_Statistical_EDA.py")

with col2:
    if st.button("📈 Go to Quality Overview", use_container_width=True):
        st.switch_page("pages/4_Quality_Overview.py")

with col3:
    if st.button("🧬 Go to Data Upload", use_container_width=True):
        st.switch_page("pages/1_Data_Upload.py")

st.markdown("---")

# ============================================================================
# FOOTER
# ============================================================================

col1, col2 = st.columns([1, 1])

with col1:
    st.caption("💡 **Tip:** Use the log scale toggle to better visualize wide abundance ranges")

with col2:
    st.caption("📖 **Next:** Explore normality and transformations in the **Statistical EDA** page")

# Cleanup
gc.collect()
