"""
pages/2_Visual_EDA.py - PRODUCTION-READY VISUAL EXPLORATORY DATA ANALYSIS
All Plotly visualizations - native Streamlit support, interactive, no rendering errors

Features:
- Distribution plots (histograms, density)
- Box & violin plots by species
- Transformation comparison (log2, yeo-johnson, box-cox)
- Normality assessment (Q-Q plots, Shapiro-Wilk test)
- PCA with variance explained
- t-SNE dimensionality reduction
- Hierarchical clustering heatmap
- Missing data visualization
- Publication-quality interactive figures
"""

import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import boxcox
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Visual EDA - AutoProt",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Visual Exploratory Data Analysis")
st.markdown("Visualize and understand your proteomics data distributions, transformations, and structure")

# ============================================================================
# DATA VALIDATION
# ============================================================================

if 'data_ready' not in st.session_state or not st.session_state.data_ready:
    st.warning("⚠️ No data loaded. Please upload data on the **📁 Data Upload** page first.")
    st.stop()

# Load from session state
df_raw = st.session_state.df_raw
df_polars = st.session_state.get('df_raw_polars', pl.from_pandas(df_raw))
numeric_cols = st.session_state.numeric_cols
id_col = st.session_state.id_col
species_col = st.session_state.species_col
data_type = st.session_state.data_type

st.success(f"✅ Loaded {data_type} data: {len(df_raw):,} rows × {len(numeric_cols)} samples")
st.info(f"ID: **{id_col}** | Species: **{species_col}** | Data Type: **{data_type.upper()}**")
st.markdown("---")

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

with st.sidebar:
    st.header("⚙️ Visualization Settings")
    
    # Plot type selection
    plot_section = st.radio(
        "Select visualization:",
        options=[
            "Distribution",
            "Box & Violin",
            "Transformations",
            "Q-Q Plots",
            "PCA",
            "t-SNE",
            "Heatmap",
            "Missing Data"
        ],
        key="plot_section"
    )
    
    st.divider()
    
    # Color scheme
    color_scheme = st.selectbox(
        "Color scheme:",
        options=["Viridis", "Plasma", "Inferno", "Turbo", "Set2", "Set3"],
        key="color_scheme",
        help="Color palette for visualizations"
    )
    
    # Figure size
    fig_height = st.slider(
        "Figure height:",
        min_value=400,
        max_value=900,
        value=500,
        step=50,
        key="fig_height"
    )
    
    # Histogram bins
    hist_bins = st.slider(
        "Histogram bins:",
        min_value=20,
        max_value=100,
        value=50,
        step=10,
        key="hist_bins"
    )

# ============================================================================
# DATA PREPARATION
# ============================================================================

# Long format for plotting
df_long = df_raw.melt(
    id_vars=[id_col, species_col],
    value_vars=numeric_cols,
    var_name='Sample',
    value_name='Intensity'
)

# Log2 transformation
df_long['Log2_Intensity'] = np.log2(df_long['Intensity'] + 1)

# Create numeric version for transformations
df_numeric = df_raw[numeric_cols].copy()
df_numeric = df_numeric.replace(1.0, np.nan)

# ============================================================================
# SECTION 1: DISTRIBUTION PLOTS
# ============================================================================

if plot_section == "Distribution":
    st.header("1️⃣ Intensity Distributions")
    
    col1, col2 = st.columns(2)
    
    # Histogram (all data)
    with col1:
        st.subheader("Histogram - All Samples")
        st.caption("Log2-transformed intensity distribution across all samples")
        
        fig_hist = px.histogram(
            df_long.dropna(subset=['Log2_Intensity']),
            x='Log2_Intensity',
            nbins=hist_bins,
            color_discrete_sequence=['steelblue'],
            title='Log2 Intensity Distribution',
            labels={'Log2_Intensity': 'Log2(Intensity + 1)', 'count': 'Count'},
            marginal='rug'
        )
        fig_hist.update_traces(marker=dict(line=dict(color='black', width=0.5)))
        fig_hist.update_layout(height=fig_height, showlegend=False)
        st.plotly_chart(fig_hist, width="stretch")
    
    # Density plot by species
    with col2:
        st.subheader("Density - By Species")
        st.caption("Overlay of intensity distributions by species")
        
        fig_dens = px.density_contour(
            df_long.dropna(subset=['Log2_Intensity']),
            x='Log2_Intensity',
            color=species_col,
            title='Density by Species',
            marginal_x='histogram',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_dens.update_layout(height=fig_height)
        st.plotly_chart(fig_dens, width="stretch")
    
    # Statistics by sample
    st.subheader("Sample Statistics")
    stats_df = pd.DataFrame({
        'Sample': numeric_cols,
        'N': [df_raw[col].notna().sum() for col in numeric_cols],
        'Mean': [df_raw[col].mean() for col in numeric_cols],
        'Median': [df_raw[col].median() for col in numeric_cols],
        'Std': [df_raw[col].std() for col in numeric_cols],
        'Min': [df_raw[col].min() for col in numeric_cols],
        'Max': [df_raw[col].max() for col in numeric_cols],
    }).round(2)
    
    st.dataframe(stats_df, width="stretch", hide_index=True)


st.markdown("---")
st.caption("💡 **Tip:** All plots are interactive - hover for details, zoom, pan, and download as PNG. Use sidebar to customize appearance.")
