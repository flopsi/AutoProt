"""
pages/2_Visual_EDA.py - PRODUCTION-READY VISUAL EXPLORATORY DATA ANALYSIS
Violin plots showing intensity distribution per sample, colored by condition
"""

import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Visual EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Visual Exploratory Data Analysis")
st.markdown("Intensity distribution by sample")

# ============================================================================
# DATA VALIDATION
# ============================================================================

if 'data_ready' not in st.session_state or not st.session_state.data_ready:
    st.warning("⚠️ No data loaded. Please upload data on the **📁 Data Upload** page first.")
    st.stop()

# Load from session state
df_raw = st.session_state.df_raw
numeric_cols = st.session_state.numeric_cols
id_col = st.session_state.id_col
species_col = st.session_state.species_col
data_type = st.session_state.data_type

st.success(f"✅ Loaded {data_type} data: {len(df_raw):,} rows × {len(numeric_cols)} samples")
st.markdown("---")

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

# Extract condition from sample name (first letter: A, B, C, etc.)
df_long['Condition'] = df_long['Sample'].str[0]

# ============================================================================
# VIOLIN PLOT
# ============================================================================

st.header("🎻 Intensity Distribution by Sample")
st.caption("All proteins per sample, colored by condition (A, B, C, ...)")

# Create violin plot
fig_violin = px.violin(
    df_long.dropna(subset=['Log2_Intensity']),
    x='Sample',
    y='Log2_Intensity',
    color='Condition',
    title='Log2 Intensity Distribution by Sample',
    labels={
        'Log2_Intensity': 'Log2(Intensity + 1)',
        'Sample': 'Sample',
        'Condition': 'Condition'
    },
    box=True,
    points=False,
    color_discrete_sequence=px.colors.qualitative.Set2
)

fig_violin.update_layout(
    height=600,
    hovermode='closest',
    xaxis_tickangle=-45,
    template='plotly_white',
    showlegend=True
)

st.plotly_chart(fig_violin, width="stretch")

# ============================================================================
# SAMPLE STATISTICS
# ============================================================================

st.subheader("📈 Sample Statistics")

stats_df = pd.DataFrame({
    'Sample': numeric_cols,
    'Condition': [col[0] for col in numeric_cols],
    'N': [df_raw[col].notna().sum() for col in numeric_cols],
    'Mean': [df_raw[col].mean() for col in numeric_cols],
    'Median': [df_raw[col].median() for col in numeric_cols],
    'Std': [df_raw[col].std() for col in numeric_cols],
    'Min': [df_raw[col].min() for col in numeric_cols],
    'Max': [df_raw[col].max() for col in numeric_cols],
}).round(2)

st.dataframe(stats_df, width="stretch", hide_index=True)

st.markdown("---")
st.caption("💡 **Interactive:** Hover for details, click legend to toggle conditions, download as PNG using camera icon")
