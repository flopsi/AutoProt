"""
pages/2_EDA.py - Exploratory Data Analysis
Visualize protein/peptide abundance data using plotnine
"""

import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
from plotnine import *
from plotnine.data import *
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Exploratory Data Analysis")
st.markdown("Visualize and explore your proteomics data")

# ============================================================================
# CHECK DATA AVAILABILITY
# ============================================================================

if 'data_ready' not in st.session_state or not st.session_state.data_ready:
    st.warning("⚠️ No data loaded. Please upload data on the **Data Upload** page first.")
    st.stop()

# Load data from session state
df = st.session_state.df_raw
numeric_cols = st.session_state.numeric_cols
id_col = st.session_state.id_col
data_type = st.session_state.data_type

st.success(f"✅ Loaded {data_type} data: {len(df):,} rows × {len(numeric_cols)} samples")

st.markdown("---")

# ============================================================================
# DATA PREPARATION
# ============================================================================

# Prepare long-format data for plotting
df_long = df.melt(
    id_vars=[id_col],
    value_vars=numeric_cols,
    var_name='Sample',
    value_name='Intensity'
)

# Log2 transform intensities
df_long['Log2_Intensity'] = np.log2(df_long['Intensity'])

# Replace -inf with NaN (from log2(0))
df_long['Log2_Intensity'] = df_long['Log2_Intensity'].replace([np.inf, -np.inf], np.nan)

st.markdown("---")

# ============================================================================
# PLOT 1: INTENSITY DISTRIBUTION (HISTOGRAM)
# ============================================================================

st.subheader("1️⃣ Intensity Distribution")
st.caption("Histogram showing the distribution of log2-transformed intensities across all samples")


plot1 = (
    ggplot(df_long, aes(x='Log2_Intensity')) +
    geom_histogram(bins=hist_bins, fill='steelblue', alpha=hist_alpha, color='black') +
    labs(
        title='Distribution of Log2 Intensities',
        x='Log2(Intensity)',
        y='Count'
    ) +
    theme_minimal() +
    theme(figure_size=(10, 5))
)

st.pyplot(ggplot.draw(plot1))

st.markdown("---")


# ============================================================================
# FOOTER
# ============================================================================

st.caption("💡 **Tip:** Use the sidebar controls to customize each plot's appearance.")
