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
# PLOT 2: BOXPLOT PER SAMPLE
# ============================================================================

# ============================================================================
# PLOT 3: VIOLIN PLOT PER SAMPLE
# ============================================================================

st.subheader("3️⃣ Density Distribution per Sample")
st.caption("Violin plot showing the density of log2 intensities for each sample")

col1, col2 = st.columns([3, 1])

with col2:
    violin_scale = st.selectbox(
        "Scale method:",
        options=['area', 'count', 'width'],
        index=0,
        key="violin_scale"
    )
    add_boxplot = st.checkbox("Add boxplot overlay", value=True, key="violin_box")

with col1:
    plot3 = (
        ggplot(df_long, aes(x='Sample', y='Log2_Intensity', fill='Sample')) +
        geom_violin(scale=violin_scale, alpha=0.6)
    )
    
    if add_boxplot:
        plot3 = plot3 + geom_boxplot(width=0.2, alpha=0.8, outlier_alpha=0)
    
    plot3 = (
        plot3 +
        labs(
            title='Violin Plot of Log2 Intensities by Sample',
            x='Sample',
            y='Log2(Intensity)'
        ) +
        theme_minimal() +
        theme(
            figure_size=(12, 6),
            axis_text_x=element_text(rotation=45, hjust=1),
            legend_position='none'
        )
    )
    
    st.pyplot(ggplot.draw(plot3))

st.markdown("---")

# ============================================================================
# PLOT 4: DENSITY PLOT (OVERLAID)
# ============================================================================

st.subheader("4️⃣ Overlaid Density Curves")
st.caption("Density curves for each sample overlaid on the same plot")

col1, col2 = st.columns([3, 1])

with col2:
    density_alpha = st.slider("Transparency:", 0.0, 1.0, 0.5, key="density_alpha")
    show_legend = st.checkbox("Show legend", value=True, key="density_legend")

with col1:
    plot4 = (
        ggplot(df_long, aes(x='Log2_Intensity', color='Sample', fill='Sample')) +
        geom_density(alpha=density_alpha) +
        labs(
            title='Density Plot of Log2 Intensities',
            x='Log2(Intensity)',
            y='Density'
        ) +
        theme_minimal() +
        theme(
            figure_size=(10, 6),
            legend_position='right' if show_legend else 'none'
        )
    )
    
    st.pyplot(ggplot.draw(plot4))

st.markdown("---")



# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

st.subheader("📈 Summary Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Proteins/Peptides",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Total Samples",
        len(numeric_cols)
    )

with col3:
    missing_pct = (df[numeric_cols].isna().sum().sum() / (len(df) * len(numeric_cols)) * 100)
    st.metric(
        "Missing Values",
        f"{missing_pct:.1f}%"
    )

with col4:
    mean_intensity = df[numeric_cols].mean().mean()
    st.metric(
        "Mean Log2 Intensity",
        f"{mean_intensity:.2f}"
    )

# Show summary table
with st.expander("📋 Detailed Statistics per Sample"):
    stats_df = pd.DataFrame({
        'Sample': numeric_cols,
        'Mean': [df[col].mean() for col in numeric_cols],
        'Median': [df[col].median() for col in numeric_cols],
        'Std Dev': [df[col].std() for col in numeric_cols],
        'Missing %': [(df[col].isna().sum() / len(df) * 100) for col in numeric_cols]
    })
    st.dataframe(stats_df, use_container_width=True)

st.markdown("---")

# ============================================================================
# FOOTER
# ============================================================================

st.caption("💡 **Tip:** Use the sidebar controls to customize each plot's appearance.")
