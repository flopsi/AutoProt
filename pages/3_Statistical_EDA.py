"""
pages/3_Statistical_EDA.py
Simplified Polars version - density distributions and differential expression
"""

import streamlit as st
import polars as pl
import numpy as np
from plotnine import *
from scipy import stats

# ============================================================================
# LOAD DATA
# ============================================================================

st.title("🧪 Statistical EDA & Differential Expression")

if 'df' not in st.session_state:
    st.warning("⚠️ Please upload data first")
    st.stop()

df = st.session_state.df
numeric_cols = st.session_state.numeric_cols
id_col = st.session_state.id_col
replicates = st.session_state.replicates

# Compute log2 if not cached
@st.cache_data
def compute_log2_safe(df_dict: dict, cols: list) -> dict:
    """Cache log2 transformation with missing value handling."""
    df = pl.from_dict(df_dict)
    
    df_log2 = df.with_columns([
        pl.when(pl.col(c).cast(pl.Utf8).str.to_uppercase() == "NAN")
        .then(1.0)
        .when(pl.col(c).is_null())
        .then(1.0)
        .when(pl.col(c) == 0.0)
        .then(1.0)
        .otherwise(pl.col(c))
        .clip(lower_bound=1.0)
        .log(2)
        .alias(c)
        for c in cols
    ])
    
    return df_log2.to_dict(as_series=False)

df_log2 = pl.from_dict(compute_log2_safe(df.to_dict(as_series=False), numeric_cols))

# ============================================================================
# 1. DENSITY HISTOGRAMS WITH KDE (3x2 GRID)
# ============================================================================

st.header("1️⃣ Log2 Intensity Distributions")
st.info("**Density histograms with KDE overlay** - Check for normality and batch effects")

# Create long format for plotting
df_long = df_log2.select([id_col] + numeric_cols).melt(
    id_vars=[id_col],
    value_vars=numeric_cols,
    variable_name='sample',
    value_name='log2_intensity'
).filter(
    pl.col('log2_intensity').is_finite()
)

# Create 3x2 grid of density plots
plots = []
for i, col in enumerate(numeric_cols[:6]):  # First 6 samples
    df_sample = df_long.filter(pl.col('sample') == col)
    
    plot = (ggplot(df_sample.to_pandas(), aes(x='log2_intensity')) +
     geom_histogram(aes(y='..density..'), bins=50, fill='lightblue', alpha=0.6) +
     geom_density(color='darkblue', size=1.5) +
     labs(title=col, x='Log2 Intensity', y='Density') +
     theme_minimal() +
     theme(figure_size=(4, 3)))
    
    plots.append(plot)

# Display in 3x2 grid
col1, col2, col3 = st.columns(3)

with col1:
    if len(plots) > 0:
        st.pyplot(ggplot.draw(plots[0]))
    if len(plots) > 3:
        st.pyplot(ggplot.draw(plots[3]))

with col2:
    if len(plots) > 1:
        st.pyplot(ggplot.draw(plots[1]))
    if len(plots) > 4:
        st.pyplot(ggplot.draw(plots[4]))

with col3:
    if len(plots) > 2:
        st.pyplot(ggplot.draw(plots[2]))
    if len(plots) > 5:
        st.pyplot(ggplot.draw(plots[5]))

# Normality tests
st.subheader("Normality Tests (Shapiro-Wilk)")

normality_results = []
for col in numeric_cols:
    data = df_log2[col].to_numpy()
    data_clean = data[np.isfinite(data)]
    
    if len(data_clean) > 3:
        stat, pval = stats.shapiro(data_clean[:5000])  # Limit to 5000 for speed
        normality_results.append({
            'Sample': col,
            'W-statistic': round(stat, 4),
            'P-value': f"{pval:.4e}",
            'Normal?': '✓' if pval > 0.05 else '✗'
        })

df_normality = pl.DataFrame(normality_results)
st.dataframe(df_normality.to_pandas(), use_container_width=True)

st.markdown("---")

# ============================================================================
# PLACEHOLDER FOR NEXT SECTIONS
# ============================================================================

st.header("2️⃣ Group Definition")
st.info("🚧 Coming next: Group selection and differential expression analysis")

st.header("3️⃣ Statistical Tests")
st.info("🚧 Coming next: T-test/ANOVA with volcano plots")
