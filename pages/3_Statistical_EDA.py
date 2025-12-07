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

# ============================================================================
# 1. RAW INTENSITY DISTRIBUTIONS WITH KDE (3x2 GRID)
# ============================================================================

st.header("1️⃣ Raw Intensity Distributions")
st.info("**Density histograms with KDE overlay** - Raw values before transformation")

# Prepare raw data (replace missing with 1.0 for visualization)
df_raw = df.with_columns([
    pl.when(pl.col(c).cast(pl.Utf8).str.to_uppercase() == "NAN")
    .then(1.0)
    .when(pl.col(c).is_null())
    .then(1.0)
    .when(pl.col(c) == 0.0)
    .then(1.0)
    .otherwise(pl.col(c))
    .alias(c)
    for c in numeric_cols
])

# Create long format with condition grouping
df_long_raw = df_raw.select([id_col] + numeric_cols).melt(
    id_vars=[id_col],
    value_vars=numeric_cols,
    variable_name='sample',
    value_name='intensity'
).filter(
    (pl.col('intensity') > 1.0) & (pl.col('intensity').is_finite())
).with_columns(
    pl.when(pl.col('sample').str.starts_with('A'))
    .then(pl.lit('A'))
    .otherwise(pl.lit('B'))
    .alias('condition')
)

# Create 3x2 grid of density plots
plots_raw = []
colors = {'A': '#66c2a5', 'B': '#fc8d62'}

for i, col in enumerate(numeric_cols[:6]):
    df_sample = df_long_raw.filter(pl.col('sample') == col)
    cond = 'A' if col.startswith('A') else 'B'
    color = colors[cond]
    
    plot = (ggplot(df_sample.to_pandas(), aes(x='intensity')) +
     geom_histogram(aes(y='..density..'), bins=50, fill=color, alpha=0.4) +
     geom_density(color=color, size=1.5) +
     labs(title=f'{col} (Condition {cond})', x='Raw Intensity', y='Density') +
     theme_minimal() +
     theme(figure_size=(4, 3)))
    
    plots_raw.append(plot)

# Display in 3x2 grid
col1, col2, col3 = st.columns(3)

with col1:
    if len(plots_raw) > 0:
        st.pyplot(ggplot.draw(plots_raw[0]))
    if len(plots_raw) > 3:
        st.pyplot(ggplot.draw(plots_raw[3]))

with col2:
    if len(plots_raw) > 1:
        st.pyplot(ggplot.draw(plots_raw[1]))
    if len(plots_raw) > 4:
        st.pyplot(ggplot.draw(plots_raw[4]))

with col3:
    if len(plots_raw) > 2:
        st.pyplot(ggplot.draw(plots_raw[2]))
    if len(plots_raw) > 5:
        st.pyplot(ggplot.draw(plots_raw[5]))

# Normality tests (extended)
st.subheader("Normality Tests (Raw Values)")

normality_raw = []
for col in numeric_cols:
    data = df_raw[col].to_numpy()
    data_clean = data[(data > 1.0) & np.isfinite(data)]
    
    if len(data_clean) > 3:
        # Shapiro-Wilk test
        stat_sw, pval_sw = stats.shapiro(data_clean[:5000])
        
        # Kurtosis (excess kurtosis, normal = 0)
        kurt = stats.kurtosis(data_clean)
        
        # Skewness (normal = 0)
        skew = stats.skew(data_clean)
        
        normality_raw.append({
            'Sample': col,
            'Shapiro W': round(stat_sw, 4),
            'Shapiro p': f"{pval_sw:.4e}",
            'Kurtosis': round(kurt, 3),
            'Skewness': round(skew, 3),
            'Normal?': '✓' if pval_sw > 0.05 and abs(kurt) < 2 and abs(skew) < 1 else '✗'
        })

df_normality_raw = pl.DataFrame(normality_raw)
st.dataframe(df_normality_raw.to_pandas(), use_container_width=True)

st.caption("**Interpretation:** Shapiro p>0.05, |Kurtosis|<2, |Skewness|<1 suggest normality")

st.markdown("---")

# ============================================================================
# 2. LOG2 TRANSFORMED DISTRIBUTIONS (3x2 GRID)
# ============================================================================

st.header("2️⃣ Log2 Transformed Intensity Distributions")
st.info("**After log2 transformation** - Should be more normal for statistical tests")

# Compute log2
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

# Long format
df_long_log2 = df_log2.select([id_col] + numeric_cols).melt(
    id_vars=[id_col],
    value_vars=numeric_cols,
    variable_name='sample',
    value_name='log2_intensity'
).filter(
    pl.col('log2_intensity').is_finite()
).with_columns(
    pl.when(pl.col('sample').str.starts_with('A'))
    .then(pl.lit('A'))
    .otherwise(pl.lit('B'))
    .alias('condition')
)

# Create plots
plots_log2 = []

for i, col in enumerate(numeric_cols[:6]):
    df_sample = df_long_log2.filter(pl.col('sample') == col)
    cond = 'A' if col.startswith('A') else 'B'
    color = colors[cond]
    
    plot = (ggplot(df_sample.to_pandas(), aes(x='log2_intensity')) +
     geom_histogram(aes(y='..density..'), bins=50, fill=color, alpha=0.4) +
     geom_density(color=color, size=1.5) +
     labs(title=f'{col} (Condition {cond})', x='Log2 Intensity', y='Density') +
     theme_minimal() +
     theme(figure_size=(4, 3)))
    
    plots_log2.append(plot)

# Display in 3x2 grid
col1, col2, col3 = st.columns(3)

with col1:
    if len(plots_log2) > 0:
        st.pyplot(ggplot.draw(plots_log2[0]))
    if len(plots_log2) > 3:
        st.pyplot(ggplot.draw(plots_log2[3]))

with col2:
    if len(plots_log2) > 1:
        st.pyplot(ggplot.draw(plots_log2[1]))
    if len(plots_log2) > 4:
        st.pyplot(ggplot.draw(plots_log2[4]))

with col3:
    if len(plots_log2) > 2:
        st.pyplot(ggplot.draw(plots_log2[2]))
    if len(plots_log2) > 5:
        st.pyplot(ggplot.draw(plots_log2[5]))

# Normality tests for log2
st.subheader("Normality Tests (Log2 Transformed)")

normality_log2 = []
for col in numeric_cols:
    data = df_log2[col].to_numpy()
    data_clean = data[np.isfinite(data)]
    
    if len(data_clean) > 3:
        stat_sw, pval_sw = stats.shapiro(data_clean[:5000])
        kurt = stats.kurtosis(data_clean)
        skew = stats.skew(data_clean)
        
        normality_log2.append({
            'Sample': col,
            'Shapiro W': round(stat_sw, 4),
            'Shapiro p': f"{pval_sw:.4e}",
            'Kurtosis': round(kurt, 3),
            'Skewness': round(skew, 3),
            'Normal?': '✓' if pval_sw > 0.05 and abs(kurt) < 2 and abs(skew) < 1 else '✗'
        })

df_normality_log2 = pl.DataFrame(normality_log2)
st.dataframe(df_normality_log2.to_pandas(), use_container_width=True)

st.markdown("---")

# ============================================================================
# PLACEHOLDER FOR NEXT SECTIONS
# ============================================================================

st.header("3️⃣ Group Definition")
st.info("🚧 Coming next: Group selection and differential expression analysis")
