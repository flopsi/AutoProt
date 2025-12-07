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
st.info("**Density histograms with KDE overlay** - Raw values with mean line and ±2σ shaded region")

# Prepare raw data
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

# Color mapping
colors = {'A': '#66c2a5', 'B': '#fc8d62'}

# Create 3x2 grid
plots_raw = []
stats_raw = []

for i, col in enumerate(numeric_cols[:6]):
    # Get clean data
    data = df_raw[col].to_numpy()
    data_clean = data[(data > 1.0) & np.isfinite(data)]
    
    if len(data_clean) < 10:
        continue
    
    # Calculate statistics
    mean_val = np.mean(data_clean)
    std_val = np.std(data_clean)
    
    # Determine condition and color
    cond = 'A' if col.startswith('A') else 'B'
    color = colors[cond]
    
    # Create dataframe for plotting
    df_plot = pl.DataFrame({'intensity': data_clean})
    
    # Build plot
    plot = (ggplot(df_plot.to_pandas(), aes(x='intensity')) +
     # Histogram
     geom_histogram(aes(y='..density..'), bins=40, fill=color, alpha=0.4, color='black', size=0.3) +
     # KDE overlay
     geom_density(color=color, size=1.5) +
     # Mean line
     geom_vline(xintercept=mean_val, linetype='dashed', color='darkred', size=1) +
     # ±2σ shaded region
     annotate('rect', xmin=mean_val-2*std_val, xmax=mean_val+2*std_val,
              ymin=-np.inf, ymax=np.inf, alpha=0.1, fill='gray') +
     # Labels
     labs(title=f'{col} (Cond {cond})', 
          x='Raw Intensity', 
          y='Density',
          subtitle=f'μ={mean_val:.1f}, σ={std_val:.1f}') +
     theme_minimal() +
     theme(
         figure_size=(4, 3.5),
         plot_title=element_text(size=11, weight='bold'),
         plot_subtitle=element_text(size=9, color='gray'),
         axis_text=element_text(size=8)
     ))
    
    plots_raw.append(plot)
    
    # Collect stats
    stats_raw.append({
        'Sample': col,
        'Mean': round(mean_val, 2),
        'SD': round(std_val, 2),
        'Min': round(np.min(data_clean), 2),
        'Max': round(np.max(data_clean), 2),
        'N': len(data_clean)
    })

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

# Basic statistics table
st.subheader("Distribution Statistics (Raw)")
df_stats_raw = pl.DataFrame(stats_raw)
st.dataframe(df_stats_raw.to_pandas(), use_container_width=True)

# Normality tests
st.subheader("Normality Tests (Raw Values)")

normality_raw = []
for col in numeric_cols:
    data = df_raw[col].to_numpy()
    data_clean = data[(data > 1.0) & np.isfinite(data)]
    
    if len(data_clean) > 3:
        stat_sw, pval_sw = stats.shapiro(data_clean[:5000])
        kurt = stats.kurtosis(data_clean)
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

st.caption("**Shaded region:** ±2σ from mean | **Red line:** Mean | **Normal if:** p>0.05, |Kurt|<2, |Skew|<1")

st.markdown("---")

# ============================================================================
# 2. LOG2 TRANSFORMED DISTRIBUTIONS (3x2 GRID)
# ============================================================================

st.header("2️⃣ Log2 Transformed Intensity Distributions")
st.info("**After log2 transformation** - Should be more normal for parametric tests")

# Compute log2
df_log2 = pl.from_dict(compute_log2_safe(df.to_dict(as_series=False), numeric_cols))

# Create plots
plots_log2 = []
stats_log2 = []

for i, col in enumerate(numeric_cols[:6]):
    data = df_log2[col].to_numpy()
    data_clean = data[np.isfinite(data)]
    
    if len(data_clean) < 10:
        continue
    
    mean_val = np.mean(data_clean)
    std_val = np.std(data_clean)
    cond = 'A' if col.startswith('A') else 'B'
    color = colors[cond]
    
    df_plot = pl.DataFrame({'log2_intensity': data_clean})
    
    plot = (ggplot(df_plot.to_pandas(), aes(x='log2_intensity')) +
     geom_histogram(aes(y='..density..'), bins=40, fill=color, alpha=0.4, color='black', size=0.3) +
     geom_density(color=color, size=1.5) +
     geom_vline(xintercept=mean_val, linetype='dashed', color='darkred', size=1) +
     annotate('rect', xmin=mean_val-2*std_val, xmax=mean_val+2*std_val,
              ymin=-np.inf, ymax=np.inf, alpha=0.1, fill='gray') +
     labs(title=f'{col} (Cond {cond})', 
          x='Log2 Intensity', 
          y='Density',
          subtitle=f'μ={mean_val:.2f}, σ={std_val:.2f}') +
     theme_minimal() +
     theme(
         figure_size=(4, 3.5),
         plot_title=element_text(size=11, weight='bold'),
         plot_subtitle=element_text(size=9, color='gray'),
         axis_text=element_text(size=8)
     ))
    
    plots_log2.append(plot)
    
    stats_log2.append({
        'Sample': col,
        'Mean': round(mean_val, 2),
        'SD': round(std_val, 2),
        'Min': round(np.min(data_clean), 2),
        'Max': round(np.max(data_clean), 2),
        'N': len(data_clean)
    })

# Display grid
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

# Statistics
st.subheader("Distribution Statistics (Log2)")
df_stats_log2 = pl.DataFrame(stats_log2)
st.dataframe(df_stats_log2.to_pandas(), use_container_width=True)

# Normality tests
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
