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
# 3. TRANSFORMATION COMPARISON
# ============================================================================

st.header("3️⃣ Transformation Comparison")
st.info("**Compare different transformations** - Select best for normality and variance stabilization")

from scipy.stats import yeojohnson, boxcox

# Define transformations (complete list)
TRANSFORM_NAMES = {
    "raw": "Raw (No Transform)",
    "log2": "Log2",
    "log10": "Log10",
    "ln": "Natural Log (ln)",
    "sqrt": "Square Root",
    "arcsinh": "Arcsinh",
    "boxcox": "Box-Cox",
    "yeo-johnson": "Yeo-Johnson",
}

# Apply all transformations and collect stats
@st.cache_data
def compute_all_transforms(df_dict: dict, cols: list) -> dict:
    """Compute all transformations and their normality metrics."""
    df = pl.from_dict(df_dict)
    
    # Clean data first
    df_clean = df.with_columns([
        pl.when(pl.col(c).cast(pl.Utf8).str.to_uppercase() == "NAN")
        .then(1.0)
        .when(pl.col(c).is_null())
        .then(1.0)
        .when(pl.col(c) == 0.0)
        .then(1.0)
        .otherwise(pl.col(c))
        .alias(c)
        for c in cols
    ])
    
    transforms = {}
    
    # Raw
    transforms['raw'] = df_clean.to_dict(as_series=False)
    
    # Log2
    transforms['log2'] = df_clean.with_columns([
        pl.col(c).clip(lower_bound=1.0).log(2).alias(c) for c in cols
    ]).to_dict(as_series=False)
    
    # Log10
    transforms['log10'] = df_clean.with_columns([
        pl.col(c).clip(lower_bound=1.0).log10().alias(c) for c in cols
    ]).to_dict(as_series=False)
    
    # Natural log
    transforms['ln'] = df_clean.with_columns([
        pl.col(c).clip(lower_bound=1.0).log().alias(c) for c in cols
    ]).to_dict(as_series=False)
    
    # Square root
    transforms['sqrt'] = df_clean.with_columns([
        pl.col(c).sqrt().alias(c) for c in cols
    ]).to_dict(as_series=False)
    
    # Arcsinh
    transforms['arcsinh'] = df_clean.with_columns([
        pl.col(c).arcsinh().alias(c) for c in cols
    ]).to_dict(as_series=False)
    
    # Box-Cox (requires positive values)
    df_boxcox = df_clean.to_pandas()[cols]
    df_boxcox_transformed = df_boxcox.copy()
    for col in cols:
        data = df_boxcox[col].values
        data_positive = data[data > 0]
        if len(data_positive) > 1:
            transformed, _ = boxcox(data_positive)
            # Map back to original indices
            df_boxcox_transformed.loc[data > 0, col] = transformed
    transforms['boxcox'] = pl.from_pandas(df_boxcox_transformed).to_dict(as_series=False)
    
    # Yeo-Johnson (works with any values)
    df_yj = df_clean.to_pandas()[cols]
    df_yj_transformed = df_yj.copy()
    for col in cols:
        data = df_yj[col].values
        data_finite = data[np.isfinite(data)]
        if len(data_finite) > 1:
            transformed, _ = yeojohnson(data_finite)
            df_yj_transformed.loc[np.isfinite(data), col] = transformed
    transforms['yeo-johnson'] = pl.from_pandas(df_yj_transformed).to_dict(as_series=False)
    
    return transforms
    
    # Raw
    transforms['raw'] = df_clean.to_dict(as_series=False)
    
    # Log2
    transforms['log2'] = df_clean.with_columns([
        pl.col(c).clip(lower_bound=1.0).log(2).alias(c) for c in cols
    ]).to_dict(as_series=False)
    
    # Log10
    transforms['log10'] = df_clean.with_columns([
        pl.col(c).clip(lower_bound=1.0).log10().alias(c) for c in cols
    ]).to_dict(as_series=False)
    
    # Natural log
    transforms['ln'] = df_clean.with_columns([
        pl.col(c).clip(lower_bound=1.0).log().alias(c) for c in cols
    ]).to_dict(as_series=False)
    
    # Square root
    transforms['sqrt'] = df_clean.with_columns([
        pl.col(c).sqrt().alias(c) for c in cols
    ]).to_dict(as_series=False)
    
    # Arcsinh
    transforms['arcsinh'] = df_clean.with_columns([
        pl.col(c).arcsinh().alias(c) for c in cols
    ]).to_dict(as_series=False)
    
    return transforms

# Compute transforms
all_transforms = compute_all_transforms(df.to_dict(as_series=False), numeric_cols)

# Calculate normality metrics for each transform
transform_stats = []

for trans_name, trans_key in TRANSFORM_NAMES.items():
    df_trans = pl.from_dict(all_transforms[trans_name])
    
    # Aggregate all samples
    all_values = []
    for col in numeric_cols:
        data = df_trans[col].to_numpy()
        data_clean = data[np.isfinite(data) & (data > 0 if trans_name == 'raw' else True)]
        all_values.extend(data_clean)
    
    all_values = np.array(all_values)
    
    if len(all_values) > 3:
        # Shapiro test
        stat_sw, pval_sw = stats.shapiro(all_values[:5000])
        
        # Kurtosis & Skewness
        kurt = stats.kurtosis(all_values)
        skew = stats.skew(all_values)
        
        # Mean-variance correlation (across proteins)
        means = df_trans.select(numeric_cols).mean_horizontal().to_numpy()
        vars = df_trans.select(numeric_cols).select([
            pl.var(c) for c in numeric_cols
        ]).row(0)
        
        # Calculate Pearson correlation between mean and variance
        mean_vals = []
        var_vals = []
        for i in range(len(df_trans)):
            row_data = [df_trans[col][i] for col in numeric_cols]
            row_data = [x for x in row_data if np.isfinite(x)]
            if len(row_data) >= 2:
                mean_vals.append(np.mean(row_data))
                var_vals.append(np.var(row_data))
        
        if len(mean_vals) > 2:
            mean_var_corr = np.corrcoef(mean_vals, var_vals)[0, 1]
        else:
            mean_var_corr = np.nan
        
        # Normality score (lower is better)
        norm_score = (1 - pval_sw) + abs(kurt)/10 + abs(skew)/5 + abs(mean_var_corr)
        
        transform_stats.append({
            'Transform': trans_key,
            'Shapiro W': round(stat_sw, 4),
            'Shapiro p': round(pval_sw, 4),
            'Kurtosis': round(kurt, 3),
            'Skewness': round(skew, 3),
            'Mean-Var Corr': round(mean_var_corr, 3),
            'Normality Score': round(norm_score, 3),
            '_key': trans_name
        })

df_transform_stats = pl.DataFrame(transform_stats).sort('Normality Score')

st.subheader("Transformation Statistics (Lower Score = Better Normality)")
st.dataframe(df_transform_stats.to_pandas().drop('_key', axis=1), use_container_width=True)

st.caption("**Normality Score:** Composite metric (lower = better). Based on Shapiro p-value, kurtosis, skewness, and mean-variance correlation.")

# Select transform to compare
selected_transform = st.selectbox(
    "Select transformation to compare with raw:",
    options=list(TRANSFORM_NAMES.keys())[1:],  # Exclude raw
    format_func=lambda x: TRANSFORM_NAMES[x],
    index=0  # Default to log2
)

st.markdown("---")

# ============================================================================
# 4. DIAGNOSTIC PLOTS: RAW VS TRANSFORMED
# ============================================================================

st.header("4️⃣ Diagnostic Plots: Raw vs Transformed")

# Get raw and transformed data
df_raw_np = pl.from_dict(all_transforms['raw'])
df_trans_np = pl.from_dict(all_transforms[selected_transform])

# Create 2 rows of 3 plots each (raw on top, transformed on bottom)
st.subheader(f"Raw Data Diagnostics")

# Row 1: Raw data
col1, col2, col3 = st.columns(3)

# Raw: Distribution
with col1:
    st.markdown("**Distribution**")
    raw_vals = np.concatenate([df_raw_np[c].to_numpy() for c in numeric_cols])
    raw_vals = raw_vals[(raw_vals > 1.0) & np.isfinite(raw_vals)]
    
    mu_raw = np.mean(raw_vals)
    sigma_raw = np.std(raw_vals)
    
    df_plot_raw = pl.DataFrame({'value': raw_vals})
    
    plot_raw_dist = (ggplot(df_plot_raw.to_pandas(), aes(x='value')) +
     geom_histogram(aes(y='..density..'), bins=50, fill='#1f77b4', alpha=0.6) +
     geom_density(color='#1f77b4', size=1.5) +
     geom_vline(xintercept=mu_raw, linetype='dashed', color='red', size=1) +
     annotate('rect', xmin=mu_raw-2*sigma_raw, xmax=mu_raw+2*sigma_raw,
              ymin=-np.inf, ymax=np.inf, alpha=0.1, fill='gray') +
     labs(title='Raw Intensities', x='Intensity', y='Density',
          subtitle=f'μ={mu_raw:.1f}, σ={sigma_raw:.1f}') +
     theme_minimal() +
     theme(figure_size=(4, 3.5), plot_subtitle=element_text(size=8)))
    
    st.pyplot(ggplot.draw(plot_raw_dist))

# Raw: Q-Q plot
with col2:
    st.markdown("**Q-Q Plot**")
    from scipy.stats import probplot
    
    qq_raw = probplot(raw_vals[:5000], dist="norm")
    df_qq_raw = pl.DataFrame({'theoretical': qq_raw[0][0], 'sample': qq_raw[0][1]})
    
    plot_qq_raw = (ggplot(df_qq_raw.to_pandas(), aes(x='theoretical', y='sample')) +
     geom_point(color='#1f77b4', alpha=0.5, size=1) +
     geom_abline(intercept=0, slope=1, color='red', linetype='dashed') +
     labs(title='Q-Q Plot (Raw)', x='Theoretical Quantiles', y='Sample Quantiles') +
     theme_minimal() +
     theme(figure_size=(4, 3.5)))
    
    st.pyplot(ggplot.draw(plot_qq_raw))

# Raw: Mean-Variance
with col3:
    st.markdown("**Mean-Variance Relationship**")
    
    means_raw = []
    vars_raw = []
    for i in range(len(df_raw_np)):
        row_data = [df_raw_np[col][i] for col in numeric_cols]
        row_data = [x for x in row_data if x > 1.0 and np.isfinite(x)]
        if len(row_data) >= 2:
            means_raw.append(np.mean(row_data))
            vars_raw.append(np.var(row_data))
    
    df_mv_raw = pl.DataFrame({'mean': means_raw, 'variance': vars_raw})
    
    plot_mv_raw = (ggplot(df_mv_raw.to_pandas(), aes(x='mean', y='variance')) +
     geom_point(color='#1f77b4', alpha=0.3, size=1) +
     labs(title='Mean-Variance (Raw)', x='Mean', y='Variance') +
     theme_minimal() +
     theme(figure_size=(4, 3.5)))
    
    st.pyplot(ggplot.draw(plot_mv_raw))

# Row 2: Transformed data
st.subheader(f"{TRANSFORM_NAMES[selected_transform]} Diagnostics")

col1, col2, col3 = st.columns(3)

# Transformed: Distribution
with col1:
    st.markdown("**Distribution**")
    trans_vals = np.concatenate([df_trans_np[c].to_numpy() for c in numeric_cols])
    trans_vals = trans_vals[np.isfinite(trans_vals)]
    
    mu_trans = np.mean(trans_vals)
    sigma_trans = np.std(trans_vals)
    
    df_plot_trans = pl.DataFrame({'value': trans_vals})
    
    plot_trans_dist = (ggplot(df_plot_trans.to_pandas(), aes(x='value')) +
     geom_histogram(aes(y='..density..'), bins=50, fill='#ff7f0e', alpha=0.6) +
     geom_density(color='#ff7f0e', size=1.5) +
     geom_vline(xintercept=mu_trans, linetype='dashed', color='darkred', size=1) +
     annotate('rect', xmin=mu_trans-2*sigma_trans, xmax=mu_trans+2*sigma_trans,
              ymin=-np.inf, ymax=np.inf, alpha=0.1, fill='gray') +
     labs(title=f'{TRANSFORM_NAMES[selected_transform]} Intensities', 
          x='Transformed Intensity', y='Density',
          subtitle=f'μ={mu_trans:.2f}, σ={sigma_trans:.2f}') +
     theme_minimal() +
     theme(figure_size=(4, 3.5), plot_subtitle=element_text(size=8)))
    
    st.pyplot(ggplot.draw(plot_trans_dist))

# Transformed: Q-Q plot
with col2:
    st.markdown("**Q-Q Plot**")
    
    qq_trans = probplot(trans_vals[:5000], dist="norm")
    df_qq_trans = pl.DataFrame({'theoretical': qq_trans[0][0], 'sample': qq_trans[0][1]})
    
    plot_qq_trans = (ggplot(df_qq_trans.to_pandas(), aes(x='theoretical', y='sample')) +
     geom_point(color='#ff7f0e', alpha=0.5, size=1) +
     geom_abline(intercept=0, slope=1, color='red', linetype='dashed') +
     labs(title='Q-Q Plot (Transformed)', x='Theoretical Quantiles', y='Sample Quantiles') +
     theme_minimal() +
     theme(figure_size=(4, 3.5)))
    
    st.pyplot(ggplot.draw(plot_qq_trans))

# Transformed: Mean-Variance
with col3:
    st.markdown("**Mean-Variance Relationship**")
    
    means_trans = []
    vars_trans = []
    for i in range(len(df_trans_np)):
        row_data = [df_trans_np[col][i] for col in numeric_cols]
        row_data = [x for x in row_data if np.isfinite(x)]
        if len(row_data) >= 2:
            means_trans.append(np.mean(row_data))
            vars_trans.append(np.var(row_data))
    
    df_mv_trans = pl.DataFrame({'mean': means_trans, 'variance': vars_trans})
    
    plot_mv_trans = (ggplot(df_mv_trans.to_pandas(), aes(x='mean', y='variance')) +
     geom_point(color='#ffb74d', alpha=0.3, size=1) +
     labs(title='Mean-Variance (Transformed)', x='Mean', y='Variance') +
     theme_minimal() +
     theme(figure_size=(4, 3.5)))
    
    st.pyplot(ggplot.draw(plot_mv_trans))

st.markdown("---")

# ============================================================================
# 5. DATA FILTERING RECOMMENDATIONS
# ============================================================================

st.header("5️⃣ Data Filtering & Quality Control")
st.info("**Smart filtering based on your data** - Remove low-quality measurements and apply optimal transformation")

# Group samples by condition
conditions = {}
for col in numeric_cols:
    condition = col[0]  # First letter (A, B, C...)
    if condition not in conditions:
        conditions[condition] = []
    conditions[condition].append(col)

# ============================================================================
# FILTER CONFIGURATION
# ============================================================================

st.subheader("Filter Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    remove_missing = st.checkbox(
        "Remove rows with any missing values",
        value=True,
        help="Drop proteins with at least 1 missing value (≤1.0) across all samples"
    )

with col2:
    cv_filter = st.checkbox(
        "Filter by CV threshold",
        value=True,
        help="Remove proteins with high variability in any condition"
    )
    
    cv_threshold = st.slider(
        "Max CV % per condition:",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
        disabled=not cv_filter,
        help="Drop protein if CV exceeds this in ANY condition"
    )

with col3:
    auto_transform = st.checkbox(
        "Apply best transformation",
        value=True,
        help="Automatically apply transformation with best normality score"
    )
    
    if not auto_transform:
        manual_transform = st.selectbox(
            "Manual selection:",
            options=list(TRANSFORM_NAMES.keys()),
            format_func=lambda x: TRANSFORM_NAMES[x],
            index=1  # Default to log2
        )

# ============================================================================
# PREVIEW FILTERING IMPACT
# ============================================================================

st.subheader("Filtering Impact Preview")

# Start with all proteins
df_filtered = df.clone()
n_original = df_filtered.shape[0]
filter_log = []

# Filter 1: Remove missing values
if remove_missing:
    # Count rows with any missing (≤1.0)
    has_missing = df_filtered.select([
        pl.any_horizontal([
            (pl.col(c) <= 1.0) | (pl.col(c).is_null()) | 
            (pl.col(c).cast(pl.Utf8).str.to_uppercase() == "NAN")
            for c in numeric_cols
        ]).alias('has_missing')
    ])
    
    n_before = df_filtered.shape[0]
    df_filtered = df_filtered.filter(~has_missing['has_missing'])
    n_removed = n_before - df_filtered.shape[0]
    
    filter_log.append(f"**Missing values:** Removed {n_removed} proteins ({n_removed/n_original*100:.1f}%)")

# Filter 2: CV threshold
if cv_filter:
    # Calculate CV for each condition
    cv_filters = []
    
    for condition, cols in conditions.items():
        # Calculate CV
        df_cv = df_filtered.select([id_col] + cols).with_columns([
            pl.concat_list(cols).list.mean().alias('mean'),
            pl.concat_list(cols).list.std().alias('std')
        ]).with_columns(
            (pl.col('std') / pl.col('mean') * 100).alias(f'cv_{condition}')
        )
        
        cv_filters.append(pl.col(f'cv_{condition}') <= cv_threshold)
        df_filtered = df_filtered.with_columns(df_cv[f'cv_{condition}'])
    
    # Keep only proteins where ALL conditions meet CV threshold
    n_before = df_filtered.shape[0]
    df_filtered = df_filtered.filter(pl.all_horizontal(cv_filters))
    n_removed = n_before - df_filtered.shape[0]
    
    filter_log.append(f"**CV threshold (≤{cv_threshold}%):** Removed {n_removed} proteins ({n_removed/n_original*100:.1f}%)")

n_final = df_filtered.shape[0]
retention_rate = n_final / n_original * 100

# Display summary
st.markdown("### Filtering Summary")

for log_entry in filter_log:
    st.markdown(log_entry)

st.markdown(f"**Final dataset:** {n_final:,} proteins retained ({retention_rate:.1f}% of original)")

# Visual summary
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Original", f"{n_original:,}", delta=None)

with col2:
    st.metric("Removed", f"{n_original - n_final:,}", delta=f"-{(n_original-n_final)/n_original*100:.1f}%")

with col3:
    st.metric("Retained", f"{n_final:,}", delta=f"{retention_rate:.1f}%", delta_color="normal")

# ============================================================================
# TRANSFORMATION SELECTION
# ============================================================================

st.subheader("Transformation Selection")

if auto_transform:
    # Get best from stats table
    best_transform_row = df_transform_stats.sort('Normality Score').row(0, named=True)
    best_transform = best_transform_row['_key']
    best_transform_name = best_transform_row['Transform']
    best_score = best_transform_row['Normality Score']
    
    st.success(f"✅ **Recommended:** {best_transform_name} (Normality Score: {best_score:.3f})")
    
    selected_final_transform = best_transform
else:
    selected_final_transform = manual_transform
    st.info(f"📌 **Manual selection:** {TRANSFORM_NAMES[manual_transform]}")

# ============================================================================
# APPLY FILTERS & TRANSFORMATION
# ============================================================================

# ============================================================================
# APPLY FILTERS (NO TRANSFORMATION YET)
# ============================================================================

if st.button("🚀 Apply Filters", type="primary", use_container_width=True):
    
    with st.spinner("Applying filters..."):
        
        # Get species column from session state
        species_col = st.session_state.species_col
        
        # Get clean numeric columns only (drop CV columns if added)
        final_numeric_cols = [c for c in numeric_cols if c in df_filtered.columns]
        
        # Build selection list
        select_cols = [id_col]
        if species_col and species_col in df_filtered.columns:
            select_cols.append(species_col)
        select_cols.extend(final_numeric_cols)
        
        df_filtered_clean = df_filtered.select(select_cols)
        
        # Store in session state (NO TRANSFORMATION)
        st.session_state.df_filtered = df_filtered_clean
        st.session_state.numeric_cols_filtered = final_numeric_cols
        st.session_state.filtering_summary = {
            'original': n_original,
            'final': n_final,
            'retention': retention_rate,
            'cv_threshold': cv_threshold if cv_filter else None,
            'remove_missing': remove_missing
        }
        
        # Store recommended transformation
        if auto_transform:
            best_transform_row = df_transform_stats.sort('Normality Score').row(0, named=True)
            st.session_state.recommended_transform = best_transform_row['_key']
            st.session_state.recommended_transform_name = best_transform_row['Transform']
        else:
            st.session_state.recommended_transform = manual_transform
            st.session_state.recommended_transform_name = TRANSFORM_NAMES[manual_transform]
    
    st.success("✅ Filters applied!")
    st.balloons()
    
    # Show what's next
    st.info("**Next step:** Go to **Quality Overview** page to review filtered data and apply transformation")

# ============================================================================
# DOWNLOAD FILTERED DATA
# ============================================================================

if 'df_filtered' in st.session_state:
    st.markdown("---")
    st.subheader("📥 Download Filtered Data")
    
    st.download_button(
        "Download Filtered Data (CSV)",
        st.session_state.df_filtered.write_csv(),
        "filtered_data.csv",
        "text/csv",
        use_container_width=True
    )

st.markdown("---")

# ============================================================================
# DOWNLOAD FILTERED DATA
# ============================================================================

if 'df_filtered' in st.session_state:
    st.markdown("---")
    st.subheader("📥 Download Filtered Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            "Download Filtered (Raw)",
            st.session_state.df_filtered.write_csv(),
            "filtered_raw.csv",
            "text/csv"
        )
    
    with col2:
        st.download_button(
            "Download Transformed",
            st.session_state.df_transformed.write_csv(),
            "filtered_transformed.csv",
            "text/csv"
        )

st.markdown("---")


# ============================================================================
# PLACEHOLDER FOR NEXT SECTIONS
# ============================================================================

st.header("3️⃣ Group Definition")
st.info("🚧 Coming next: Group selection and differential expression analysis")
