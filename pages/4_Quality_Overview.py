"""
pages/4_Quality_Overview.py
Quality overview panel after filtering with transformation application
"""

import streamlit as st
import polars as pl
import numpy as np
from plotnine import *
from scipy import stats

# ============================================================================
# LOAD FILTERED DATA
# ============================================================================

st.title("📋 Quality Overview & Transformation")

if 'df_filtered' not in st.session_state:
    st.warning("⚠️ Please apply filters first in Statistical EDA page")
    if st.button("← Go to Statistical EDA"):
        st.switch_page("pages/3_Statistical_EDA.py")
    st.stop()

df = st.session_state.df_filtered
numeric_cols = st.session_state.numeric_cols_filtered
id_col = st.session_state.id_col
species_col = st.session_state.species_col
replicates = st.session_state.replicates

# Display filtering summary
if 'filtering_summary' in st.session_state:
    summary = st.session_state.filtering_summary
    st.info(f"""
    **Filtered Dataset:** {summary['final']:,} proteins ({summary['retention']:.1f}% retained from {summary['original']:,})
    - Missing values removed: {'Yes' if summary['remove_missing'] else 'No'}
    - CV threshold: {summary['cv_threshold']}% (if applied)
    """)

st.markdown("---")

# ============================================================================
# 1. OVERVIEW METRICS
# ============================================================================

st.header("1️⃣ Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

n_proteins = df.shape[0]
n_samples = len(numeric_cols)
n_species = df[species_col].n_unique() if species_col else 1

# Calculate completeness properly
valid_count = 0
total_count = n_proteins * n_samples

for col in numeric_cols:
    valid_count += df.filter((pl.col(col) > 1.0) & (pl.col(col).is_finite())).shape[0]

completeness = (valid_count / total_count * 100) if total_count > 0 else 0

col1.metric("Proteins", f"{n_proteins:,}")
col2.metric("Samples", n_samples)
col3.metric("Species", n_species)
col4.metric("Completeness", f"{completeness:.1f}%")

st.markdown("---")

# ============================================================================
# 2. CV DISTRIBUTION BY CONDITION
# ============================================================================

st.header("2️⃣ CV Distribution by Condition")

# Group by condition
conditions = {}
for col in numeric_cols:
    condition = col[0]
    if condition not in conditions:
        conditions[condition] = []
    conditions[condition].append(col)

# Calculate CVs
cv_data = []
for condition, cols in conditions.items():
    df_cv = df.select([id_col] + cols).with_columns([
        pl.concat_list(cols).list.mean().alias('mean'),
        pl.concat_list(cols).list.std().alias('std')
    ]).with_columns(
        (pl.col('std') / pl.col('mean') * 100).alias('cv')
    ).filter(
        pl.col('cv').is_finite() & (pl.col('cv') > 0)
    )
    
    for row in df_cv.select([id_col, 'cv']).iter_rows(named=True):
        cv_data.append({'condition': condition, 'cv': row['cv']})

df_cv_long = pl.DataFrame(cv_data)

# Violin plot
plot_cv = (ggplot(df_cv_long.to_pandas(), aes(x='condition', y='cv', fill='condition')) +
 geom_violin(alpha=0.6) +
 geom_boxplot(width=0.1, fill='white', outlier_alpha=0) +
 scale_fill_brewer(type='qual', palette='Set2') +
 scale_y_continuous(limits=[0, 50]) +
 labs(title='CV Distribution by Condition (Capped at 50%)',
      x='Condition', y='CV (%)') +
 theme_minimal() +
 theme(figure_size=(10, 5), legend_position='none'))

st.pyplot(ggplot.draw(plot_cv))

# CV summary stats
cv_stats = df_cv_long.group_by('condition').agg([
    pl.col('cv').mean().alias('Mean CV'),
    pl.col('cv').median().alias('Median CV'),
    pl.col('cv').quantile(0.25).alias('Q1'),
    pl.col('cv').quantile(0.75).alias('Q3'),
    (pl.col('cv') < 10).sum().alias('CV < 10%'),
    (pl.col('cv') < 20).sum().alias('CV < 20%')
]).sort('condition')

st.dataframe(cv_stats.to_pandas(), use_container_width=True)

st.markdown("---")

# ============================================================================
# 3. PROTEINS PER SAMPLE PER SPECIES
# ============================================================================

st.header("3️⃣ Valid Proteins per Sample per Species")

# Count valid proteins
df_counts = df.select([id_col, species_col] + numeric_cols).melt(
    id_vars=[id_col, species_col],
    value_vars=numeric_cols,
    variable_name='sample'
).filter(
    (pl.col('value') > 1.0) & (pl.col('value').is_finite())
).group_by(['sample', species_col]).agg(
    pl.count().alias('count')
).sort(['sample', species_col]).with_columns([
    (pl.col('count').cum_sum().over('sample') - pl.col('count') / 2).alias('label_pos')
])

# Stacked bar
plot_species = (ggplot(df_counts.to_pandas(), aes(x='sample', y='count', fill=species_col)) +
 geom_bar(stat='identity') +
 geom_text(aes(y='label_pos', label='count'), size=7, color='white', fontweight='bold') +
 labs(title='Valid Protein Count by Species per Sample',
      x='Sample', y='Count', fill='Species') +
 theme_minimal() +
 theme(axis_text_x=element_text(rotation=45, hjust=1), figure_size=(10, 5)))

st.pyplot(ggplot.draw(plot_species))

st.markdown("---")

# ============================================================================
# 4. APPLY TRANSFORMATION
# ============================================================================

st.header("4️⃣ Apply Transformation")

# Show recommendation
if 'recommended_transform_name' in st.session_state:
    st.success(f"✅ **Recommended:** {st.session_state.recommended_transform_name}")

# Let user confirm or change
transform_choice = st.selectbox(
    "Select transformation to apply:",
    options=['log2', 'log10', 'ln', 'sqrt', 'arcsinh', 'boxcox', 'yeo-johnson'],
    index=0,  # Default to log2
    format_func=lambda x: {'log2': 'Log2', 'log10': 'Log10', 'ln': 'Natural Log', 
                           'sqrt': 'Square Root', 'arcsinh': 'Arcsinh',
                           'boxcox': 'Box-Cox', 'yeo-johnson': 'Yeo-Johnson'}[x]
)

if st.button("🎯 Apply Transformation & Continue", type="primary", use_container_width=True):
    
    with st.spinner(f"Applying {transform_choice} transformation..."):
        
        # Apply transformation
        if transform_choice == 'log2':
            df_trans = df.with_columns([
                pl.col(c).clip(lower_bound=1.0).log(2).alias(c) for c in numeric_cols
            ])
        elif transform_choice == 'log10':
            df_trans = df.with_columns([
                pl.col(c).clip(lower_bound=1.0).log10().alias(c) for c in numeric_cols
            ])
        elif transform_choice == 'ln':
            df_trans = df.with_columns([
                pl.col(c).clip(lower_bound=1.0).log().alias(c) for c in numeric_cols
            ])
        elif transform_choice == 'sqrt':
            df_trans = df.with_columns([
                pl.col(c).sqrt().alias(c) for c in numeric_cols
            ])
        elif transform_choice == 'arcsinh':
            df_trans = df.with_columns([
                pl.col(c).arcsinh().alias(c) for c in numeric_cols
            ])
        elif transform_choice == 'boxcox':
            from scipy.stats import boxcox
            df_pandas = df.to_pandas()
            for col in numeric_cols:
                data = df_pandas[col].values
                data_positive = data[data > 0]
                if len(data_positive) > 1:
                    transformed, _ = boxcox(data_positive)
                    df_pandas.loc[data > 0, col] = transformed
            df_trans = pl.from_pandas(df_pandas)
        elif transform_choice == 'yeo-johnson':
            from scipy.stats import yeojohnson
            df_pandas = df.to_pandas()
            for col in numeric_cols:
                data = df_pandas[col].values
                data_finite = data[np.isfinite(data)]
                if len(data_finite) > 1:
                    transformed, _ = yeojohnson(data_finite)
                    df_pandas.loc[np.isfinite(data), col] = transformed
            df_trans = pl.from_pandas(df_pandas)
        else:
            df_trans = df
        
        # Store transformed data
        st.session_state.df_transformed = df_trans
        st.session_state.transform_applied = transform_choice
    
    st.success(f"✅ {transform_choice} transformation applied!")
    
    # ============================================================================
    # SHOW DIAGNOSTIC PLOTS
    # ============================================================================
    
    st.markdown("---")
    st.subheader(f"📊 Transformation Diagnostics: {transform_choice.upper()}")
    
    # Get all transformed values
    trans_vals = np.concatenate([df_trans[c].to_numpy() for c in numeric_cols])
    trans_vals = trans_vals[np.isfinite(trans_vals)]
    
    if len(trans_vals) > 3:
        # Calculate stats
        mu_trans = np.mean(trans_vals)
        sigma_trans = np.std(trans_vals)
        
        # Normality tests
        stat_sw, pval_sw = stats.shapiro(trans_vals[:5000])
        kurt = stats.kurtosis(trans_vals)
        skew = stats.skew(trans_vals)
        
        # Display normality stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Shapiro W", f"{stat_sw:.4f}")
        col2.metric("Shapiro p", f"{pval_sw:.4e}")
        col3.metric("Kurtosis", f"{kurt:.3f}")
        col4.metric("Skewness", f"{skew:.3f}")
        
        is_normal = pval_sw > 0.05 and abs(kurt) < 2 and abs(skew) < 1
        if is_normal:
            st.success("✅ Data appears normally distributed")
        else:
            st.warning("⚠️ Data may deviate from normality")
        
        # Three diagnostic plots
        col1, col2, col3 = st.columns(3)
        
        # Plot 1: Distribution
        with col1:
            st.markdown("**Distribution**")
            df_plot = pl.DataFrame({'value': trans_vals})
            
            plot_dist = (ggplot(df_plot.to_pandas(), aes(x='value')) +
             geom_histogram(aes(y='..density..'), bins=50, fill='#ff7f0e', alpha=0.6) +
             geom_density(color='#ff7f0e', size=1.5) +
             geom_vline(xintercept=mu_trans, linetype='dashed', color='darkred', size=1) +
             annotate('rect', xmin=mu_trans-2*sigma_trans, xmax=mu_trans+2*sigma_trans,
                      ymin=-np.inf, ymax=np.inf, alpha=0.1, fill='gray') +
             labs(title='Transformed Intensities', x='Value', y='Density',
                  subtitle=f'μ={mu_trans:.2f}, σ={sigma_trans:.2f}') +
             theme_minimal() +
             theme(figure_size=(4, 3.5), plot_subtitle=element_text(size=8)))
            
            st.pyplot(ggplot.draw(plot_dist))
        
        # Plot 2: Q-Q plot
        with col2:
            st.markdown("**Q-Q Plot**")
            from scipy.stats import probplot
            
            qq = probplot(trans_vals[:5000], dist="norm")
            df_qq = pl.DataFrame({'theoretical': qq[0][0], 'sample': qq[0][1]})
            
            plot_qq = (ggplot(df_qq.to_pandas(), aes(x='theoretical', y='sample')) +
             geom_point(color='#ff7f0e', alpha=0.5, size=1) +
             geom_abline(intercept=0, slope=1, color='red', linetype='dashed') +
             labs(title='Q-Q Plot', x='Theoretical Quantiles', y='Sample Quantiles') +
             theme_minimal() +
             theme(figure_size=(4, 3.5)))
            
            st.pyplot(ggplot.draw(plot_qq))
        
        # Plot 3: Mean-Variance
        with col3:
            st.markdown("**Mean-Variance**")
            
            means_trans = []
            vars_trans = []
            for i in range(len(df_trans)):
                row_data = [df_trans[col][i] for col in numeric_cols]
                row_data = [x for x in row_data if np.isfinite(x)]
                if len(row_data) >= 2:
                    means_trans.append(np.mean(row_data))
                    vars_trans.append(np.var(row_data))
            
            df_mv = pl.DataFrame({'mean': means_trans, 'variance': vars_trans})
            
            plot_mv = (ggplot(df_mv.to_pandas(), aes(x='mean', y='variance')) +
             geom_point(color='#ffb74d', alpha=0.3, size=1) +
             labs(title='Mean-Variance', x='Mean', y='Variance') +
             theme_minimal() +
             theme(figure_size=(4, 3.5)))
            
            st.pyplot(ggplot.draw(plot_mv))
    
    st.markdown("---")
    st.info("**Next step:** Proceed to Differential Expression Analysis")

st.markdown("---")
