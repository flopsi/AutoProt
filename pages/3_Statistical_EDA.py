"""
pages/3_Statistical_EDA.py
Statistical EDA with filtered data from Visual EDA page - CORRECTED & OPTIMIZED

Key Features:
- Works with filtered protein/peptide data from Visual EDA
- Log2 intensity distributions with KDE
- Normality metrics for both raw and log2 transformed data
- Separate tabs for protein and peptide analysis
- Transformation comparison and recommendations
"""

import streamlit as st
import polars as pl
import numpy as np
from plotnine import *
from scipy import stats
from scipy.stats import probplot, yeojohnson, boxcox
import matplotlib.pyplot as plt
import gc

st.set_page_config(page_title="Statistical EDA", page_icon="📊", layout="wide")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clear_plot_memory():
    """Close all matplotlib figures and collect garbage."""
    plt.close('all')
    gc.collect()

@st.cache_data
def compute_log2(df_dict: dict, cols: list) -> dict:
    """Cache log2 transformation."""
    df_temp = pl.from_dict(df_dict)
    df_log2 = df_temp.with_columns([
        pl.col(c).clip(lower_bound=1.0).log(2).alias(c) for c in cols
    ])
    return df_log2.to_dict(as_series=False)

@st.cache_data
def compute_normality_tests(df_dict: dict, cols: list) -> dict:
    """Compute normality tests for all samples."""
    df_temp = pl.from_dict(df_dict)
    
    results = []
    for col in cols:
        data = df_temp[col].to_numpy()
        data_clean = data[np.isfinite(data)]
        
        if len(data_clean) > 3:
            # Shapiro-Wilk test (use first 5000 points for speed)
            stat_sw, pval_sw = stats.shapiro(data_clean[:5000])
            
            # Kurtosis and Skewness
            kurt = stats.kurtosis(data_clean)
            skew = stats.skew(data_clean)
            
            # Normality assessment
            is_normal = pval_sw > 0.05 and abs(kurt) < 2 and abs(skew) < 1
            
            results.append({
                'Sample': col,
                'Shapiro_W': round(stat_sw, 4),
                'Shapiro_p': pval_sw,
                'Kurtosis': round(kurt, 3),
                'Skewness': round(skew, 3),
                'Normal': '✓' if is_normal else '✗',
                'N': len(data_clean)
            })
    
    return {'results': results}

@st.cache_data
def compute_all_transforms(df_dict: dict, cols: list) -> dict:
    """Compute multiple transformations and their normality metrics."""
    df = pl.from_dict(df_dict)
    
    transforms = {}
    
    # Raw (already processed, just clip to positive)
    transforms['raw'] = df.with_columns([
        pl.col(c).clip(lower_bound=1.0).alias(c) for c in cols
    ]).to_dict(as_series=False)
    
    # Log2
    transforms['log2'] = df.with_columns([
        pl.col(c).clip(lower_bound=1.0).log(2).alias(c) for c in cols
    ]).to_dict(as_series=False)
    
    # Log10
    transforms['log10'] = df.with_columns([
        pl.col(c).clip(lower_bound=1.0).log10().alias(c) for c in cols
    ]).to_dict(as_series=False)
    
    # Natural log
    transforms['ln'] = df.with_columns([
        pl.col(c).clip(lower_bound=1.0).log().alias(c) for c in cols
    ]).to_dict(as_series=False)
    
    # Square root
    transforms['sqrt'] = df.with_columns([
        pl.col(c).sqrt().alias(c) for c in cols
    ]).to_dict(as_series=False)
    
    # Arcsinh
    transforms['arcsinh'] = df.with_columns([
        pl.col(c).arcsinh().alias(c) for c in cols
    ]).to_dict(as_series=False)
    
    # Box-Cox and Yeo-Johnson (need numpy conversion)
    df_pandas = df.select(cols).to_pandas()
    
    # Box-Cox (requires positive values)
    df_boxcox = df_pandas.copy()
    for col in cols:
        data = df_pandas[col].values
        data_positive = data[data > 0]
        if len(data_positive) > 1:
            try:
                transformed, _ = boxcox(data_positive)
                df_boxcox.loc[data > 0, col] = transformed
            except:
                pass
    transforms['boxcox'] = pl.from_pandas(df_boxcox).to_dict(as_series=False)
    
    # Yeo-Johnson (works with any values)
    df_yj = df_pandas.copy()
    for col in cols:
        data = df_pandas[col].values
        data_finite = data[np.isfinite(data)]
        if len(data_finite) > 1:
            try:
                transformed, _ = yeojohnson(data_finite)
                df_yj.loc[np.isfinite(data), col] = transformed
            except:
                pass
    transforms['yeo-johnson'] = pl.from_pandas(df_yj).to_dict(as_series=False)
    
    return transforms

def calculate_transform_scores(all_transforms: dict, numeric_cols: list) -> pl.DataFrame:
    """Calculate normality scores for each transformation."""
    
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
    
    transform_stats = []
    
    for trans_name in all_transforms.keys():
        df_trans = pl.from_dict(all_transforms[trans_name])
        
        # Aggregate all samples
        all_values = []
        for col in numeric_cols:
            data = df_trans[col].to_numpy()
            data_clean = data[np.isfinite(data)]
            all_values.extend(data_clean)
        
        all_values = np.array(all_values)
        
        if len(all_values) > 3:
            # Shapiro test
            stat_sw, pval_sw = stats.shapiro(all_values[:5000])
            
            # Kurtosis & Skewness
            kurt = stats.kurtosis(all_values)
            skew = stats.skew(all_values)
            
            # Mean-variance correlation
            means = []
            variances = []
            for i in range(len(df_trans)):
                row_data = [df_trans[col][i] for col in numeric_cols]
                row_data = [x for x in row_data if np.isfinite(x)]
                if len(row_data) >= 2:
                    means.append(np.mean(row_data))
                    variances.append(np.var(row_data))
            
            if len(means) > 2:
                mean_var_corr = np.corrcoef(means, variances)[0, 1]
            else:
                mean_var_corr = np.nan
            
            # Normality score (lower is better)
            norm_score = (1 - pval_sw) + abs(kurt)/10 + abs(skew)/5 + abs(mean_var_corr)
            
            transform_stats.append({
                'Transform': TRANSFORM_NAMES[trans_name],
                'Shapiro_W': round(stat_sw, 4),
                'Shapiro_p': round(pval_sw, 4),
                'Kurtosis': round(kurt, 3),
                'Skewness': round(skew, 3),
                'Mean_Var_Corr': round(mean_var_corr, 3),
                'Normality_Score': round(norm_score, 3),
                '_key': trans_name
            })
    
    return pl.DataFrame(transform_stats).sort('Normality_Score')

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_intensity_distributions(df, numeric_cols, data_type, transform_type="log2"):
    """Plot intensity distributions in 3x2 grid."""
    
    colors = {'A': '#66c2a5', 'B': '#fc8d62', 'C': '#8da0cb', 'D': '#e78ac3'}
    
    st.subheader(f"{transform_type.upper()} Intensity Distributions")
    st.info(f"**Density histograms with KDE overlay** - {data_type.title()} data with mean line and ±2σ shaded region")
    
    plots = []
    stats_data = []
    
    # Generate up to 6 plots
    for i, col in enumerate(numeric_cols[:6]):
        data = df[col].to_numpy()
        data_clean = data[np.isfinite(data)]
        
        if len(data_clean) < 10:
            continue
        
        # Calculate statistics
        mean_val = np.mean(data_clean)
        std_val = np.std(data_clean)
        
        # Determine condition and color
        cond = col[0]
        color = colors.get(cond, '#999999')
        
        # Create dataframe for plotting
        df_plot = pl.DataFrame({'intensity': data_clean})
        
        # Build plot
        plot = (ggplot(df_plot.to_pandas(), aes(x='intensity')) +
         geom_histogram(aes(y='..density..'), bins=40, fill=color, alpha=0.4, color='black', size=0.3) +
         geom_density(color=color, size=1.5) +
         geom_vline(xintercept=mean_val, linetype='dashed', color='darkred', size=1) +
         annotate('rect', xmin=mean_val-2*std_val, xmax=mean_val+2*std_val,
                  ymin=-np.inf, ymax=np.inf, alpha=0.1, fill='gray') +
         labs(title=f'{col} (Cond {cond})', 
              x=f'{transform_type.upper()} Intensity', 
              y='Density',
              subtitle=f'μ={mean_val:.2f}, σ={std_val:.2f}') +
         theme_minimal() +
         theme(
             figure_size=(4, 3.5),
             plot_title=element_text(size=11, weight='bold'),
             plot_subtitle=element_text(size=9, color='gray'),
             axis_text=element_text(size=8)
         ))
        
        plots.append(plot)
        
        # Collect stats
        stats_data.append({
            'Sample': col,
            'Mean': round(mean_val, 2),
            'SD': round(std_val, 2),
            'Min': round(np.min(data_clean), 2),
            'Max': round(np.max(data_clean), 2),
            'N': len(data_clean)
        })
    
    # Display in 3x2 grid
    if len(plots) > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if len(plots) > 0:
                fig = ggplot.draw(plots[0])
                st.pyplot(fig)
                plt.close(fig)
            if len(plots) > 3:
                fig = ggplot.draw(plots[3])
                st.pyplot(fig)
                plt.close(fig)
        
        with col2:
            if len(plots) > 1:
                fig = ggplot.draw(plots[1])
                st.pyplot(fig)
                plt.close(fig)
            if len(plots) > 4:
                fig = ggplot.draw(plots[4])
                st.pyplot(fig)
                plt.close(fig)
        
        with col3:
            if len(plots) > 2:
                fig = ggplot.draw(plots[2])
                st.pyplot(fig)
                plt.close(fig)
            if len(plots) > 5:
                fig = ggplot.draw(plots[5])
                st.pyplot(fig)
                plt.close(fig)
    
    # Statistics table
    st.markdown(f"**Distribution Statistics ({transform_type.upper()})**")
    if stats_data:
        df_stats = pl.DataFrame(stats_data)
        st.dataframe(df_stats.to_pandas(), use_container_width=True)
    
    del plots, stats_data
    clear_plot_memory()

def plot_diagnostic_comparison(df_raw, df_trans, numeric_cols, transform_name):
    """Plot diagnostic plots comparing raw vs transformed data."""
    
    st.subheader("Raw Data Diagnostics")
    col1, col2, col3 = st.columns(3)
    
    # Concatenate all raw values
    raw_vals = np.concatenate([df_raw[c].to_numpy() for c in numeric_cols])
    raw_vals = raw_vals[np.isfinite(raw_vals)]
    
    # Distribution
    with col1:
        st.markdown("**Distribution**")
        mu_raw = np.mean(raw_vals)
        sigma_raw = np.std(raw_vals)
        
        df_plot = pl.DataFrame({'value': raw_vals})
        
        plot = (ggplot(df_plot.to_pandas(), aes(x='value')) +
         geom_histogram(aes(y='..density..'), bins=50, fill='#1f77b4', alpha=0.6) +
         geom_density(color='#1f77b4', size=1.5) +
         geom_vline(xintercept=mu_raw, linetype='dashed', color='red', size=1) +
         annotate('rect', xmin=mu_raw-2*sigma_raw, xmax=mu_raw+2*sigma_raw,
                  ymin=-np.inf, ymax=np.inf, alpha=0.1, fill='gray') +
         labs(title='Raw Intensities', x='Intensity', y='Density',
              subtitle=f'μ={mu_raw:.1f}, σ={sigma_raw:.1f}') +
         theme_minimal() +
         theme(figure_size=(4, 3.5), plot_subtitle=element_text(size=8)))
        
        fig = ggplot.draw(plot)
        st.pyplot(fig)
        plt.close(fig)
    
    # Q-Q plot
    with col2:
        st.markdown("**Q-Q Plot**")
        qq_raw = probplot(raw_vals[:5000], dist="norm")
        df_qq = pl.DataFrame({'theoretical': qq_raw[0][0], 'sample': qq_raw[0][1]})
        
        plot = (ggplot(df_qq.to_pandas(), aes(x='theoretical', y='sample')) +
         geom_point(color='#1f77b4', alpha=0.5, size=1) +
         geom_abline(intercept=0, slope=1, color='red', linetype='dashed') +
         labs(title='Q-Q Plot (Raw)', x='Theoretical Quantiles', y='Sample Quantiles') +
         theme_minimal() +
         theme(figure_size=(4, 3.5)))
        
        fig = ggplot.draw(plot)
        st.pyplot(fig)
        plt.close(fig)
    
    # Mean-Variance
    with col3:
        st.markdown("**Mean-Variance Relationship**")
        means, variances = [], []
        for i in range(len(df_raw)):
            row_data = [df_raw[col][i] for col in numeric_cols]
            row_data = [x for x in row_data if np.isfinite(x)]
            if len(row_data) >= 2:
                means.append(np.mean(row_data))
                variances.append(np.var(row_data))
        
        df_mv = pl.DataFrame({'mean': means, 'variance': variances})
        
        plot = (ggplot(df_mv.to_pandas(), aes(x='mean', y='variance')) +
         geom_point(color='#1f77b4', alpha=0.3, size=1) +
         labs(title='Mean-Variance (Raw)', x='Mean', y='Variance') +
         theme_minimal() +
         theme(figure_size=(4, 3.5)))
        
        fig = ggplot.draw(plot)
        st.pyplot(fig)
        plt.close(fig)
    
    # Transformed data
    st.subheader(f"{transform_name} Diagnostics")
    col1, col2, col3 = st.columns(3)
    
    trans_vals = np.concatenate([df_trans[c].to_numpy() for c in numeric_cols])
    trans_vals = trans_vals[np.isfinite(trans_vals)]
    
    # Distribution
    with col1:
        st.markdown("**Distribution**")
        mu_trans = np.mean(trans_vals)
        sigma_trans = np.std(trans_vals)
        
        df_plot = pl.DataFrame({'value': trans_vals})
        
        plot = (ggplot(df_plot.to_pandas(), aes(x='value')) +
         geom_histogram(aes(y='..density..'), bins=50, fill='#ff7f0e', alpha=0.6) +
         geom_density(color='#ff7f0e', size=1.5) +
         geom_vline(xintercept=mu_trans, linetype='dashed', color='darkred', size=1) +
         annotate('rect', xmin=mu_trans-2*sigma_trans, xmax=mu_trans+2*sigma_trans,
                  ymin=-np.inf, ymax=np.inf, alpha=0.1, fill='gray') +
         labs(title=f'{transform_name} Intensities', x='Transformed Intensity', y='Density',
              subtitle=f'μ={mu_trans:.2f}, σ={sigma_trans:.2f}') +
         theme_minimal() +
         theme(figure_size=(4, 3.5), plot_subtitle=element_text(size=8)))
        
        fig = ggplot.draw(plot)
        st.pyplot(fig)
        plt.close(fig)
    
    # Q-Q plot
    with col2:
        st.markdown("**Q-Q Plot**")
        qq_trans = probplot(trans_vals[:5000], dist="norm")
        df_qq = pl.DataFrame({'theoretical': qq_trans[0][0], 'sample': qq_trans[0][1]})
        
        plot = (ggplot(df_qq.to_pandas(), aes(x='theoretical', y='sample')) +
         geom_point(color='#ff7f0e', alpha=0.5, size=1) +
         geom_abline(intercept=0, slope=1, color='red', linetype='dashed') +
         labs(title='Q-Q Plot (Transformed)', x='Theoretical Quantiles', y='Sample Quantiles') +
         theme_minimal() +
         theme(figure_size=(4, 3.5)))
        
        fig = ggplot.draw(plot)
        st.pyplot(fig)
        plt.close(fig)
    
    # Mean-Variance
    with col3:
        st.markdown("**Mean-Variance Relationship**")
        means, variances = [], []
        for i in range(len(df_trans)):
            row_data = [df_trans[col][i] for col in numeric_cols]
            row_data = [x for x in row_data if np.isfinite(x)]
            if len(row_data) >= 2:
                means.append(np.mean(row_data))
                variances.append(np.var(row_data))
        
        df_mv = pl.DataFrame({'mean': means, 'variance': variances})
        
        plot = (ggplot(df_mv.to_pandas(), aes(x='mean', y='variance')) +
         geom_point(color='#ff7f0e', alpha=0.3, size=1) +
         labs(title='Mean-Variance (Transformed)', x='Mean', y='Variance') +
         theme_minimal() +
         theme(figure_size=(4, 3.5)))
        
        fig = ggplot.draw(plot)
        st.pyplot(fig)
        plt.close(fig)
    
    clear_plot_memory()

# ============================================================================
# PROCESS DATASET FUNCTION
# ============================================================================

def process_statistical_eda(df, numeric_cols, id_col, data_type, key_prefix):
    """Process statistical EDA for a dataset."""
    
    st.header(f"{data_type.title()}-Level Statistical Analysis")
    
    # Dataset info
    st.info(f"📊 Analyzing **{df.shape[0]:,} {data_type}s** across **{len(numeric_cols)} samples**")
    
    # ========================================================================
    # 1. LOG2 INTENSITY DISTRIBUTIONS
    # ========================================================================
    
    st.markdown("---")
    st.header("1️⃣ Log2 Intensity Distributions")
    
    # Compute log2
    df_log2 = pl.from_dict(compute_log2(df.to_dict(as_series=False), numeric_cols))
    
    # Plot distributions
    plot_intensity_distributions(df_log2, numeric_cols, data_type, "log2")
    
    # ========================================================================
    # 2. NORMALITY TESTS
    # ========================================================================
    
    st.markdown("---")
    st.header("2️⃣ Normality Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Raw Values")
        
        # Prepare raw data (clip to positive)
        df_raw = df.with_columns([
            pl.col(c).clip(lower_bound=1.0).alias(c) for c in numeric_cols
        ])
        
        norm_raw = compute_normality_tests(df_raw.to_dict(as_series=False), numeric_cols)
        df_norm_raw = pl.DataFrame(norm_raw['results'])
        
        st.dataframe(
            df_norm_raw.to_pandas().style.format({
                'Shapiro_p': '{:.4e}',
                'Shapiro_W': '{:.4f}'
            }),
            use_container_width=True
        )
    
    with col2:
        st.subheader("Log2 Transformed")
        
        norm_log2 = compute_normality_tests(df_log2.to_dict(as_series=False), numeric_cols)
        df_norm_log2 = pl.DataFrame(norm_log2['results'])
        
        st.dataframe(
            df_norm_log2.to_pandas().style.format({
                'Shapiro_p': '{:.4e}',
                'Shapiro_W': '{:.4f}'
            }),
            use_container_width=True
        )
    
    st.caption("**Normality criteria:** Shapiro p > 0.05, |Kurtosis| < 2, |Skewness| < 1")
    
    # Summary
    n_normal_raw = df_norm_raw.filter(pl.col('Normal') == '✓').shape[0]
    n_normal_log2 = df_norm_log2.filter(pl.col('Normal') == '✓').shape[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Normal Samples (Raw)", f"{n_normal_raw}/{len(numeric_cols)}")
    with col2:
        st.metric("Normal Samples (Log2)", f"{n_normal_log2}/{len(numeric_cols)}")
    
    # ========================================================================
    # 3. TRANSFORMATION COMPARISON
    # ========================================================================
    
    st.markdown("---")
    st.header("3️⃣ Transformation Comparison")
    st.info("Compare multiple transformations to find the best for normality and variance stabilization")
    
    with st.spinner("Computing transformations..."):
        all_transforms = compute_all_transforms(df.to_dict(as_series=False), numeric_cols)
        transform_scores = calculate_transform_scores(all_transforms, numeric_cols)
    
    st.subheader("Transformation Rankings")
    st.dataframe(
        transform_scores.drop('_key').to_pandas().style.format({
            'Shapiro_W': '{:.4f}',
            'Shapiro_p': '{:.4f}',
            'Kurtosis': '{:.3f}',
            'Skewness': '{:.3f}',
            'Mean_Var_Corr': '{:.3f}',
            'Normality_Score': '{:.3f}'
        }),
        use_container_width=True
    )
    
    st.caption("**Normality Score:** Lower is better. Composite metric based on Shapiro p-value, kurtosis, skewness, and mean-variance correlation.")
    
    # Recommended transformation
    best_transform_row = transform_scores.row(0, named=True)
    best_transform = best_transform_row['Transform']
    best_score = best_transform_row['Normality_Score']
    
    st.success(f"✅ **Recommended transformation:** {best_transform} (Score: {best_score:.3f})")
    
    # ========================================================================
    # 4. DIAGNOSTIC PLOTS
    # ========================================================================
    
    st.markdown("---")
    st.header("4️⃣ Diagnostic Plots: Raw vs Transformed")
    
    # Select transformation to compare
    TRANSFORM_NAMES = {
        "log2": "Log2",
        "log10": "Log10",
        "ln": "Natural Log (ln)",
        "sqrt": "Square Root",
        "arcsinh": "Arcsinh",
        "boxcox": "Box-Cox",
        "yeo-johnson": "Yeo-Johnson"
    }
    
    selected_transform = st.selectbox(
        "Select transformation to compare with raw:",
        options=list(TRANSFORM_NAMES.keys()),
        format_func=lambda x: TRANSFORM_NAMES[x],
        index=0,  # Default to log2
        key=f"{key_prefix}_transform_select"
    )
    
    st.markdown("---")
    
    # Plot diagnostics
    df_raw_diag = pl.from_dict(all_transforms['raw'])
    df_trans_diag = pl.from_dict(all_transforms[selected_transform])
    
    plot_diagnostic_comparison(df_raw_diag, df_trans_diag, numeric_cols, TRANSFORM_NAMES[selected_transform])
    
    # ========================================================================
    # 5. DOWNLOAD OPTIONS
    # ========================================================================
    
    st.markdown("---")
    st.header("5️⃣ Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            "📥 Download Raw Normality Tests (CSV)",
            df_norm_raw.write_csv(),
            f"{data_type}_normality_raw.csv",
            "text/csv",
            use_container_width=True,
            key=f"{key_prefix}_download_raw"
        )
    
    with col2:
        st.download_button(
            "📥 Download Log2 Normality Tests (CSV)",
            df_norm_log2.write_csv(),
            f"{data_type}_normality_log2.csv",
            "text/csv",
            use_container_width=True,
            key=f"{key_prefix}_download_log2"
        )
    
    st.download_button(
        "📥 Download Transformation Comparison (CSV)",
        transform_scores.drop('_key').write_csv(),
        f"{data_type}_transformation_comparison.csv",
        "text/csv",
        use_container_width=True,
        key=f"{key_prefix}_download_transforms"
    )

# ============================================================================
# MAIN APP
# ============================================================================

st.title("📊 Statistical EDA & Distribution Analysis")

# Check data availability
has_protein = 'df_protein_filtered' in st.session_state
has_peptide = 'df_peptide_filtered' in st.session_state

if not has_protein and not has_peptide:
    st.warning("⚠️ No filtered data available. Please complete Visual EDA first.")
    if st.button("← Go to Visual EDA"):
        st.switch_page("pages/2_Visual_EDA.py")
    st.stop()

# Show what's available
st.info("**Analyzing filtered data from Visual EDA**")

col1, col2 = st.columns(2)

with col1:
    if has_protein:
        st.success(f"✅ **Protein data:** {st.session_state.df_protein_filtered.shape[0]:,} proteins")
        if 'protein_filters_applied' in st.session_state:
            filters = st.session_state.protein_filters_applied
            st.caption(f"Filters: CV≤{filters['max_cv']}%, Drop missing: {filters['drop_missing']}")
    else:
        st.info("ℹ️ No protein data")

with col2:
    if has_peptide:
        st.success(f"✅ **Peptide data:** {st.session_state.df_peptide_filtered.shape[0]:,} peptides")
        if 'peptide_filters_applied' in st.session_state:
            filters = st.session_state.peptide_filters_applied
            enabled = []
            if filters['enable_completeness']:
                enabled.append(f"Completeness≥{filters['min_completeness']}%")
            if filters['enable_cv_filter']:
                enabled.append(f"CV≤{filters['max_cv']}%")
            if filters['enable_min_peptides']:
                enabled.append(f"Min {filters['min_peptides']} peptides")
            st.caption(f"Filters: {', '.join(enabled) if enabled else 'None'}")
    else:
        st.info("ℹ️ No peptide data")

st.markdown("---")

# ============================================================================
# CREATE TABS
# ============================================================================

if has_protein and has_peptide:
    tab_protein, tab_peptide = st.tabs(["🧬 Protein Analysis", "🔬 Peptide Analysis"])
elif has_protein:
    tab_protein = st.container()
    tab_peptide = None
else:
    tab_protein = None
    tab_peptide = st.container()

# ============================================================================
# PROTEIN ANALYSIS
# ============================================================================

if has_protein and tab_protein:
    with tab_protein:
        df = st.session_state.df_protein_filtered
        numeric_cols = st.session_state.protein_cols
        id_col = st.session_state.protein_id_col
        
        process_statistical_eda(df, numeric_cols, id_col, 'protein', 'prot')

# ============================================================================
# PEPTIDE ANALYSIS
# ============================================================================

if has_peptide and tab_peptide:
    with tab_peptide:
        df = st.session_state.df_peptide_filtered
        numeric_cols = st.session_state.peptide_cols
        id_col = st.session_state.peptide_id_col
        
        process_statistical_eda(df, numeric_cols, id_col, 'peptide', 'pep')

# ============================================================================
# NAVIGATION
# ============================================================================

clear_plot_memory()

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    if st.button("← Back to Visual EDA", use_container_width=True):
        st.switch_page("pages/2_Visual_EDA.py")

with col2:
    if st.button("Continue to Normalization →", type="primary", use_container_width=True):
        # Check if we have data to continue
        if has_protein or has_peptide:
            st.switch_page("pages/4_Normalization.py")
        else:
            st.warning("No data available to continue")
