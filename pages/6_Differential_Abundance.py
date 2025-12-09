"""
pages/6_Differential_Abundance.py - DIFFERENTIAL EXPRESSION ANALYSIS
Limma-style empirical Bayes moderated t-statistics for proteomics
"""

import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Statistical imports
from scipy import stats
from scipy.stats import t as t_dist
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(str(Path(__file__).parent.parent))

# ============================================================================
# HELPER FUNCTIONS - LIMMA-STYLE EMPIRICAL BAYES
# ============================================================================

def fit_linear_model(df: pd.DataFrame, 
                     group1_cols: List[str], 
                     group2_cols: List[str]) -> pd.DataFrame:
    """
    Fit linear models for each protein (gene).
    
    Args:
        df: Log2-transformed intensity data
        group1_cols: Reference/control samples
        group2_cols: Treatment samples
    
    Returns:
        DataFrame with coefficients, standard errors, sigma
    """
    results = []
    
    for protein_id, row in df.iterrows():
        g1_vals = row[group1_cols].dropna()
        g2_vals = row[group2_cols].dropna()
        
        # Need at least 2 values per group
        if len(g1_vals) < 2 or len(g2_vals) < 2:
            results.append({
                'protein_id': protein_id,
                'log2fc': np.nan,
                'mean_g1': np.nan,
                'mean_g2': np.nan,
                'se': np.nan,
                'sigma': np.nan,
                'df': np.nan,
                'n_g1': len(g1_vals),
                'n_g2': len(g2_vals)
            })
            continue
        
        # Calculate means
        mean_g1 = g1_vals.mean()
        mean_g2 = g2_vals.mean()
        
        # Log2 fold change: log2(g1/g2) = mean_g1 - mean_g2
        log2fc = mean_g1 - mean_g2
        
        # Pooled variance (assuming equal variances)
        n1, n2 = len(g1_vals), len(g2_vals)
        var1 = g1_vals.var(ddof=1)
        var2 = g2_vals.var(ddof=1)
        
        # Pooled standard deviation
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        sigma = np.sqrt(pooled_var)
        
        # Standard error of difference in means
        se = sigma * np.sqrt(1/n1 + 1/n2)
        
        # Degrees of freedom
        df = n1 + n2 - 2
        
        results.append({
            'protein_id': protein_id,
            'log2fc': log2fc,
            'mean_g1': mean_g1,
            'mean_g2': mean_g2,
            'se': se,
            'sigma': sigma,
            'df': df,
            'n_g1': n1,
            'n_g2': n2
        })
    
    results_df = pd.DataFrame(results)
    results_df.set_index('protein_id', inplace=True)
    
    return results_df


def empirical_bayes_moderation(fit_df: pd.DataFrame, 
                                robust: bool = False) -> pd.DataFrame:
    """
    Apply empirical Bayes moderation to variances (limma-style).
    
    Shrinks gene-wise variances towards a common value using
    inverse chi-squared prior distribution.
    
    Args:
        fit_df: Output from fit_linear_model()
        robust: Use robust estimation (downweight outlier variances)
    
    Returns:
        DataFrame with moderated statistics
    """
    # Remove proteins with missing values
    valid_df = fit_df.dropna(subset=['sigma', 'df']).copy()
    
    if len(valid_df) == 0:
        st.error("No valid proteins for moderation")
        return fit_df
    
    # Extract variances and df
    s2 = valid_df['sigma'] ** 2  # Gene-wise variances
    df_gene = valid_df['df']      # Gene-wise degrees of freedom
    
    # === Step 1: Estimate prior hyperparameters ===
    # Method: Method of moments for inverse chi-squared distribution
    
    # Log variances
    log_s2 = np.log(s2)
    
    # Fit linear trend (optional mean-variance trend)
    # For proteomics, we use constant prior (no trend)
    log_s2_mean = log_s2.mean()
    log_s2_var = log_s2.var()
    
    # Prior degrees of freedom (d0) and scale (s0^2)
    # Using method of moments:
    # E[log(s2)] ≈ log(s0^2) - trigamma(d0/2)
    # Var[log(s2)] ≈ trigamma(d0/2)
    
    # Approximate d0 from variance of log variances
    from scipy.special import polygamma
    
    # Solve for d0: trigamma(d0/2) = var(log(s2))
    def trigamma_inv(y):
        """Inverse trigamma function (approximation)"""
        if y < 1e-6:
            return 1/y
        else:
            # Newton-Raphson
            d = 1/y  # Initial guess
            for _ in range(10):
                tri = polygamma(1, d/2)
                tetra = -0.5 * polygamma(2, d/2)
                d = d - (tri - y) / tetra
            return d
    
    d0 = trigamma_inv(log_s2_var)
    d0 = max(d0, 1)  # Ensure positive
    
    # Estimate s0^2
    s0_squared = np.exp(log_s2_mean + polygamma(1, d0/2))
    
    # === Step 2: Posterior variance (moderated) ===
    # Posterior is also inverse chi-squared with:
    # d_post = d0 + df_gene
    # s_post^2 = (d0 * s0^2 + df_gene * s^2) / d_post
    
    df_post = d0 + df_gene
    s2_post = (d0 * s0_squared + df_gene * s2) / df_post
    
    # Add to dataframe
    valid_df['s2_prior'] = s0_squared
    valid_df['df_prior'] = d0
    valid_df['s2_post'] = s2_post
    valid_df['df_post'] = df_post
    
    # Moderated standard error
    valid_df['se_post'] = np.sqrt(s2_post) * valid_df['se'] / valid_df['sigma']
    
    # === Step 3: Moderated t-statistic ===
    valid_df['t_stat'] = valid_df['log2fc'] / valid_df['se_post']
    
    # P-values from t-distribution with df_post
    valid_df['pvalue'] = 2 * (1 - t_dist.cdf(np.abs(valid_df['t_stat']), valid_df['df_post']))
    
    # Merge back with original dataframe
    result_df = fit_df.copy()
    for col in ['s2_prior', 'df_prior', 's2_post', 'df_post', 'se_post', 't_stat', 'pvalue']:
        result_df[col] = valid_df[col]
    
    return result_df


def benjamini_hochberg_fdr(pvalues: pd.Series) -> pd.Series:
    """
    Calculate FDR using Benjamini-Hochberg procedure.
    
    Args:
        pvalues: Series of p-values
    
    Returns:
        Series of FDR-adjusted p-values
    """
    # Remove NaN
    valid_pvals = pvalues.dropna().sort_values()
    n = len(valid_pvals)
    
    if n == 0:
        return pd.Series(index=pvalues.index, dtype=float)
    
    # Benjamini-Hochberg correction
    ranks = np.arange(1, n + 1)
    fdr_vals = valid_pvals.values * n / ranks
    
    # Ensure monotonicity (cumulative minimum from right to left)
    fdr_vals = np.minimum.accumulate(fdr_vals[::-1])[::-1]
    fdr_vals = np.minimum(fdr_vals, 1.0)  # Cap at 1
    
    # Map back to original indices
    fdr_dict = dict(zip(valid_pvals.index, fdr_vals))
    
    return pvalues.index.map(fdr_dict)


def classify_regulation(log2fc: float, 
                         pvalue: float, 
                         fc_threshold: float = 1.0,
                         pval_threshold: float = 0.05) -> str:
    """
    Classify protein regulation status.
    
    Args:
        log2fc: Log2 fold change
        pvalue: Adjusted p-value (FDR)
        fc_threshold: Absolute log2FC cutoff
        pval_threshold: Significance cutoff
    
    Returns:
        "up", "down", or "not_significant"
    """
    if pd.isna(log2fc) or pd.isna(pvalue):
        return "not_tested"
    
    if pvalue > pval_threshold:
        return "not_significant"
    
    if log2fc > fc_threshold:
        return "up"
    elif log2fc < -fc_threshold:
        return "down"
    else:
        return "not_significant"

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Differential Abundance",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Differential Abundance Analysis")
st.markdown("Limma-style empirical Bayes moderated t-statistics")
st.markdown("---")

# ============================================================================
# CHECK FOR IMPUTED DATA
# ============================================================================

if 'df_imputed' not in st.session_state or st.session_state.df_imputed is None:
    st.error("❌ No imputed data. Please complete **🔧 Missing Value Imputation** first")
    st.stop()

# Load data
df = st.session_state.df_imputed.copy()
numeric_cols = st.session_state.numeric_cols
sample_to_condition = st.session_state.get('sample_to_condition', {})
species_col = st.session_state.species_col

# Get conditions
conditions = sorted(list(set(sample_to_condition.values())))
condition_samples = {}
for sample, condition in sample_to_condition.items():
    if sample in numeric_cols:
        if condition not in condition_samples:
            condition_samples[condition] = []
        condition_samples[condition].append(sample)

st.info(f"📊 **Data**: {len(df):,} proteins × {len(numeric_cols)} samples | **Conditions**: {', '.join(conditions)}")

# ============================================================================
# 1. EXPERIMENTAL DESIGN
# ============================================================================

st.subheader("1️⃣ Select Comparison Groups")

col1, col2 = st.columns(2)

with col1:
    reference_group = st.selectbox(
        "Reference Group (Control):",
        options=conditions,
        index=0,
        help="Baseline/control condition"
    )

with col2:
    treatment_group = st.selectbox(
        "Treatment Group:",
        options=[c for c in conditions if c != reference_group],
        index=0 if len(conditions) > 1 else None,
        help="Treatment/experimental condition"
    )

if reference_group == treatment_group:
    st.error("❌ Please select different groups for comparison")
    st.stop()

# Get sample columns for each group
ref_samples = condition_samples[reference_group]
treat_samples = condition_samples[treatment_group]

st.markdown(f"""
**Comparison**: `{reference_group}` (n={len(ref_samples)}) vs `{treatment_group}` (n={len(treat_samples)})

**Log2FC Convention**: Positive FC = higher in **{reference_group}** | Negative FC = higher in **{treatment_group}**
""")

st.markdown("---")

# ============================================================================
# 2. THEORETICAL FOLD CHANGES (OPTIONAL)
# ============================================================================

st.subheader("2️⃣ Theoretical Fold Changes (Optional)")

st.markdown("""
For spike-in or simulation studies, provide expected fold changes to calculate error metrics.
Leave empty for real biological data with unknown true fold changes.
""")

use_theoretical = st.checkbox("I have theoretical/expected fold changes")

theoretical_fc_dict = {}

if use_theoretical:
    st.markdown("**Enter theoretical fold changes per species:**")
    
    # Get unique species
    species_list = sorted(df[species_col].unique())
    
    fc_input_method = st.radio(
        "Input method:",
        options=["Log2 Fold Change", "Ratio (e.g., 2:1, 1:2)"],
        horizontal=True
    )
    
    col1, col2, col3 = st.columns(3)
    
    for idx, species in enumerate(species_list[:6]):  # Limit to 6 species for UI
        with [col1, col2, col3][idx % 3]:
            if fc_input_method == "Log2 Fold Change":
                fc_value = st.number_input(
                    f"{species}:",
                    value=0.0,
                    step=0.5,
                    format="%.2f",
                    key=f"fc_{species}",
                    help="log2(Ref/Treatment)"
                )
                theoretical_fc_dict[species] = fc_value
            else:
                ratio_str = st.text_input(
                    f"{species} (Ref:Treatment):",
                    value="1:1",
                    key=f"ratio_{species}",
                    help="e.g., 2:1 or 1:2"
                )
                try:
                    ref_ratio, treat_ratio = map(float, ratio_str.split(':'))
                    log2fc = np.log2(ref_ratio / treat_ratio)
                    theoretical_fc_dict[species] = log2fc
                except:
                    st.warning(f"Invalid ratio for {species}")
                    theoretical_fc_dict[species] = 0.0

st.markdown("---")

# ============================================================================
# 3. STATISTICAL PARAMETERS
# ============================================================================

st.subheader("3️⃣ Statistical Parameters")

col1, col2 = st.columns(2)

with col1:
    fc_threshold = st.slider(
        "Log2 Fold Change Threshold:",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Minimum absolute log2FC for biological significance"
    )
    
    st.markdown(f"""
    **FC cutoff**: ±{fc_threshold} log2FC  
    **Linear scale**: {2**fc_threshold:.2f}-fold change
    """)

with col2:
    pval_threshold = st.selectbox(
        "FDR Significance Threshold:",
        options=[0.01, 0.05, 0.10],
        index=1,
        format_func=lambda x: f"{x*100:.0f}% FDR",
        help="False Discovery Rate cutoff"
    )
    
    use_fdr = st.checkbox("Use FDR correction", value=True, help="Benjamini-Hochberg FDR")

st.markdown("---")

# ============================================================================
# 4. RUN ANALYSIS
# ============================================================================

st.subheader("4️⃣ Run Differential Abundance Analysis")

if st.button("🚀 Run Analysis", type="primary"):
    with st.spinner("Performing limma-style analysis..."):
        
        # Log2 transform if not already (data should be log2 transformed)
        df_log2 = df[numeric_cols].apply(lambda x: np.log2(x + 1) if x.min() > 1 else x)
        
        # Step 1: Fit linear models
        fit_results = fit_linear_model(df_log2, ref_samples, treat_samples)
        
        # Step 2: Empirical Bayes moderation
        moderated_results = empirical_bayes_moderation(fit_results)
        
        # Step 3: FDR correction
        if use_fdr:
            moderated_results['fdr'] = benjamini_hochberg_fdr(moderated_results['pvalue'])
            test_col = 'fdr'
        else:
            moderated_results['fdr'] = moderated_results['pvalue']
            test_col = 'pvalue'
        
        # Step 4: Classification
        moderated_results['regulation'] = moderated_results.apply(
            lambda row: classify_regulation(row['log2fc'], row[test_col], fc_threshold, pval_threshold),
            axis=1
        )
        
        # Step 5: Add -log10(p) for volcano
        moderated_results['neg_log10_pval'] = -np.log10(moderated_results[test_col].replace(0, 1e-300))
        
        # Add species information
        moderated_results['species'] = moderated_results.index.map(
            lambda x: df.loc[x, species_col] if x in df.index else 'Unknown'
        )
        
        # Store in session state
        st.session_state.dea_results = moderated_results
        st.session_state.dea_ref_group = reference_group
        st.session_state.dea_treat_group = treatment_group
        st.session_state.dea_fc_threshold = fc_threshold
        st.session_state.dea_pval_threshold = pval_threshold
        
        st.success("✅ Analysis complete!")

# ============================================================================
# 5. RESULTS VISUALIZATION
# ============================================================================

if 'dea_results' in st.session_state:
    results = st.session_state.dea_results
    
    st.markdown("---")
    st.subheader("5️⃣ Results Summary")
    
    # Summary statistics
    n_total = len(results)
    n_up = (results['regulation'] == 'up').sum()
    n_down = (results['regulation'] == 'down').sum()
    n_ns = (results['regulation'] == 'not_significant').sum()
    n_tested = n_total - (results['regulation'] == 'not_tested').sum()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Proteins", f"{n_total:,}")
    col2.metric("Upregulated", f"{n_up:,}", delta=f"{n_up/n_tested*100:.1f}%")
    col3.metric("Downregulated", f"{n_down:,}", delta=f"{n_down/n_tested*100:.1f}%")
    col4.metric("Not Significant", f"{n_ns:,}")
    
    # Volcano plot
    st.markdown("### 🌋 Volcano Plot")
    
    volcano_df = results[results['regulation'] != 'not_tested'].copy()
    volcano_df['Regulation'] = volcano_df['regulation'].map({
        'up': f'Up ({n_up})',
        'down': f'Down ({n_down})',
        'not_significant': f'NS ({n_ns})'
    })
    
    color_map = {
        f'Up ({n_up})': '#FF6B6B',
        f'Down ({n_down})': '#4ECDC4',
        f'NS ({n_ns})': '#95A5A6'
    }
    
    fig = px.scatter(
        volcano_df,
        x='log2fc',
        y='neg_log10_pval',
        color='Regulation',
        hover_data=['species'],
        color_discrete_map=color_map,
        title=f'Volcano Plot: {st.session_state.dea_ref_group} vs {st.session_state.dea_treat_group}',
        labels={
            'log2fc': f'Log2 Fold Change ({st.session_state.dea_ref_group} / {st.session_state.dea_treat_group})',
            'neg_log10_pval': '-Log10(FDR)' if use_fdr else '-Log10(p-value)'
        },
        height=600
    )
    
    # Add threshold lines
    fig.add_hline(y=-np.log10(pval_threshold), line_dash="dash", line_color="gray", annotation_text=f"FDR = {pval_threshold}")
    fig.add_vline(x=fc_threshold, line_dash="dash", line_color="gray")
    fig.add_vline(x=-fc_threshold, line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Results table
    st.markdown("### 📋 Differentially Abundant Proteins")
    
    sig_results = results[results['regulation'].isin(['up', 'down'])].copy()
    sig_results = sig_results.sort_values('fdr')
    
    display_cols = ['log2fc', 'mean_g1', 'mean_g2', 't_stat', 'pvalue', 'fdr', 'regulation', 'species']
    display_df = sig_results[display_cols].head(50)
    
    st.dataframe(display_df.round(4), use_container_width=True)
    
    # Download results
    st.markdown("### 💾 Export Results")
    
    csv = results.to_csv()
    st.download_button(
        label="📥 Download Full Results (CSV)",
        data=csv,
        file_name=f"dea_{reference_group}_vs_{treatment_group}.csv",
        mime="text/csv"
    )
    
    st.success(f"✅ Analysis complete! Found {n_up + n_down:,} differentially abundant proteins")

else:
    st.info("👆 Configure parameters and click 'Run Analysis' to begin")
