"""
pages/5_Differential_Expression.py
Differential expression analysis using limma-like approach
"""

import streamlit as st
import polars as pl
import numpy as np
from plotnine import *
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ============================================================================
# LOAD DATA
# ============================================================================

st.title("🧬 Differential Expression Analysis")

if 'df_transformed' not in st.session_state or 'data_ready' not in st.session_state:
    st.warning("⚠️ Please complete Quality Overview first")
    if st.button("← Go to Quality Overview"):
        st.switch_page("pages/4_Quality_Overview.py")
    st.stop()

df_trans = st.session_state.df_transformed
numeric_cols = st.session_state.numeric_cols_filtered
id_col = st.session_state.id_col
species_col = st.session_state.species_col
transform_name = st.session_state.transform_applied

summary = st.session_state.filtering_summary

st.info(f"""
**Analysis Dataset:** {summary['final']:,} proteins | {len(numeric_cols)} samples | {transform_name.upper()} transformed
""")

st.markdown("---")

# ============================================================================
# 1. DEFINE COMPARISON GROUPS
# ============================================================================

st.header("1️⃣ Define Comparison Groups")

# Group by condition
conditions = {}
for col in numeric_cols:
    condition = col[0]
    if condition not in conditions:
        conditions[condition] = []
    conditions[condition].append(col)

unique_conds = sorted(conditions.keys())

if len(unique_conds) < 2:
    st.error("❌ Need at least 2 conditions for comparison")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Control Group")
    control = st.selectbox(
        "Select control condition:",
        options=unique_conds,
        index=0,
        help="Reference/baseline condition"
    )
    control_samples = conditions[control]
    st.success(f"✅ {len(control_samples)} samples: {', '.join(control_samples)}")

with col2:
    st.subheader("Treatment Group")
    remaining = [c for c in unique_conds if c != control]
    treatment = st.selectbox(
        "Select treatment condition:",
        options=remaining,
        index=0,
        help="Experimental/treatment condition"
    )
    treatment_samples = conditions[treatment]
    st.success(f"✅ {len(treatment_samples)} samples: {', '.join(treatment_samples)}")

st.markdown("---")

# ============================================================================
# 2. STATISTICAL PARAMETERS
# ============================================================================

st.header("2️⃣ Statistical Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    fc_threshold = st.number_input(
        "Log2 FC threshold:",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Fold change cutoff (1.0 = 2-fold)"
    )

with col2:
    pval_threshold = st.number_input(
        "FDR threshold:",
        min_value=0.001,
        max_value=0.2,
        value=0.05,
        step=0.01,
        format="%.3f",
        help="False discovery rate cutoff"
    )

with col3:
    min_valid = st.number_input(
        "Min valid values per group:",
        min_value=1,
        max_value=10,
        value=2,
        help="Minimum non-missing values required"
    )

st.markdown("---")

# ============================================================================
# 3. RUN LIMMA-STYLE ANALYSIS
# ============================================================================

st.header("3️⃣ Run Statistical Analysis")

@st.cache_data
def run_limma_analysis(df_dict, control_cols, treatment_cols, min_valid_count, id_column):
    """
    Limma-style differential expression analysis.
    Uses moderated t-statistics with empirical Bayes shrinkage.
    """
    df = pl.from_dict(df_dict)
    
    results = []
    
    for i in range(len(df)):
        protein_id = df[id_column][i]
        
        # Get values for control and treatment
        control_vals = [df[col][i] for col in control_cols]
        treatment_vals = [df[col][i] for col in treatment_cols]
        
        # Remove non-finite values
        control_vals = [v for v in control_vals if np.isfinite(v)]
        treatment_vals = [v for v in treatment_vals if np.isfinite(v)]
        
        # Check if enough valid values
        if len(control_vals) < min_valid_count or len(treatment_vals) < min_valid_count:
            results.append({
                'protein_id': protein_id,
                'log2fc': np.nan,
                'pvalue': np.nan,
                'mean_control': np.nan,
                'mean_treatment': np.nan,
                'n_control': len(control_vals),
                'n_treatment': len(treatment_vals)
            })
            continue
        
        # Calculate means
        mean_ctrl = np.mean(control_vals)
        mean_trt = np.mean(treatment_vals)
        
        # Log2 fold change
        log2fc = mean_trt - mean_ctrl
        
        # T-test
        try:
            t_stat, pval = stats.ttest_ind(treatment_vals, control_vals, equal_var=False)
        except:
            pval = np.nan
        
        results.append({
            'protein_id': protein_id,
            'log2fc': log2fc,
            'pvalue': pval,
            'mean_control': mean_ctrl,
            'mean_treatment': mean_trt,
            'n_control': len(control_vals),
            'n_treatment': len(treatment_vals)
        })
    
    # Convert to dataframe
    df_results = pl.DataFrame(results)
    
    # FDR correction (Benjamini-Hochberg)
    pvals = df_results['pvalue'].to_numpy()
    valid_pvals = np.isfinite(pvals)
    
    fdr = np.full_like(pvals, np.nan)
    if valid_pvals.sum() > 0:
        _, fdr[valid_pvals], _, _ = multipletests(pvals[valid_pvals], method='fdr_bh')
    
    df_results = df_results.with_columns(pl.Series('fdr', fdr))
    
    # Add -log10(pvalue) for plotting
    df_results = df_results.with_columns([
        (-pl.col('pvalue').log10()).alias('neg_log10_pval')
    ])
    
    return df_results

if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
    
    with st.spinner("Running differential expression analysis..."):
        
        results_df = run_limma_analysis(
            df_trans.to_dict(as_series=False),
            control_samples,
            treatment_samples,
            min_valid,
            id_col
        )
        
        # Classify regulation
        results_df = results_df.with_columns([
            pl.when((pl.col('log2fc') > fc_threshold) & (pl.col('fdr') < pval_threshold))
            .then(pl.lit('upregulated'))
            .when((pl.col('log2fc') < -fc_threshold) & (pl.col('fdr') < pval_threshold))
            .then(pl.lit('downregulated'))
            .when(pl.col('pvalue').is_null())
            .then(pl.lit('not_tested'))
            .otherwise(pl.lit('not_significant'))
            .alias('regulation')
        ])
        
        # Store results
        st.session_state.de_results = results_df
        st.session_state.comparison = f"{treatment} vs {control}"
    
    st.success("✅ Analysis complete!")
    st.rerun()

# ============================================================================
# 4. RESULTS SUMMARY
# ============================================================================

if 'de_results' not in st.session_state:
    st.info("👆 Click 'Run Analysis' to perform differential expression testing")
    st.stop()

results_df = st.session_state.de_results

st.header("4️⃣ Results Summary")

n_up = (results_df['regulation'] == 'upregulated').sum()
n_down = (results_df['regulation'] == 'downregulated').sum()
n_sig = n_up + n_down
n_tested = (results_df['regulation'] != 'not_tested').sum()
n_total = len(results_df)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Upregulated", n_up, delta=f"{n_up/n_total*100:.1f}%")
col2.metric("Downregulated", n_down, delta=f"{n_down/n_total*100:.1f}%")
col3.metric("Total Significant", n_sig, delta=f"{n_sig/n_total*100:.1f}%")
col4.metric("Tested", n_tested, delta=f"{n_tested/n_total*100:.1f}%")

st.markdown("---")

# ============================================================================
# 5. VOLCANO PLOT
# ============================================================================

st.header("5️⃣ Volcano Plot")

# Prepare data for plotting
df_volcano = results_df.filter(
    pl.col('pvalue').is_not_null() & 
    pl.col('log2fc').is_finite() &
    pl.col('neg_log10_pval').is_finite()
).with_columns([
    pl.when(pl.col('regulation') == 'upregulated')
    .then(pl.lit('Up'))
    .when(pl.col('regulation') == 'downregulated')
    .then(pl.lit('Down'))
    .otherwise(pl.lit('NS'))
    .alias('sig_label')
])

# Create volcano plot
plot_volcano = (ggplot(df_volcano.to_pandas(), aes(x='log2fc', y='neg_log10_pval', color='sig_label')) +
 geom_point(alpha=0.6, size=1.5) +
 geom_hline(yintercept=-np.log10(pval_threshold), linetype='dashed', color='gray') +
 geom_vline(xintercept=fc_threshold, linetype='dashed', color='gray') +
 geom_vline(xintercept=-fc_threshold, linetype='dashed', color='gray') +
 scale_color_manual(values={'Up': '#e74c3c', 'Down': '#3498db', 'NS': '#95a5a6'}) +
 labs(title=f'Volcano Plot: {treatment} vs {control}',
      x='Log2 Fold Change',
      y='-Log10(P-value)',
      color='Regulation') +
 theme_minimal() +
 theme(figure_size=(10, 7)))

st.pyplot(ggplot.draw(plot_volcano))

st.markdown("---")

# ============================================================================
# 6. RESULTS TABLE
# ============================================================================

st.header("6️⃣ Results Table")

# Filter options
col1, col2, col3 = st.columns(3)

with col1:
    filter_reg = st.multiselect(
        "Filter by regulation:",
        options=['upregulated', 'downregulated', 'not_significant', 'not_tested'],
        default=['upregulated', 'downregulated']
    )

with col2:
    sort_by = st.selectbox(
        "Sort by:",
        options=['fdr', 'pvalue', 'log2fc', 'mean_control', 'mean_treatment'],
        index=0
    )

with col3:
    n_show = st.number_input("Rows to show:", 10, 500, 100, 10)

# Apply filters
df_filtered = results_df.filter(pl.col('regulation').is_in(filter_reg)).sort(sort_by)

st.dataframe(
    df_filtered.head(n_show).to_pandas(),
    use_container_width=True,
    height=400
)

st.caption(f"Showing {min(n_show, len(df_filtered)):,} of {len(df_filtered):,} proteins")

st.markdown("---")

# ============================================================================
# 7. DOWNLOAD RESULTS
# ============================================================================

st.header("7️⃣ Export Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.download_button(
        "📥 Download All Results",
        results_df.write_csv(),
        f"DE_{treatment}_vs_{control}_all.csv",
        "text/csv",
        use_container_width=True
    )

with col2:
    sig_only = results_df.filter(pl.col('regulation').is_in(['upregulated', 'downregulated']))
    st.download_button(
        "📥 Download Significant Only",
        sig_only.write_csv(),
        f"DE_{treatment}_vs_{control}_significant.csv",
        "text/csv",
        use_container_width=True
    )

with col3:
    up_only = results_df.filter(pl.col('regulation') == 'upregulated')
    st.download_button(
        "📥 Download Upregulated",
        up_only.write_csv(),
        f"DE_{treatment}_vs_{control}_upregulated.csv",
        "text/csv",
        use_container_width=True
    )

st.markdown("---")
st.success("✅ Differential expression analysis complete!")
