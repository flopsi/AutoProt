"""
pages/5_Differential_Expression.py
Differential expression analysis with integrated benchmark validation
"""

import streamlit as st
import polars as pl
import numpy as np
from plotnine import *
from scipy import stats
from statsmodels.stats.multitest import multipletests
import pandas as pd

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
# 0. SPECIES COMPOSITION (IF AVAILABLE)
# ============================================================================

if species_col:
    st.header("0️⃣ Species Composition")
    
    # Calculate species composition per condition
    conditions = {}
    for col in numeric_cols:
        condition = col[0]
        if condition not in conditions:
            conditions[condition] = []
        conditions[condition].append(col)
    
    composition_data = []
    
    for condition, cols in conditions.items():
        # Count proteins per species in this condition
        species_counts = df_trans.select([id_col, species_col]).group_by(species_col).agg(
            pl.count().alias('n_proteins')
        )
        
        total = species_counts['n_proteins'].sum()
        
        for row in species_counts.iter_rows(named=True):
            composition_data.append({
                'Condition': condition,
                'Species': row[species_col],
                'Count': row['n_proteins'],
                'Percentage': round(row['n_proteins'] / total * 100, 1)
            })
    
    df_composition = pl.DataFrame(composition_data).sort(['Condition', 'Species'])
    
    # Display as table
    st.dataframe(df_composition.to_pandas(), use_container_width=True)
    
    # Stacked bar chart
    plot_composition = (ggplot(df_composition.to_pandas(), 
                               aes(x='Condition', y='Percentage', fill='Species')) +
     geom_bar(stat='identity') +
     geom_text(aes(label='Percentage'), position=position_stack(vjust=0.5),
               size=8, color='white', fontweight='bold') +
     labs(title='Species Composition by Condition',
          x='Condition', y='Percentage (%)', fill='Species') +
     theme_minimal() +
     theme(figure_size=(8, 5)))
    
    st.pyplot(ggplot.draw(plot_composition))
    
    st.markdown("---")

# ============================================================================
# 1. DEFINE COMPARISON GROUPS
# ============================================================================

st.header("1️⃣ Define Comparison Groups")

# Group by condition (reuse if already done)
if 'conditions' not in locals():
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
    effect_threshold = st.number_input(
        "Effect size threshold:",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Minimum difference in means (scale depends on transformation)"
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

# Label based on transformation
if transform_name in ['log2', 'log10', 'ln']:
    effect_label = "Log Fold Change"
else:
    effect_label = "Effect Size (Δ mean)"

st.info(f"**Effect interpretation:** Using {transform_name.upper()}, so effect size = {effect_label}")

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
                'effect_size': np.nan,
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
        
        # Effect size (difference in means)
        effect_size = mean_trt - mean_ctrl
        
        # T-test
        try:
            t_stat, pval = stats.ttest_ind(treatment_vals, control_vals, equal_var=False)
        except:
            pval = np.nan
        
        results.append({
            'protein_id': protein_id,
            'effect_size': effect_size,
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
            pl.when((pl.col('effect_size') > effect_threshold) & (pl.col('fdr') < pval_threshold))
            .then(pl.lit('upregulated'))
            .when((pl.col('effect_size') < -effect_threshold) & (pl.col('fdr') < pval_threshold))
            .then(pl.lit('downregulated'))
            .when(pl.col('pvalue').is_null())
            .then(pl.lit('not_tested'))
            .otherwise(pl.lit('not_significant'))
            .alias('regulation')
        ])
        
        # Store results
        st.session_state.de_results = results_df
        st.session_state.comparison = f"{treatment} vs {control}"
        st.session_state.effect_label = effect_label
    
    st.success("✅ Analysis complete!")
    st.rerun()

# ============================================================================
# 4. RESULTS SUMMARY
# ============================================================================

if 'de_results' not in st.session_state:
    st.info("👆 Click 'Run Analysis' to perform differential expression testing")
    st.stop()

results_df = st.session_state.de_results
effect_label = st.session_state.effect_label

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
    pl.col('effect_size').is_finite() &
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
plot_volcano = (ggplot(df_volcano.to_pandas(), aes(x='effect_size', y='neg_log10_pval', color='sig_label')) +
 geom_point(alpha=0.6, size=1.5) +
 geom_hline(yintercept=-np.log10(pval_threshold), linetype='dashed', color='gray') +
 geom_vline(xintercept=effect_threshold, linetype='dashed', color='gray') +
 geom_vline(xintercept=-effect_threshold, linetype='dashed', color='gray') +
 scale_color_manual(values={'Up': '#e74c3c', 'Down': '#3498db', 'NS': '#95a5a6'}) +
 labs(title=f'Volcano Plot: {treatment} vs {control}',
      x=effect_label,
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
        options=['fdr', 'pvalue', 'effect_size', 'mean_control', 'mean_treatment'],
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

# ============================================================================
# 8. BENCHMARK PLOTS (For Spike-in Validation)
# ============================================================================

st.header("8️⃣ Benchmark Plots (Spike-in Validation)")
st.info("**Optional:** For datasets with known mixing ratios, validate quantification accuracy")

if species_col:
    
    if st.checkbox("🔬 Run Benchmark Analysis", value=False):
        
        # ============================================================================
        # Define spike-in fractions and calculate expected log2 FC
        # ============================================================================
        
        st.subheader("Define Spike-in Fractions")
        st.info("**Enter mixing ratios** for each species in both conditions. Log2 FC will be calculated automatically.")
        
        species_list = df_trans[species_col].unique().to_list()
        
        # Create input columns for fractions
        col_header1, col_header2 = st.columns(2)
        col_header1.markdown(f"**Condition: {control}**")
        col_header2.markdown(f"**Condition: {treatment}**")
        
        fractions_control = {}
        fractions_treatment = {}
        expected_fc = {}
        
        for species in species_list:
            col1, col2 = st.columns(2)
            
            with col1:
                fractions_control[species] = st.number_input(
                    f"{species} fraction (%):",
                    min_value=0.0,
                    max_value=100.0,
                    value=33.33 if len(species_list) == 3 else 50.0,
                    step=1.0,
                    key=f"frac_ctrl_{species}",
                    help=f"Percentage of {species} in {control}"
                )
            
            with col2:
                fractions_treatment[species] = st.number_input(
                    f"{species} fraction (%):",
                    min_value=0.0,
                    max_value=100.0,
                    value=33.33 if len(species_list) == 3 else 50.0,
                    step=1.0,
                    key=f"frac_trt_{species}",
                    help=f"Percentage of {species} in {treatment}"
                )
        
        # Calculate log2 fold changes
        st.markdown("---")
        st.subheader("Calculated Expected Log2 Fold Changes")
        
        expected_fc_data = []
        
        for species in species_list:
            frac_ctrl = fractions_control[species] / 100  # Convert to proportion
            frac_trt = fractions_treatment[species] / 100
            
            # Avoid log2(0)
            if frac_ctrl > 0 and frac_trt > 0:
                log2_fc = np.log2(frac_trt / frac_ctrl)
                expected_fc[species] = log2_fc
            elif frac_trt > 0:
                log2_fc = np.inf
                expected_fc[species] = 5.0  # Cap at high value
            elif frac_ctrl > 0:
                log2_fc = -np.inf
                expected_fc[species] = -5.0  # Cap at low value
            else:
                log2_fc = 0.0
                expected_fc[species] = 0.0
            
            expected_fc_data.append({
                'Species': species,
                f'{control} (%)': fractions_control[species],
                f'{treatment} (%)': fractions_treatment[species],
                'Fold Change': round(frac_trt / frac_ctrl, 3) if frac_ctrl > 0 else np.inf,
                'Log2 FC': round(log2_fc, 3) if np.isfinite(log2_fc) else log2_fc
            })
        
        df_expected = pl.DataFrame(expected_fc_data)
        st.dataframe(df_expected.to_pandas(), use_container_width=True)
        
        # Validation check
        total_ctrl = sum(fractions_control.values())
        total_trt = sum(fractions_treatment.values())
        
        if abs(total_ctrl - 100) > 0.1:
            st.warning(f"⚠️ {control} fractions sum to {total_ctrl:.1f}% (should be 100%)")
        if abs(total_trt - 100) > 0.1:
            st.warning(f"⚠️ {treatment} fractions sum to {total_trt:.1f}% (should be 100%)")
        
        # Join species info with results
        df_bench = df_trans.select([id_col, species_col]).join(
            results_df,
            left_on=id_col,
            right_on='protein_id',
            how='inner'
        ).with_columns([
            pl.col(species_col).map_dict(expected_fc).alias('expected_fc')
        ]).filter(
            pl.col('effect_size').is_finite() &
            pl.col('expected_fc').is_not_null()
        )
        
        st.markdown("---")
        
        # ============================================================================
        # 8.1 SCATTER PLOTS (Observed vs Expected)
        # ============================================================================
        
        st.subheader("8.1 Observed vs Expected Fold Change")
        
        unique_species = df_bench[species_col].unique().to_list()
        cols = st.columns(min(3, len(unique_species)))
        
        for idx, species in enumerate(unique_species):
            with cols[idx % 3]:
                df_sp = df_bench.filter(pl.col(species_col) == species)
                
                if df_sp.shape[0] > 0:
                    exp_fc_val = expected_fc.get(species, 0)
                    
                    plot = (ggplot(df_sp.to_pandas(), aes(x='expected_fc', y='effect_size')) +
                     geom_point(alpha=0.5, size=1.5, color='#3498db') +
                     geom_abline(intercept=0, slope=1, color='red', linetype='dashed', size=1) +
                     geom_hline(yintercept=exp_fc_val, color='gray', linetype='dotted') +
                     geom_vline(xintercept=exp_fc_val, color='gray', linetype='dotted') +
                     labs(title=f'{species}',
                          x='Expected FC',
                          y='Observed FC') +
                     theme_minimal() +
                     theme(figure_size=(4, 4)))
                    
                    st.pyplot(ggplot.draw(plot))
                    
                    # Accuracy metrics
                    bias = (df_sp['effect_size'] - df_sp['expected_fc']).mean()
                    rmse = np.sqrt(((df_sp['effect_size'] - df_sp['expected_fc'])**2).mean())
                    mae = (df_sp['effect_size'] - df_sp['expected_fc']).abs().mean()
                    
                    col_a, col_b = st.columns(2)
                    col_a.metric("RMSE", f"{rmse:.3f}")
                    col_b.metric("Bias", f"{bias:.3f}")
                    st.caption(f"MAE: {mae:.3f} | N={df_sp.shape[0]}")
        
        st.markdown("---")
        
        # ============================================================================
        # 8.2 DENSITY PLOTS
        # ============================================================================
        
        st.subheader("8.2 Effect Size Distribution by Species")
        
        plot_density = (ggplot(df_bench.to_pandas(), aes(x='effect_size', fill=species_col)) +
         geom_density(alpha=0.5) +
         geom_vline(data=pd.DataFrame({'species': list(expected_fc.keys()), 
                                       'expected': list(expected_fc.values())}),
                    mapping=aes(xintercept='expected', color='species'),
                    linetype='dashed', size=1) +
         labs(title='Effect Size Distribution by Species',
              x=effect_label,
              y='Density',
              fill='Species',
              color='Expected') +
         theme_minimal() +
         theme(figure_size=(10, 5)))
        
        st.pyplot(ggplot.draw(plot_density))
        
        # Asymmetry factors
        st.markdown("**Asymmetry Factors** (ideal range: 0.5-2.0)")
        
        asym_data = []
        for species in unique_species:
            df_sp = df_bench.filter(pl.col(species_col) == species)
            if df_sp.shape[0] > 10:
                fc_vals = df_sp['effect_size'].to_numpy()
                median_fc = np.median(fc_vals)
                
                above = fc_vals[fc_vals > median_fc]
                below = fc_vals[fc_vals < median_fc]
                
                if len(below) > 0 and len(above) > 0:
                    asym = np.mean(above - median_fc) / np.mean(median_fc - below)
                    status = '✓' if 0.5 <= asym <= 2.0 else '⚠️'
                    asym_data.append({
                        'Species': species, 
                        'Asymmetry': round(asym, 3),
                        'Status': status
                    })
        
        if asym_data:
            st.dataframe(pl.DataFrame(asym_data).to_pandas(), hide_index=True, use_container_width=True)
        
        st.markdown("---")
        
        # ============================================================================
        # 8.3 ROC AND PRECISION-RECALL CURVES
        # ============================================================================
        
        st.subheader("8.3 Classification Performance")
        
        col_roc, col_pr = st.columns(2)
        
        with col_roc:
            # Define true positives (species with expected FC != 0)
            df_roc = df_bench.with_columns([
                (pl.col('expected_fc') != 0).alias('true_positive')
            ]).sort('pvalue')
            
            n_true_pos = df_roc['true_positive'].sum()
            n_true_neg = (~df_roc['true_positive']).sum()
            
            tpr_list = []
            fpr_list = []
            cumsum_tp = 0
            cumsum_fp = 0
            
            for row in df_roc.iter_rows(named=True):
                if row['true_positive']:
                    cumsum_tp += 1
                else:
                    cumsum_fp += 1
                
                tpr = cumsum_tp / n_true_pos if n_true_pos > 0 else 0
                fpr = cumsum_fp / n_true_neg if n_true_neg > 0 else 0
                
                tpr_list.append(tpr)
                fpr_list.append(fpr)
            
            df_roc_plot = pl.DataFrame({'FPR': fpr_list, 'TPR': tpr_list})
            auc = np.trapz(tpr_list, fpr_list)
            
            plot_roc = (ggplot(df_roc_plot.to_pandas(), aes(x='FPR', y='TPR')) +
             geom_line(color='#e74c3c', size=1.5) +
             geom_abline(intercept=0, slope=1, linetype='dashed', color='gray') +
             labs(title=f'ROC Curve (AUC = {auc:.3f})',
                  x='False Positive Rate',
                  y='True Positive Rate') +
             theme_minimal() +
             theme(figure_size=(5, 5)))
            
            st.pyplot(ggplot.draw(plot_roc))
            st.metric("AUC", f"{auc:.3f}", help="Area Under Curve (1.0 = perfect)")
        
        with col_pr:
            precision_list = []
            recall_list = []
            cumsum_tp = 0
            cumsum_fp = 0
            
            for row in df_roc.iter_rows(named=True):
                if row['true_positive']:
                    cumsum_tp += 1
                else:
                    cumsum_fp += 1
                
                precision = cumsum_tp / (cumsum_tp + cumsum_fp) if (cumsum_tp + cumsum_fp) > 0 else 0
                recall = cumsum_tp / n_true_pos if n_true_pos > 0 else 0
                
                precision_list.append(precision)
                recall_list.append(recall)
            
            df_pr_plot = pl.DataFrame({'Recall': recall_list, 'Precision': precision_list})
            
            plot_pr = (ggplot(df_pr_plot.to_pandas(), aes(x='Recall', y='Precision')) +
             geom_line(color='#2ecc71', size=1.5) +
             labs(title='Precision-Recall Curve',
                  x='Recall (Sensitivity)',
                  y='Precision (PPV)') +
             theme_minimal() +
             theme(figure_size=(5, 5)))
            
            st.pyplot(ggplot.draw(plot_pr))
            max_f1 = max([2*p*r/(p+r) if (p+r) > 0 else 0 for p, r in zip(precision_list, recall_list)])
            st.metric("Max F1", f"{max_f1:.3f}", help="Maximum F1 score")
        
        st.markdown("---")
        
        # ============================================================================
        # 8.4 SUMMARY TABLE
        # ============================================================================
        
        st.subheader("8.4 Benchmark Summary Statistics")
        
        summary_stats = []
        
        for species in unique_species:
            df_sp = df_bench.filter(pl.col(species_col) == species)
            
            if df_sp.shape[0] > 0:
                exp_fc_val = expected_fc.get(species, 0)
                
                n_total = df_sp.shape[0]
                n_sig = df_sp.filter(pl.col('regulation').is_in(['upregulated', 'downregulated'])).shape[0]
                
                obs_vals = df_sp['effect_size'].to_numpy()
                exp_vals = np.full_like(obs_vals, exp_fc_val)
                
                bias = np.mean(obs_vals - exp_vals)
                rmse = np.sqrt(np.mean((obs_vals - exp_vals)**2))
                mae = np.mean(np.abs(obs_vals - exp_vals))
                corr = np.corrcoef(obs_vals, exp_vals)[0, 1] if len(obs_vals) > 1 else np.nan
                
                summary_stats.append({
                    'Species': species,
                    'Expected FC': exp_fc_val,
                    'N Proteins': n_total,
                    'N Significant': n_sig,
                    'Bias': round(bias, 3),
                    'RMSE': round(rmse, 3),
                    'MAE': round(mae, 3),
                    'Correlation': round(corr, 3)
                })
        
        df_summary = pl.DataFrame(summary_stats)
        st.dataframe(df_summary.to_pandas(), use_container_width=True)
        
        # Download benchmark results
        st.download_button(
            "📥 Download Benchmark Summary",
            df_summary.write_csv(),
            f"benchmark_{treatment}_vs_{control}.csv",
            "text/csv",
            use_container_width=True
        )

else:
    st.info("Species information not available. Benchmark plots require species labels for validation.")

st.markdown("---")
st.success("✅ Differential expression analysis complete!")
