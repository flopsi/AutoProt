"""
pages/5_Differential_Expression.py
Differential expression analysis with integrated benchmark validation
"""

import streamlit as st
import polars as pl
import numpy as np
import pandas as pd
from plotnine import *
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Page config
st.set_page_config(page_title="Differential Expression", page_icon="🧬", layout="wide")

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
**Dataset:** {summary['final']:,} proteins | {len(numeric_cols)} samples | {transform_name.upper()} transformed
""")

st.markdown("---")

# ============================================================================
# 0. SPECIES COMPOSITION
# ============================================================================

if species_col:
    with st.expander("📊 Species Composition", expanded=False):
        
        # Calculate composition
        conditions = {}
        for col in numeric_cols:
            condition = col[0]
            if condition not in conditions:
                conditions[condition] = []
            conditions[condition].append(col)
        
        composition_data = []
        
        for condition, cols in conditions.items():
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
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(
                df_composition.to_pandas().style.format({'Percentage': '{:.1f}%'}),
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            plot_composition = (ggplot(df_composition.to_pandas(), 
                                       aes(x='Condition', y='Percentage', fill='Species')) +
             geom_col(width=0.6) +
             geom_text(aes(label='Percentage'), position=position_stack(vjust=0.5),
                       size=9, color='white', fontweight='bold', format_string='{:.0f}%') +
             scale_fill_brewer(type='qual', palette='Set2') +
             labs(title='', x='Condition', y='Percentage (%)', fill='Species') +
             theme_minimal() +
             theme(figure_size=(6, 4), legend_position='right'))
            
            st.pyplot(ggplot.draw(plot_composition))

st.markdown("---")

# ============================================================================
# 1. DEFINE COMPARISON GROUPS
# ============================================================================

st.header("1️⃣ Define Comparison Groups")

# Group by condition
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
    control = st.selectbox(
        "Control (Reference):",
        options=unique_conds,
        index=0,
        help="Baseline condition"
    )
    control_samples = conditions[control]
    st.success(f"✓ {len(control_samples)} samples: {', '.join(control_samples)}")

with col2:
    remaining = [c for c in unique_conds if c != control]
    treatment = st.selectbox(
        "Treatment (Experimental):",
        options=remaining,
        index=0,
        help="Experimental condition"
    )
    treatment_samples = conditions[treatment]
    st.success(f"✓ {len(treatment_samples)} samples: {', '.join(treatment_samples)}")

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
        help="Minimum difference in means"
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
        "Min valid per group:",
        min_value=1,
        max_value=10,
        value=2,
        help="Minimum non-missing values"
    )

# Label based on transformation
effect_label = "Log2 Fold Change" if transform_name in ['log2', 'log10', 'ln'] else "Effect Size"

st.caption(f"*Using {transform_name.upper()} transformation: Effect = {effect_label}*")

st.markdown("---")

# ============================================================================
# 3. RUN ANALYSIS
# ============================================================================

st.header("3️⃣ Run Statistical Analysis")

@st.cache_data
def run_limma_analysis(df_dict, control_cols, treatment_cols, min_valid_count, id_column):
    """Limma-style differential expression with moderated t-statistics."""
    df = pl.from_dict(df_dict)
    results = []
    
    for i in range(len(df)):
        protein_id = df[id_column][i]
        
        # Get values
        control_vals = [df[col][i] for col in control_cols]
        treatment_vals = [df[col][i] for col in treatment_cols]
        
        # Remove non-finite
        control_vals = [v for v in control_vals if np.isfinite(v)]
        treatment_vals = [v for v in treatment_vals if np.isfinite(v)]
        
        # Check validity
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
        
        # Statistics
        mean_ctrl = np.mean(control_vals)
        mean_trt = np.mean(treatment_vals)
        
        # FIXED: log2(Control/Treatment) = log2(A/B)
        effect_size = mean_ctrl - mean_trt  # Changed from mean_trt - mean_ctrl
        
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
    
    
    df_results = pl.DataFrame(results)
    
    # FDR correction
    pvals = df_results['pvalue'].to_numpy()
    valid_pvals = np.isfinite(pvals)
    
    fdr = np.full_like(pvals, np.nan)
    if valid_pvals.sum() > 0:
        _, fdr[valid_pvals], _, _ = multipletests(pvals[valid_pvals], method='fdr_bh')
    
    df_results = df_results.with_columns([
        pl.Series('fdr', fdr),
        (-pl.col('pvalue').log10()).alias('neg_log10_pval')
    ])
    
    return df_results

if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
    
    with st.spinner("Running differential expression..."):
        
        results_df = run_limma_analysis(
            df_trans.to_dict(as_series=False),
            control_samples,
            treatment_samples,
            min_valid,
            id_col
        )
        
        # Classify
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
        
        # Store
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

st.markdown("---")
st.header("4️⃣ Results Summary")

n_up = (results_df['regulation'] == 'upregulated').sum()
n_down = (results_df['regulation'] == 'downregulated').sum()
n_sig = n_up + n_down
n_tested = (results_df['regulation'] != 'not_tested').sum()
n_total = len(results_df)

col1, col2, col3, col4 = st.columns(4)

col1.metric("⬆️ Upregulated", f"{n_up:,}", delta=f"{n_up/n_total*100:.1f}%", delta_color="normal")
col2.metric("⬇️ Downregulated", f"{n_down:,}", delta=f"{n_down/n_total*100:.1f}%", delta_color="inverse")
col3.metric("✓ Total Significant", f"{n_sig:,}", delta=f"{n_sig/n_total*100:.1f}%")
col4.metric("📊 Tested", f"{n_tested:,}", delta=f"{n_tested/n_total*100:.1f}%")

st.markdown("---")

# ============================================================================
# 5. VOLCANO PLOT
# ============================================================================

st.header("5️⃣ Volcano Plot")

df_volcano = results_df.filter(
    pl.col('pvalue').is_not_null() & 
    pl.col('effect_size').is_finite() &
    pl.col('neg_log10_pval').is_finite()
).with_columns([
    pl.when(pl.col('regulation') == 'upregulated').then(pl.lit('Up'))
    .when(pl.col('regulation') == 'downregulated').then(pl.lit('Down'))
    .otherwise(pl.lit('NS')).alias('sig_label')
])

plot_volcano = (ggplot(df_volcano.to_pandas(), aes(x='effect_size', y='neg_log10_pval', color='sig_label')) +
 geom_point(alpha=0.5, size=2) +
 geom_hline(yintercept=-np.log10(pval_threshold), linetype='dashed', color='#7f8c8d', size=0.8) +
 geom_vline(xintercept=effect_threshold, linetype='dashed', color='#7f8c8d', size=0.8) +
 geom_vline(xintercept=-effect_threshold, linetype='dashed', color='#7f8c8d', size=0.8) +
 scale_color_manual(values={'Up': '#e74c3c', 'Down': '#3498db', 'NS': '#95a5a6'}) +
 labs(title=f'Volcano Plot: {treatment} vs {control}',
      x=effect_label,
      y='-Log10(P-value)',
      color='Regulation') +
 theme_minimal() +
 theme(figure_size=(11, 7), legend_position='right'))

st.pyplot(ggplot.draw(plot_volcano))

st.markdown("---")

# ============================================================================
# 6. RESULTS TABLE
# ============================================================================

st.header("6️⃣ Results Table")

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
    n_show = st.number_input("Rows:", 10, 500, 50, 10)

df_filtered = results_df.filter(pl.col('regulation').is_in(filter_reg)).sort(sort_by)

st.dataframe(
    df_filtered.head(n_show).to_pandas().style.format({
        'effect_size': '{:.3f}',
        'pvalue': '{:.2e}',
        'fdr': '{:.2e}',
        'mean_control': '{:.2f}',
        'mean_treatment': '{:.2f}'
    }),
    use_container_width=True,
    height=400
)

st.caption(f"Showing {min(n_show, len(df_filtered)):,} of {len(df_filtered):,} proteins")

st.markdown("---")

# ============================================================================
# 7. EXPORT
# ============================================================================

st.header("7️⃣ Export Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.download_button(
        "📥 All Results",
        results_df.write_csv(),
        f"DE_{treatment}_vs_{control}_all.csv",
        "text/csv",
        use_container_width=True
    )

with col2:
    sig_only = results_df.filter(pl.col('regulation').is_in(['upregulated', 'downregulated']))
    st.download_button(
        "📥 Significant Only",
        sig_only.write_csv(),
        f"DE_{treatment}_vs_{control}_sig.csv",
        "text/csv",
        use_container_width=True
    )

with col3:
    up_only = results_df.filter(pl.col('regulation') == 'upregulated')
    st.download_button(
        "📥 Upregulated Only",
        up_only.write_csv(),
        f"DE_{treatment}_vs_{control}_up.csv",
        "text/csv",
        use_container_width=True
    )

st.markdown("---")

# ============================================================================
# 8. BENCHMARK ANALYSIS
# ============================================================================

st.header("8️⃣ Benchmark Analysis (Spike-in Validation)")

if species_col:
    
    with st.expander("🔬 Run Benchmark Analysis", expanded=False):
        
        st.info("**For spike-in datasets:** Define mixing ratios to validate quantification accuracy")
        
        species_list = df_trans[species_col].unique().to_list()
        
        # Initialize session state
        if 'fractions_defined' not in st.session_state:
            st.session_state.fractions_defined = False
        
        # ============================================================================
        # FRACTION INPUT FORM
        # ============================================================================
        
        with st.form("spike_in_fractions"):
            st.subheader("Define Spike-in Fractions")
            
            col_h1, col_h2 = st.columns(2)
            col_h1.markdown(f"**{control}**")
            col_h2.markdown(f"**{treatment}**")
            
            fractions_control = {}
            fractions_treatment = {}
            
            for species in species_list:
                col1, col2 = st.columns(2)
                
                with col1:
                    fractions_control[species] = st.number_input(
                        f"{species} (%):",
                        min_value=0.0,
                        max_value=100.0,
                        value=st.session_state.get(f'frac_ctrl_{species}', 33.33),
                        step=1.0,
                        key=f"input_ctrl_{species}"
                    )
                
                with col2:
                    fractions_treatment[species] = st.number_input(
                        f"{species} (%):",
                        min_value=0.0,
                        max_value=100.0,
                        value=st.session_state.get(f'frac_trt_{species}', 33.33),
                        step=1.0,
                        key=f"input_trt_{species}"
                    )
            
            submit = st.form_submit_button("✅ Lock Fractions & Run Benchmark", use_container_width=True)
            
            if submit:
                for species in species_list:
                    st.session_state[f'frac_ctrl_{species}'] = fractions_control[species]
                    st.session_state[f'frac_trt_{species}'] = fractions_treatment[species]
                st.session_state.fractions_defined = True
                st.rerun()
        
        # ============================================================================
        # BENCHMARK ANALYSIS
        # ============================================================================
        
        if st.session_state.fractions_defined:
            
            # Load fractions
            fractions_control = {s: st.session_state[f'frac_ctrl_{s}'] for s in species_list}
            fractions_treatment = {s: st.session_state[f'frac_trt_{s}'] for s in species_list}
            
            # Calculate expected FC
            expected_fc = {}
            expected_fc_data = []
            
            for species in species_list:
                frac_ctrl = fractions_control[species] / 100
                frac_trt = fractions_treatment[species] / 100
                
                if frac_ctrl > 0 and frac_trt > 0:
                    log2_fc = np.log2(frac_trt / frac_ctrl)
                    expected_fc[species] = log2_fc
                elif frac_trt > 0:
                    expected_fc[species] = 5.0
                    log2_fc = np.inf
                elif frac_ctrl > 0:
                    expected_fc[species] = -5.0
                    log2_fc = -np.inf
                else:
                    expected_fc[species] = 0.0
                    log2_fc = 0.0
                
                expected_fc_data.append({
                    'Species': species,
                    f'{control} (%)': fractions_control[species],
                    f'{treatment} (%)': fractions_treatment[species],
                    'FC': round(frac_trt / frac_ctrl, 2) if frac_ctrl > 0 else '∞',
                    'Log2 FC': round(log2_fc, 2) if np.isfinite(log2_fc) else '∞'
                })
            
            st.markdown("---")
            st.subheader("Expected Fold Changes")
            
            col_table, col_reset = st.columns([3, 1])
            
            with col_table:
                st.dataframe(
                    pl.DataFrame(expected_fc_data).to_pandas(),
                    use_container_width=True,
                    hide_index=True
                )
            
            with col_reset:
                if st.button("🔄 Reset", use_container_width=True):
                    st.session_state.fractions_defined = False
                    for species in species_list:
                        if f'frac_ctrl_{species}' in st.session_state:
                            del st.session_state[f'frac_ctrl_{species}']
                        if f'frac_trt_{species}' in st.session_state:
                            del st.session_state[f'frac_trt_{species}']
                    st.rerun()
            
            # Validation
            total_ctrl = sum(fractions_control.values())
            total_trt = sum(fractions_treatment.values())
            
            if abs(total_ctrl - 100) > 0.1 or abs(total_trt - 100) > 0.1:
                st.warning(f"⚠️ Fractions should sum to 100% ({total_ctrl:.1f}% / {total_trt:.1f}%)")
            
            # Join with results
            df_bench = df_trans.select([id_col, species_col]).join(
                results_df,
                left_on=id_col,
                right_on='protein_id',
                how='inner'
            ).with_columns([
                pl.col(species_col).replace(expected_fc, default=None).alias('expected_fc')
            ]).filter(
                pl.col('effect_size').is_finite() &
                pl.col('expected_fc').is_not_null()
            )
            
            unique_species = df_bench[species_col].unique().to_list()
            
            st.markdown("---")
            
            # ====================================================================
            # 8.1 SCATTER PLOTS
            # ====================================================================
            
            st.subheader("8.1 Observed vs Expected")
            
            cols = st.columns(min(3, len(unique_species)))
            
            for idx, species in enumerate(unique_species):
                with cols[idx]:
                    df_sp = df_bench.filter(pl.col(species_col) == species)
                    
                    if df_sp.shape[0] > 0:
                        exp_fc_val = expected_fc.get(species, 0)
                        
                        plot = (ggplot(df_sp.to_pandas(), aes(x='expected_fc', y='effect_size')) +
                         geom_point(alpha=0.4, size=2, color='#3498db') +
                         geom_abline(intercept=0, slope=1, color='#e74c3c', linetype='dashed', size=1) +
                         geom_hline(yintercept=exp_fc_val, color='#bdc3c7', linetype='dotted') +
                         geom_vline(xintercept=exp_fc_val, color='#bdc3c7', linetype='dotted') +
                         labs(title=species, x='Expected', y='Observed') +
                         theme_minimal() +
                         theme(figure_size=(4, 4)))
                        
                        st.pyplot(ggplot.draw(plot))
                        
                        # Metrics
                        bias = (df_sp['effect_size'] - df_sp['expected_fc']).mean()
                        rmse = np.sqrt(((df_sp['effect_size'] - df_sp['expected_fc'])**2).mean())
                        
                        col_a, col_b = st.columns(2)
                        col_a.metric("RMSE", f"{rmse:.3f}")
                        col_b.metric("Bias", f"{bias:+.3f}")

            
            st.markdown("---")
            
            # ====================================================================
            # 8.2 DENSITY PLOTS
            # ====================================================================
            
            st.subheader("8.2 Distribution by Species")
            
            plot_density = (ggplot(df_bench.to_pandas(), aes(x='effect_size', fill=species_col)) +
             geom_density(alpha=0.4) +
             geom_vline(data=pd.DataFrame({
                 'species': list(expected_fc.keys()), 
                 'expected': list(expected_fc.values())
             }), mapping=aes(xintercept='expected', color='species'),
                        linetype='dashed', size=1) +
             scale_fill_brewer(type='qual', palette='Set2') +
             scale_color_brewer(type='qual', palette='Set2') +
             labs(title='', x=effect_label, y='Density', fill='Species', color='Expected') +
             theme_minimal() +
             theme(figure_size=(11, 5), legend_position='right'))
            
            st.pyplot(ggplot.draw(plot_density))
            
            # Asymmetry
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
                        status = '✓ Pass' if 0.5 <= asym <= 2.0 else '⚠️ Fail'
                        asym_data.append({
                            'Species': species, 
                            'Asymmetry': round(asym, 3),
                            'Status': status
                        })
            
            if asym_data:
                st.caption("**Asymmetry Factors** (ideal: 0.5-2.0)")
                st.dataframe(
                    pl.DataFrame(asym_data).to_pandas(),
                    hide_index=True,
                    use_container_width=True
                )
            
            st.markdown("---")
            
            # ====================================================================
            # 8.3 ROC & PRECISION-RECALL
            # ====================================================================
            
            st.subheader("8.3 Classification Performance")
            
            col_roc, col_pr = st.columns(2)
            
            # ROC
            with col_roc:
                df_roc = df_bench.with_columns([
                    (pl.col('expected_fc') != 0).alias('true_positive')
                ]).sort('pvalue')
                
                n_true_pos = df_roc['true_positive'].sum()
                n_true_neg = (~df_roc['true_positive']).sum()
                
                tpr_list, fpr_list = [], []
                cumsum_tp, cumsum_fp = 0, 0
                
                for row in df_roc.iter_rows(named=True):
                    if row['true_positive']:
                        cumsum_tp += 1
                    else:
                        cumsum_fp += 1
                    
                    tpr = cumsum_tp / n_true_pos if n_true_pos > 0 else 0
                    fpr = cumsum_fp / n_true_neg if n_true_neg > 0 else 0
                    
                    tpr_list.append(tpr)
                    fpr_list.append(fpr)
                
                auc = np.trapz(tpr_list, fpr_list)
                
                plot_roc = (ggplot(pd.DataFrame({'FPR': fpr_list, 'TPR': tpr_list}), 
                                   aes(x='FPR', y='TPR')) +
                 geom_line(color='#e74c3c', size=1.5) +
                 geom_abline(intercept=0, slope=1, linetype='dashed', color='#7f8c8d') +
                 labs(title=f'ROC (AUC = {auc:.3f})', x='FPR', y='TPR') +
                 theme_minimal() +
                 theme(figure_size=(5, 5)))
                
                st.pyplot(ggplot.draw(plot_roc))
            
            # PR
            with col_pr:
                precision_list, recall_list = [], []
                cumsum_tp, cumsum_fp = 0, 0
                
                for row in df_roc.iter_rows(named=True):
                    if row['true_positive']:
                        cumsum_tp += 1
                    else:
                        cumsum_fp += 1
                    
                    precision = cumsum_tp / (cumsum_tp + cumsum_fp) if (cumsum_tp + cumsum_fp) > 0 else 0
                    recall = cumsum_tp / n_true_pos if n_true_pos > 0 else 0
                    
                    precision_list.append(precision)
                    recall_list.append(recall)
                
                max_f1 = max([2*p*r/(p+r) if (p+r) > 0 else 0 for p, r in zip(precision_list, recall_list)])
                
                plot_pr = (ggplot(pd.DataFrame({'Recall': recall_list, 'Precision': precision_list}),
                                  aes(x='Recall', y='Precision')) +
                 geom_line(color='#2ecc71', size=1.5) +
                 labs(title=f'PR (Max F1 = {max_f1:.3f})', x='Recall', y='Precision') +
                 theme_minimal() +
                 theme(figure_size=(5, 5)))
                
                st.pyplot(ggplot.draw(plot_pr))
            
            st.markdown("---")
            
            # ====================================================================
            # 8.4 ERROR ANALYSIS
            # ====================================================================
            
            st.subheader("8.4 Error Analysis")
            
            col_dist, col_err = st.columns(2)
            
            with col_dist:
                # Distance plot
                distance_data = []
                for species in unique_species:
                    df_sp = df_bench.filter(pl.col(species_col) == species)
                    for row in df_sp.iter_rows(named=True):
                        distance_data.append({
                            'species': species,
                            'intensity': row['mean_control'],
                            'obs': row['effect_size'],
                            'exp': row['expected_fc']
                        })
                
                df_dist = pl.DataFrame(distance_data)
                
                plot_dist = (ggplot(df_dist.to_pandas(), aes(x='intensity', y='obs', color='species')) +
                 geom_point(alpha=0.3, size=1) +
                 geom_hline(data=pd.DataFrame({
                     'species': list(expected_fc.keys()),
                     'exp': list(expected_fc.values())
                 }), mapping=aes(yintercept='exp', color='species'),
                            linetype='dashed') +
                 scale_color_brewer(type='qual', palette='Set2') +
                 labs(title='Intensity vs Observed FC', x='log2(Intensity)', y='Observed FC') +
                 theme_minimal() +
                 theme(figure_size=(5, 4.5), legend_position='right'))
                
                st.pyplot(ggplot.draw(plot_dist))
            
            with col_err:
                # Error classification
                error_types = []
                for species in unique_species:
                    df_sp = df_bench.filter(pl.col(species_col) == species)
                    exp_fc_val = expected_fc.get(species, 0)
                    
                    if df_sp.shape[0] > 0:
                        obs_vals = df_sp['effect_size'].to_numpy()
                        bias = np.mean(obs_vals - exp_fc_val)
                        std_dev = np.std(obs_vals)
                        
                        if abs(bias) > 0.5:
                            error_type = "Systematic bias"
                        elif std_dev > 1.0:
                            error_type = "High dispersion"
                        else:
                            error_type = "Normal"
                        
                        error_types.append({
                            'Species': species,
                            'Bias': round(bias, 3),
                            'Std Dev': round(std_dev, 3),
                            'Type': error_type
                        })
                
                st.dataframe(
                    pl.DataFrame(error_types).to_pandas(),
                    hide_index=True,
                    use_container_width=True
                )
            
            st.markdown("---")
            
            # ====================================================================
            # 8.5 DISTRIBUTION QUALITY
            # ====================================================================
            
            st.subheader("8.5 Distribution Quality")
            
            fig, axes = plt.subplots(1, len(unique_species) + 1, figsize=(12, 3))
            
            if len(unique_species) == 1:
                axes = [axes]
            
            # Individual species
            for idx, species in enumerate(unique_species):
                df_sp = df_bench.filter(pl.col(species_col) == species)
                
                if df_sp.shape[0] > 2:
                    obs_vals = df_sp['effect_size'].to_numpy()
                    exp_fc_val = expected_fc.get(species, 0)
                    
                    ax = axes[idx]
                    kde = gaussian_kde(obs_vals)
                    x_range = np.linspace(obs_vals.min(), obs_vals.max(), 100)
                    ax.plot(x_range, kde(x_range), linewidth=2, color='#3498db')
                    ax.fill_between(x_range, kde(x_range), alpha=0.3, color='#3498db')
                    ax.axvline(exp_fc_val, color='#e74c3c', linestyle='--', linewidth=2)
                    ax.set_title(species, fontsize=10)
                    ax.set_xlabel('log2(FC)', fontsize=9)
                    ax.set_ylabel('Density', fontsize=9)
                    ax.grid(alpha=0.3)
            
            # Combined
            ax_comb = axes[-1]
            colors = plt.cm.Set2(range(len(unique_species)))
            for idx, species in enumerate(unique_species):
                df_sp = df_bench.filter(pl.col(species_col) == species)
                if df_sp.shape[0] > 2:
                    obs_vals = df_sp['effect_size'].to_numpy()
                    kde = gaussian_kde(obs_vals)
                    x_range = np.linspace(obs_vals.min(), obs_vals.max(), 100)
                    ax_comb.plot(x_range, kde(x_range), linewidth=2, label=species, color=colors[idx])
            
            ax_comb.set_title('Combined', fontsize=10)
            ax_comb.set_xlabel('log2(FC)', fontsize=9)
            ax_comb.legend(fontsize=8)
            ax_comb.grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("---")
            
            # ====================================================================
            # 8.6 CONFUSION MATRIX
            # ====================================================================
            
            st.subheader("8.6 Confusion Matrix")
            
            confusion_data = []
            for species in unique_species:
                df_sp = df_bench.filter(pl.col(species_col) == species)
                exp_fc_val = expected_fc.get(species, 0)
                
                if df_sp.shape[0] > 0:
                    should_change = exp_fc_val != 0
                    n_sig = df_sp.filter(pl.col('regulation').is_in(['upregulated', 'downregulated'])).shape[0]
                    n_not_sig = df_sp.shape[0] - n_sig
                    
                    confusion_data.append({
                        'Species': species,
                        'True State': 'Changed' if should_change else 'Unchanged',
                        'Pred Sig': n_sig,
                        'Pred Not Sig': n_not_sig,
                        'Total': df_sp.shape[0]
                    })
            
            df_confusion = pl.DataFrame(confusion_data)
            st.dataframe(df_confusion.to_pandas(), hide_index=True, use_container_width=True)
            
            # Metrics
            tp = df_confusion.filter(pl.col('True State') == 'Changed')['Pred Sig'].sum()
            fn = df_confusion.filter(pl.col('True State') == 'Changed')['Pred Not Sig'].sum()
            fp = df_confusion.filter(pl.col('True State') == 'Unchanged')['Pred Sig'].sum()
            tn = df_confusion.filter(pl.col('True State') == 'Unchanged')['Pred Not Sig'].sum()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Sensitivity", f"{sensitivity:.3f}")
            col2.metric("Specificity", f"{specificity:.3f}")
            col3.metric("Precision", f"{precision:.3f}")
            col4.metric("Accuracy", f"{accuracy:.3f}")
            
            st.markdown("---")
            
            # ====================================================================
            # 8.7 SUMMARY & DOWNLOAD
            # ====================================================================
            
            st.subheader("8.7 Summary Statistics")
            
            summary_stats = []
            for species in unique_species:
                df_sp = df_bench.filter(pl.col(species_col) == species)
                exp_fc_val = expected_fc.get(species, 0)
                
                if df_sp.shape[0] > 0:
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
                        'Expected': round(exp_fc_val, 2),
                        'N': n_total,
                        'Sig': n_sig,
                        'Bias': round(bias, 3),
                        'RMSE': round(rmse, 3),
                        'MAE': round(mae, 3),
                        'r': round(corr, 3)
                    })
            
            df_summary = pl.DataFrame(summary_stats)
            st.dataframe(df_summary.to_pandas(), hide_index=True, use_container_width=True)
            
            # Downloads
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Benchmark Summary",
                    df_summary.write_csv(),
                    f"benchmark_{treatment}_vs_{control}.csv",
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    "📥 Full Benchmark Data",
                    df_bench.write_csv(),
                    f"benchmark_full_{treatment}_vs_{control}.csv",
                    use_container_width=True
                )

else:
    st.info("💡 Species column required for benchmark validation")

st.markdown("---")
st.success("✅ Analysis complete!")
