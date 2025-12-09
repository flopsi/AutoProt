"""
pages/6_Differential_Abundance.py - COMPLETE VERSION WITH SPIKE-IN DEFAULTS
Limma-style empirical Bayes + LFQ benchmark default parameters
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
from sklearn.metrics import roc_curve, auc, precision_recall_curve

sys.path.append(str(Path(__file__).parent.parent))

# ============================================================================
# HELPER FUNCTIONS - LIMMA-STYLE EMPIRICAL BAYES
# ============================================================================

def fit_linear_model(df: pd.DataFrame, 
                     group1_cols: List[str], 
                     group2_cols: List[str]) -> pd.DataFrame:
    """
    Fit linear models for each protein (gene).
    """
    results = []
    
    for protein_id, row in df.iterrows():
        g1_vals = row[group1_cols].dropna()
        g2_vals = row[group2_cols].dropna()
        
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
        
        mean_g1 = g1_vals.mean()
        mean_g2 = g2_vals.mean()
        log2fc = mean_g1 - mean_g2
        
        n1, n2 = len(g1_vals), len(g2_vals)
        var1 = g1_vals.var(ddof=1)
        var2 = g2_vals.var(ddof=1)
        
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        sigma = np.sqrt(pooled_var)
        se = sigma * np.sqrt(1/n1 + 1/n2)
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


def empirical_bayes_moderation(fit_df: pd.DataFrame) -> pd.DataFrame:
    """Apply empirical Bayes moderation to variances (limma-style)."""
    from scipy.special import polygamma
    
    valid_df = fit_df.dropna(subset=['sigma', 'df']).copy()
    
    if len(valid_df) == 0:
        return fit_df
    
    s2 = valid_df['sigma'] ** 2
    df_gene = valid_df['df']
    
    log_s2 = np.log(s2)
    log_s2_mean = log_s2.mean()
    log_s2_var = log_s2.var()
    
    def trigamma_inv(y):
        if y < 1e-6:
            return 1/y
        else:
            d = 1/y
            for _ in range(10):
                tri = polygamma(1, d/2)
                tetra = -0.5 * polygamma(2, d/2)
                d = d - (tri - y) / tetra
            return d
    
    d0 = trigamma_inv(log_s2_var)
    d0 = max(d0, 1)
    
    s0_squared = np.exp(log_s2_mean + polygamma(1, d0/2))
    
    df_post = d0 + df_gene
    s2_post = (d0 * s0_squared + df_gene * s2) / df_post
    
    valid_df['s2_prior'] = s0_squared
    valid_df['df_prior'] = d0
    valid_df['s2_post'] = s2_post
    valid_df['df_post'] = df_post
    valid_df['se_post'] = np.sqrt(s2_post) * valid_df['se'] / valid_df['sigma']
    valid_df['t_stat'] = valid_df['log2fc'] / valid_df['se_post']
    valid_df['pvalue'] = 2 * (1 - t_dist.cdf(np.abs(valid_df['t_stat']), valid_df['df_post']))
    
    result_df = fit_df.copy()
    for col in ['s2_prior', 'df_prior', 's2_post', 'df_post', 'se_post', 't_stat', 'pvalue']:
        result_df[col] = valid_df[col]
    
    return result_df


def benjamini_hochberg_fdr(pvalues: pd.Series) -> pd.Series:
    """Calculate FDR using Benjamini-Hochberg procedure."""
    valid_pvals = pvalues.dropna().sort_values()
    n = len(valid_pvals)
    
    if n == 0:
        return pd.Series(index=pvalues.index, dtype=float)
    
    ranks = np.arange(1, n + 1)
    fdr_vals = valid_pvals.values * n / ranks
    fdr_vals = np.minimum.accumulate(fdr_vals[::-1])[::-1]
    fdr_vals = np.minimum(fdr_vals, 1.0)
    
    fdr_dict = dict(zip(valid_pvals.index, fdr_vals))
    return pvalues.index.map(fdr_dict)


def classify_regulation(log2fc: float, pvalue: float, fc_threshold: float = 1.0, pval_threshold: float = 0.05) -> str:
    """Classify protein regulation status."""
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


def calculate_error_metrics(results_df: pd.DataFrame,
                           true_fc_dict: Dict[str, float],
                           species_mapping: Dict,
                           fc_threshold: float = 1.0,
                           pval_threshold: float = 0.05) -> Tuple[Dict, pd.DataFrame]:
    """Calculate confusion matrix and performance metrics."""
    results_df = results_df.copy()
    results_df = results_df[results_df['regulation'] != 'not_tested']
    
    if len(results_df) == 0:
        return {}, pd.DataFrame()
    
    results_df['species_key'] = results_df.index.map(lambda x: species_mapping.get(x, 'Unknown'))
    results_df['true_log2fc'] = results_df['species_key'].map(lambda x: true_fc_dict.get(x, np.nan))
    
    valid_results = results_df.dropna(subset=['true_log2fc'])
    
    if len(valid_results) == 0:
        return {}, pd.DataFrame()
    
    valid_results['true_regulated'] = valid_results['true_log2fc'].apply(lambda x: abs(x) > fc_threshold)
    valid_results['observed_regulated'] = valid_results['regulation'].isin(['up', 'down'])
    
    TP = ((valid_results["true_regulated"]) & (valid_results["observed_regulated"])).sum()
    FP = ((~valid_results["true_regulated"]) & (valid_results["observed_regulated"])).sum()
    TN = ((~valid_results["true_regulated"]) & (~valid_results["observed_regulated"])).sum()
    FN = ((valid_results["true_regulated"]) & (~valid_results["observed_regulated"])).sum()
    
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    def wilson_ci(successes, trials, z=1.96):
        p = successes / trials if trials > 0 else 0
        denominator = 1 + z**2 / trials if trials > 0 else 1
        center = (p + z**2 / (2*trials)) / denominator if trials > 0 else 0
        adjustment = z * np.sqrt(p*(1-p)/trials + z**2/(4*trials**2)) / denominator if trials > 0 else 0
        return (max(0, center - adjustment), min(1, center + adjustment))
    
    sens_ci = wilson_ci(int(TP), int(TP + FN)) if (TP + FN) > 0 else (0, 0)
    spec_ci = wilson_ci(int(TN), int(TN + FP)) if (TN + FP) > 0 else (0, 0)
    
    overall_metrics = {
        'TP': int(TP),
        'FP': int(FP),
        'TN': int(TN),
        'FN': int(FN),
        'Sensitivity': sensitivity,
        'Sensitivity_CI': sens_ci,
        'Specificity': specificity,
        'Specificity_CI': spec_ci,
        'Precision': precision,
        'FPR': fpr,
        'FNR': fnr,
        'Total_Tested': len(valid_results)
    }
    
    species_metrics = []
    for species in valid_results['species_key'].unique():
        species_data = valid_results[valid_results['species_key'] == species]
        theo_fc = true_fc_dict.get(species, 0.0)
        
        species_data = species_data.copy()
        species_data['error'] = species_data['log2fc'] - theo_fc
        
        n_proteins = len(species_data)
        rmse = np.sqrt((species_data['error'] ** 2).mean())
        mae = species_data['error'].abs().mean()
        bias = species_data['error'].mean()
        
        rmse_ci = (
            rmse * np.sqrt(1 - 1.96/np.sqrt(n_proteins)),
            rmse * np.sqrt(1 + 1.96/np.sqrt(n_proteins))
        ) if n_proteins > 1 else (rmse, rmse)
        
        n_detected = (species_data['observed_regulated']).sum()
        detection_rate = n_detected / n_proteins if n_proteins > 0 else 0
        
        species_metrics.append({
            'Species': species,
            'N': n_proteins,
            'Theo_FC': f"{theo_fc:.2f}",
            'RMSE': rmse,
            'RMSE_CI_Low': rmse_ci[0],
            'RMSE_CI_High': rmse_ci[1],
            'MAE': mae,
            'Bias': bias,
            'Detection_Rate': detection_rate,
        })
    
    species_df = pd.DataFrame(species_metrics)
    return overall_metrics, species_df


def compute_roc_curve(results_df: pd.DataFrame,
                      true_fc_dict: Dict[str, float],
                      species_mapping: Dict,
                      fc_threshold: float = 1.0) -> Tuple[list, list, list, float]:
    """Compute ROC curve."""
    results_df = results_df.copy()
    results_df['species_key'] = results_df.index.map(lambda x: species_mapping.get(x, 'Unknown'))
    results_df['true_log2fc'] = results_df['species_key'].map(lambda x: true_fc_dict.get(x, np.nan))
    
    valid_results = results_df.dropna(subset=['true_log2fc', 'pvalue'])
    
    if len(valid_results) == 0:
        return [0, 1], [0, 1], [1, 0], 0.0
    
    valid_results['true_regulated'] = valid_results['true_log2fc'].apply(lambda x: abs(x) > fc_threshold)
    valid_results = valid_results.sort_values('pvalue')
    
    fpr_list, tpr_list, thresholds = [], [], []
    n_neg = (~valid_results['true_regulated']).sum()
    n_pos = valid_results['true_regulated'].sum()
    
    if n_neg == 0 or n_pos == 0:
        return [0, 1], [0, 1], [1, 0], 0.0
    
    for pval_threshold in np.linspace(1, 0, 50):
        valid_results['predicted'] = valid_results['pvalue'] < pval_threshold
        tp = (valid_results['true_regulated'] & valid_results['predicted']).sum()
        fp = (~valid_results['true_regulated'] & valid_results['predicted']).sum()
        
        fpr_list.append(fp / n_neg)
        tpr_list.append(tp / n_pos)
        thresholds.append(pval_threshold)
    
    roc_auc = auc(fpr_list, tpr_list)
    return fpr_list, tpr_list, thresholds, roc_auc


def compute_precision_recall_curve(results_df: pd.DataFrame,
                                   true_fc_dict: Dict[str, float],
                                   species_mapping: Dict,
                                   fc_threshold: float = 1.0) -> Tuple[list, list, list, float]:
    """Compute precision-recall curve."""
    results_df = results_df.copy()
    results_df['species_key'] = results_df.index.map(lambda x: species_mapping.get(x, 'Unknown'))
    results_df['true_log2fc'] = results_df['species_key'].map(lambda x: true_fc_dict.get(x, np.nan))
    
    valid_results = results_df.dropna(subset=['true_log2fc', 'pvalue'])
    
    if len(valid_results) == 0:
        return [0, 1], [1, 0], [1, 0], 0.0
    
    valid_results['true_regulated'] = valid_results['true_log2fc'].apply(lambda x: abs(x) > fc_threshold)
    valid_results = valid_results.sort_values('pvalue')
    
    precision_list, recall_list, thresholds = [], [], []
    n_pos = valid_results['true_regulated'].sum()
    
    if n_pos == 0:
        return [0, 1], [1, 0], [1, 0], 0.0
    
    for pval_threshold in np.linspace(1, 0, 50):
        valid_results['predicted'] = valid_results['pvalue'] < pval_threshold
        tp = (valid_results['true_regulated'] & valid_results['predicted']).sum()
        fp = (~valid_results['true_regulated'] & valid_results['predicted']).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / n_pos
        
        precision_list.append(precision)
        recall_list.append(recall)
        thresholds.append(pval_threshold)
    
    pr_auc = auc(recall_list, precision_list)
    return recall_list, precision_list, thresholds, pr_auc


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Differential Abundance",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Differential Abundance Analysis")
st.markdown("Limma-style empirical Bayes moderated t-statistics with spike-in validation")
st.markdown("---")

# ============================================================================
# CHECK FOR IMPUTED DATA
# ============================================================================

if 'df_imputed' not in st.session_state or st.session_state.df_imputed is None:
    st.error("❌ No imputed data. Please complete **🔧 Missing Value Imputation** first")
    st.stop()

df = st.session_state.df_imputed.copy()
numeric_cols = st.session_state.numeric_cols
sample_to_condition = st.session_state.get('sample_to_condition', {})
species_col = st.session_state.species_col

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

ref_samples = condition_samples[reference_group]
treat_samples = condition_samples[treatment_group]

st.markdown(f"""
**Comparison**: `{reference_group}` (n={len(ref_samples)}) vs `{treatment_group}` (n={len(treat_samples)})

**Log2FC Convention**: Positive FC = higher in **{reference_group}** | Negative FC = higher in **{treatment_group}**
""")

st.markdown("---")

# ============================================================================
# 2. SPIKE-IN VALIDATION (OPTIONAL) - LFQ BENCHMARK DEFAULTS
# ============================================================================

st.subheader("2️⃣ Spike-in Validation (Optional)")

st.markdown("""
For spike-in studies, specify expected fold changes per species.
**LFQ Benchmark Default Protocols:**
""")

use_spikein = st.checkbox("Use spike-in validation with known fold changes")

theoretical_fc_dict = {}

if use_spikein:
    st.markdown("### 🧪 Benchmark Spike-in Scenarios (from LFQb v3.4.1)")
    
    scenario = st.radio(
        "Select Spike-in Scenario:",
        options=[
            "Custom",
            "Scenario A: Human-Yeast-EColi (1:5:0.2)",
            "Scenario B: Human-Yeast-EColi (1:0.2:5)",
            "Benchmark Reference: 50% Human, 25% Yeast, 25% EColi"
        ],
        index=0
    )
    
    # Default fold changes from LFQb benchmark papers
    if scenario == "Scenario A: Human-Yeast-EColi (1:5:0.2)":
        # Human as reference
        st.markdown("**Reference**: Human (unchanged)")
        st.info("""
        **Expected Fold Changes**:
        - **Yeast**: 5-fold increase (log2 = 2.32)
        - **EColi**: 0.2-fold decrease (log2 = -2.32)
        - **Human**: 1:1 (log2 = 0)
        """)
        theoretical_fc_dict = {
            'HUMAN': 0.0,
            'YEAST': 2.32,
            'ECOLI': -2.32
        }
        
    elif scenario == "Scenario B: Human-Yeast-EColi (1:0.2:5)":
        st.markdown("**Reference**: Human (unchanged)")
        st.info("""
        **Expected Fold Changes**:
        - **Yeast**: 0.2-fold decrease (log2 = -2.32)
        - **EColi**: 5-fold increase (log2 = 2.32)
        - **Human**: 1:1 (log2 = 0)
        """)
        theoretical_fc_dict = {
            'HUMAN': 0.0,
            'YEAST': -2.32,
            'ECOLI': 2.32
        }
        
    elif scenario == "Benchmark Reference: 50% Human, 25% Yeast, 25% EColi":
        st.markdown("**Reference**: 50% Human, 25% Yeast, 25% EColi (balanced mixture)")
        st.info("""
        **Comparison samples can vary yeast/EColi from 0.4-fold to 1.6-fold relative to reference**
        - Current implementation uses custom entry below
        """)
        
    if scenario == "Custom":
        st.markdown("### Custom Fold Changes per Species")
        st.markdown("Enter log2 fold change or ratio (Ref:Treatment)")
        
        fc_input_method = st.radio(
            "Input method:",
            options=["Log2 Fold Change", "Ratio (e.g., 2:1, 1:2)"],
            horizontal=True
        )
        
        species_list = sorted([s for s in df[species_col].unique() if s != 'Unknown'])
        
        col1, col2, col3 = st.columns(3)
        
        for idx, species in enumerate(species_list[:6]):
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
    
    if theoretical_fc_dict:
        st.markdown("### 📋 Expected Fold Changes Summary")
        fc_summary = pd.DataFrame([
            {'Species': k, 'Log2FC': v, 'Linear_FC': f"{2**v:.2f}x"} 
            for k, v in theoretical_fc_dict.items()
        ])
        st.dataframe(fc_summary, use_container_width=True)

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
        help="False Discovery Rate cutoff (Benjamini-Hochberg)"
    )
    
    use_fdr = st.checkbox("Use FDR correction", value=True, help="Benjamini-Hochberg FDR")

st.markdown("---")

# ============================================================================
# 4. RUN ANALYSIS
# ============================================================================

st.subheader("4️⃣ Run Differential Abundance Analysis")

if st.button("🚀 Run Analysis", type="primary"):
    with st.spinner("Performing limma-style analysis..."):
        
        df_log2 = df[numeric_cols].apply(lambda x: np.log2(x + 1) if x.min() > 1 else x)
        
        fit_results = fit_linear_model(df_log2, ref_samples, treat_samples)
        moderated_results = empirical_bayes_moderation(fit_results)
        
        if use_fdr:
            moderated_results['fdr'] = benjamini_hochberg_fdr(moderated_results['pvalue'])
            test_col = 'fdr'
        else:
            moderated_results['fdr'] = moderated_results['pvalue']
            test_col = 'pvalue'
        
        moderated_results['regulation'] = moderated_results.apply(
            lambda row: classify_regulation(row['log2fc'], row[test_col], fc_threshold, pval_threshold),
            axis=1
        )
        
        moderated_results['neg_log10_pval'] = -np.log10(moderated_results[test_col].replace(0, 1e-300))
        moderated_results['species'] = moderated_results.index.map(
            lambda x: df.loc[x, species_col] if x in df.index else 'Unknown'
        )
        
        st.session_state.dea_results = moderated_results
        st.session_state.dea_ref_group = reference_group
        st.session_state.dea_treat_group = treatment_group
        st.session_state.dea_fc_threshold = fc_threshold
        st.session_state.dea_pval_threshold = pval_threshold
        st.session_state.theoretical_fc_dict = theoretical_fc_dict
        
        st.success("✅ Analysis complete!")

# ============================================================================
# 5. RESULTS VISUALIZATION
# ============================================================================

if 'dea_results' in st.session_state:
    results = st.session_state.dea_results
    reference_group = st.session_state.dea_ref_group
    treatment_group = st.session_state.dea_treat_group
    fc_threshold = st.session_state.dea_fc_threshold
    pval_threshold = st.session_state.dea_pval_threshold
    theoretical_fc_dict = st.session_state.theoretical_fc_dict
    
    st.markdown("---")
    st.subheader("5️⃣ Results Summary")
    
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
    
    # ================================================================
    # VOLCANO PLOT
    # ================================================================
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
        title=f'Volcano Plot: {reference_group} vs {treatment_group}',
        labels={
            'log2fc': f'Log2 Fold Change ({reference_group} / {treatment_group})',
            'neg_log10_pval': '-Log10(FDR)' if use_fdr else '-Log10(p-value)'
        },
        height=600
    )
    
    fig.add_hline(y=-np.log10(pval_threshold), line_dash="dash", line_color="gray", annotation_text=f"FDR = {pval_threshold}")
    fig.add_vline(x=fc_threshold, line_dash="dash", line_color="gray")
    fig.add_vline(x=-fc_threshold, line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ================================================================
    # MA PLOT
    # ================================================================
    st.markdown("### 📈 MA Plot (Mean vs Log2FC)")
    
    ma_df = results[results['regulation'] != 'not_tested'].copy()
    ma_df['A'] = (ma_df['mean_g1'] + ma_df['mean_g2']) / 2
    ma_df['M'] = ma_df['log2fc']
    ma_df['Regulation'] = ma_df['regulation'].map({
        'up': f'Up ({n_up})',
        'down': f'Down ({n_down})',
        'not_significant': f'NS ({n_ns})'
    })
    
    fig_ma = px.scatter(
        ma_df,
        x='A',
        y='M',
        color='Regulation',
        hover_data=['species'],
        color_discrete_map=color_map,
        title='MA Plot: Mean vs Log2 Fold Change',
        labels={
            'A': 'Mean Log2 Intensity',
            'M': f'Log2 Fold Change ({reference_group} / {treatment_group})'
        },
        height=600
    )
    
    fig_ma.add_hline(y=fc_threshold, line_dash="dash", line_color="gray")
    fig_ma.add_hline(y=-fc_threshold, line_dash="dash", line_color="gray")
    fig_ma.add_hline(y=0, line_dash="solid", line_color="red", opacity=0.3)
    
    st.plotly_chart(fig_ma, use_container_width=True)
    
    # ================================================================
    # RESULTS TABLE
    # ================================================================
    st.markdown("### 📋 Differentially Abundant Proteins")
    
    sig_results = results[results['regulation'].isin(['up', 'down'])].copy()
    sig_results = sig_results.sort_values('fdr')
    
    display_cols = ['log2fc', 'mean_g1', 'mean_g2', 't_stat', 'pvalue', 'fdr', 'regulation', 'species']
    display_df = sig_results[display_cols].head(50)
    
    st.dataframe(display_df.round(4), use_container_width=True)
    
    # ================================================================
    # VALIDATION METRICS (if spike-in)
    # ================================================================
    if len(theoretical_fc_dict) > 0:
        st.markdown("---")
        st.subheader("6️⃣ Spike-in Validation Metrics")
        
        species_mapping = dict(zip(results.index, results['species']))
        overall_metrics, species_metrics_df = calculate_error_metrics(
            results,
            theoretical_fc_dict,
            species_mapping,
            fc_threshold,
            pval_threshold
        )
        
        if len(overall_metrics) > 0:
            st.markdown("#### Overall Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Sensitivity", f"{overall_metrics['Sensitivity']:.1%}", 
                       help=f"CI: {overall_metrics['Sensitivity_CI'][0]:.1%}-{overall_metrics['Sensitivity_CI'][1]:.1%}")
            col2.metric("Specificity", f"{overall_metrics['Specificity']:.1%}",
                       help=f"CI: {overall_metrics['Specificity_CI'][0]:.1%}-{overall_metrics['Specificity_CI'][1]:.1%}")
            col3.metric("Precision", f"{overall_metrics['Precision']:.1%}")
            col4.metric("False Positive Rate", f"{overall_metrics['FPR']:.1%}")
            
            st.markdown("#### Confusion Matrix")
            cm_data = {
                'True Positive': overall_metrics['TP'],
                'False Positive': overall_metrics['FP'],
                'True Negative': overall_metrics['TN'],
                'False Negative': overall_metrics['FN']
            }
            cm_df = pd.DataFrame([cm_data])
            st.dataframe(cm_df, use_container_width=True)
            
            # Per-species metrics
            if len(species_metrics_df) > 0:
                st.markdown("#### Per-Species Accuracy Metrics")
                
                display_species_cols = ['Species', 'N', 'Theo_FC', 'RMSE', 'MAE', 'Bias', 'Detection_Rate']
                display_species = species_metrics_df[[c for c in display_species_cols if c in species_metrics_df.columns]]
                st.dataframe(display_species.round(3), use_container_width=True)
                
                # RMSE plot with CI
                if 'RMSE' in species_metrics_df.columns:
                    st.markdown("#### RMSE by Species (with 95% CI)")
                    
                    fig_rmse = go.Figure()
                    
                    for idx, row in species_metrics_df.iterrows():
                        fig_rmse.add_trace(go.Scatter(
                            x=[row['Species'], row['Species']],
                            y=[row['RMSE_CI_Low'], row['RMSE_CI_High']],
                            mode='lines',
                            line=dict(color='lightblue', width=2),
                            hoverinfo='skip',
                            showlegend=False
                        ))
                        
                        fig_rmse.add_trace(go.Scatter(
                            x=[row['Species']],
                            y=[row['RMSE']],
                            mode='markers',
                            marker=dict(size=10, color='blue'),
                            text=f"{row['Species']}<br>RMSE: {row['RMSE']:.3f}",
                            hovertemplate='%{text}',
                            showlegend=False
                        ))
                    
                    fig_rmse.update_layout(
                        title='RMSE with 95% Confidence Intervals',
                        xaxis_title='Species',
                        yaxis_title='RMSE',
                        height=500
                    )
                    
                    st.plotly_chart(fig_rmse, use_container_width=True)
                
                # Detection rate plot
                if 'Detection_Rate' in species_metrics_df.columns:
                    st.markdown("#### Detection Rate by Species")
                    
                    fig_det = px.bar(
                        species_metrics_df,
                        x='Species',
                        y='Detection_Rate',
                        title='Detection Rate (% of Proteins Called Significant)',
                        labels={'Detection_Rate': 'Detection Rate'},
                        height=500
                    )
                    fig_det.update_yaxes(tickformat='.0%')
                    st.plotly_chart(fig_det, use_container_width=True)
        
        # ================================================================
        # ROC AND PR CURVES
        # ================================================================
        st.markdown("---")
        st.subheader("7️⃣ Classification Performance Curves")
        
        fpr_list, tpr_list, thresholds, roc_auc = compute_roc_curve(
            results,
            theoretical_fc_dict,
            species_mapping,
            fc_threshold
        )
        
        recall_list, precision_list, pr_thresholds, pr_auc = compute_precision_recall_curve(
            results,
            theoretical_fc_dict,
            species_mapping,
            fc_threshold
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ROC Curve")
            fig_roc = go.Figure()
            
            fig_roc.add_trace(go.Scatter(
                x=fpr_list,
                y=tpr_list,
                mode='lines',
                line=dict(color='#1f77b4', width=3),
                name=f'ROC (AUC = {roc_auc:.3f})',
                fill='tozeroy'
            ))
            
            fig_roc.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name='Chance',
                showlegend=True
            ))
            
            fig_roc.update_layout(
                title='ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=500,
                hovermode='closest'
            )
            
            st.plotly_chart(fig_roc, use_container_width=True)
        
        with col2:
            st.markdown("#### Precision-Recall Curve")
            fig_pr = go.Figure()
            
            fig_pr.add_trace(go.Scatter(
                x=recall_list,
                y=precision_list,
                mode='lines',
                line=dict(color='#ff7f0e', width=3),
                name=f'PR (AP = {pr_auc:.3f})',
                fill='tozeroy'
            ))
            
            fig_pr.update_layout(
                title='Precision-Recall Curve',
                xaxis_title='Recall (Sensitivity)',
                yaxis_title='Precision',
                height=500,
                hovermode='closest'
            )
            
            st.plotly_chart(fig_pr, use_container_width=True)
        
        # ================================================================
        # ERROR DISTRIBUTION
        # ================================================================
        st.markdown("---")
        st.markdown("#### Error Distribution")
        
        results_copy = results.copy()
        results_copy['species_key'] = results_copy.index.map(species_mapping)
        results_copy['true_log2fc'] = results_copy['species_key'].map(
            lambda x: theoretical_fc_dict.get(x, np.nan)
        )
        valid_for_error = results_copy.dropna(subset=['true_log2fc'])
        
        if len(valid_for_error) > 0:
            valid_for_error = valid_for_error.copy()
            valid_for_error['error'] = valid_for_error['log2fc'] - valid_for_error['true_log2fc']
            
            fig_error = px.histogram(
                valid_for_error,
                x='error',
                color='regulation',
                title='Error Distribution: Observed - Theoretical Log2FC',
                labels={'error': 'Error (log2FC)', 'regulation': 'Regulation'},
                nbins=50,
                height=500
            )
            fig_error.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero Error")
            
            st.plotly_chart(fig_error, use_container_width=True)
    
    # ================================================================
    # DOWNLOAD RESULTS
    # ================================================================
    st.markdown("---")
    st.subheader("💾 Export Results")
    
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
