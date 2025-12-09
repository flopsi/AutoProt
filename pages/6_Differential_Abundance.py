"""
pages/6_Differential_Abundance.py - COMPLETE VERSION WITH STABLE PROTEOME FP CONTROL
Limma-style empirical Bayes + composition input + stable proteome for FP rate
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
    """Fit linear models for each protein (gene)."""
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


def calculate_error_metrics_stable_proteome(results_df: pd.DataFrame,
                                           theoretical_fc_dict: Dict[str, float],
                                           species_mapping: Dict,
                                           stable_proteome_threshold: float = 0.5,
                                           pval_threshold: float = 0.05) -> Tuple[Dict, pd.DataFrame]:
    """
    Calculate confusion matrix and performance metrics using ONLY stable proteome.
    
    Stable proteome: proteins with |log2FC| < stable_proteome_threshold
    These proteins should NOT be significantly different (false positives counted here).
    
    Args:
        results_df: Results DataFrame
        theoretical_fc_dict: Expected fold changes per species
        species_mapping: Protein ID → species mapping
        stable_proteome_threshold: Max |log2FC| for stable proteome (typically 0.5)
        pval_threshold: Significance threshold (FDR)
    
    Returns:
        (overall_metrics_dict, per_species_metrics_df)
    """
    results_df = results_df.copy()
    results_df = results_df[results_df['regulation'] != 'not_tested']
    
    if len(results_df) == 0:
        return {}, pd.DataFrame()
    
    results_df['species_key'] = results_df.index.map(lambda x: species_mapping.get(x, 'Unknown'))
    results_df['true_log2fc'] = results_df['species_key'].map(lambda x: theoretical_fc_dict.get(x, np.nan))
    
    valid_results = results_df.dropna(subset=['true_log2fc'])
    
    if len(valid_results) == 0:
        return {}, pd.DataFrame()
    
    # === FILTER TO STABLE PROTEOME ONLY ===
    # Stable proteome: |true_log2fc| < threshold (proteins that should NOT change)
    valid_results['is_stable'] = valid_results['true_log2fc'].apply(
        lambda x: abs(x) < stable_proteome_threshold
    )
    stable_results = valid_results[valid_results['is_stable']].copy()
    
    if len(stable_results) == 0:
        st.warning(f"⚠️ No proteins in stable proteome range (|log2FC| < {stable_proteome_threshold})")
        return {}, pd.DataFrame()
    
    # For stable proteome, "true" means NOT regulated (should have small FC)
    # These are used to calculate false positive rate
    stable_results['true_not_regulated'] = True  # All stable proteins should be not regulated
    stable_results['observed_regulated'] = stable_results['regulation'].isin(['up', 'down'])
    
    # Confusion matrix (for stable proteome only)
    TN = (~stable_results['observed_regulated']).sum()  # Correctly called as not significant
    FP = (stable_results['observed_regulated']).sum()   # Incorrectly called as significant (FALSE POSITIVES)
    
    # For stable proteome, we can calculate False Positive Rate
    total_stable = len(stable_results)
    false_positive_rate = FP / total_stable if total_stable > 0 else 0.0
    true_negative_rate = TN / total_stable if total_stable > 0 else 0.0
    
    # CI for FPR (Wilson score interval)
    def wilson_ci(successes, trials, z=1.96):
        p = successes / trials if trials > 0 else 0
        denominator = 1 + z**2 / trials if trials > 0 else 1
        center = (p + z**2 / (2*trials)) / denominator if trials > 0 else 0
        adjustment = z * np.sqrt(p*(1-p)/trials + z**2/(4*trials**2)) / denominator if trials > 0 else 0
        return (max(0, center - adjustment), min(1, center + adjustment))
    
    fpr_ci = wilson_ci(int(FP), int(total_stable)) if total_stable > 0 else (0, 0)
    tnr_ci = wilson_ci(int(TN), int(total_stable)) if total_stable > 0 else (0, 0)
    
    overall_metrics = {
        'Dataset': 'Stable Proteome (False Positive Rate)',
        'Total_Stable_Proteins': total_stable,
        'True_Negatives': int(TN),
        'False_Positives': int(FP),
        'False_Positive_Rate': false_positive_rate,
        'FPR_CI': fpr_ci,
        'True_Negative_Rate': true_negative_rate,
        'TNR_CI': tnr_ci,
    }
    
    # Per-species metrics (for stable proteome)
    species_metrics = []
    for species in stable_results['species_key'].unique():
        species_data = stable_results[stable_results['species_key'] == species]
        
        n_proteins = len(species_data)
        n_fp = (species_data['observed_regulated']).sum()
        n_tn = (~species_data['observed_regulated']).sum()
        
        fpr_species = n_fp / n_proteins if n_proteins > 0 else 0
        
        # Absolute error (should be close to 0 for stable)
        species_data_copy = species_data.copy()
        species_data_copy['abs_error'] = species_data_copy['log2fc'].abs()
        mae = species_data_copy['abs_error'].mean()
        
        species_metrics.append({
            'Species': species,
            'N': n_proteins,
            'True_Negatives': n_tn,
            'False_Positives': n_fp,
            'FP_Rate': f"{fpr_species:.1%}",
            'Mean_Abs_Error': f"{mae:.3f}",
            'Note': 'Stable proteome metrics'
        })
    
    species_df = pd.DataFrame(species_metrics)
    return overall_metrics, species_df


def calculate_error_metrics_variable_proteins(results_df: pd.DataFrame,
                                              theoretical_fc_dict: Dict[str, float],
                                              species_mapping: Dict,
                                              stable_proteome_threshold: float = 0.5,
                                              fc_threshold: float = 1.0,
                                              pval_threshold: float = 0.05) -> Tuple[Dict, pd.DataFrame]:
    """
    Calculate performance metrics for variable proteome proteins.
    
    Variable proteome: proteins with |log2FC| >= stable_proteome_threshold
    These proteins SHOULD be significantly different.
    
    Args:
        results_df: Results DataFrame
        theoretical_fc_dict: Expected fold changes per species
        species_mapping: Protein ID → species mapping
        stable_proteome_threshold: Min |log2FC| for variable proteome
        fc_threshold: Fold change threshold for calling significant
        pval_threshold: Significance threshold (FDR)
    
    Returns:
        (overall_metrics_dict, per_species_metrics_df)
    """
    results_df = results_df.copy()
    results_df = results_df[results_df['regulation'] != 'not_tested']
    
    if len(results_df) == 0:
        return {}, pd.DataFrame()
    
    results_df['species_key'] = results_df.index.map(lambda x: species_mapping.get(x, 'Unknown'))
    results_df['true_log2fc'] = results_df['species_key'].map(lambda x: theoretical_fc_dict.get(x, np.nan))
    
    valid_results = results_df.dropna(subset=['true_log2fc'])
    
    if len(valid_results) == 0:
        return {}, pd.DataFrame()
    
    # === FILTER TO VARIABLE PROTEOME ONLY ===
    # Variable proteome: |true_log2fc| >= threshold (proteins that SHOULD change)
    valid_results['is_variable'] = valid_results['true_log2fc'].apply(
        lambda x: abs(x) >= stable_proteome_threshold
    )
    variable_results = valid_results[valid_results['is_variable']].copy()
    
    if len(variable_results) == 0:
        st.warning(f"⚠️ No proteins in variable proteome range (|log2FC| >= {stable_proteome_threshold})")
        return {}, pd.DataFrame()
    
    # True regulation (protein SHOULD be regulated with |log2FC| >= fc_threshold)
    variable_results['true_regulated'] = variable_results['true_log2fc'].apply(
        lambda x: abs(x) >= fc_threshold
    )
    variable_results['observed_regulated'] = variable_results['regulation'].isin(['up', 'down'])
    
    # Confusion matrix (for variable proteome)
    TP = ((variable_results['true_regulated']) & (variable_results['observed_regulated'])).sum()
    FN = ((variable_results['true_regulated']) & (~variable_results['observed_regulated'])).sum()
    TN = ((~variable_results['true_regulated']) & (~variable_results['observed_regulated'])).sum()
    FP = ((~variable_results['true_regulated']) & (variable_results['observed_regulated'])).sum()
    
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
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
        'Dataset': 'Variable Proteome (Sensitivity/Specificity)',
        'Total_Variable_Proteins': len(variable_results),
        'True_Positives': int(TP),
        'False_Negatives': int(FN),
        'True_Negatives': int(TN),
        'False_Positives': int(FP),
        'Sensitivity': sensitivity,
        'Sensitivity_CI': sens_ci,
        'Specificity': specificity,
        'Specificity_CI': spec_ci,
        'Precision': precision,
    }
    
    # Per-species metrics
    species_metrics = []
    for species in variable_results['species_key'].unique():
        species_data = variable_results[variable_results['species_key'] == species]
        theo_fc = theoretical_fc_dict.get(species, 0.0)
        
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


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Differential Abundance",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Differential Abundance Analysis")
st.markdown("Limma-style empirical Bayes with stable proteome FP control")
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
# 2. SPIKE-IN VALIDATION - COMPOSITION-BASED APPROACH
# ============================================================================

st.subheader("2️⃣ Spike-in Validation (Optional)")

st.markdown("""
For spike-in studies, specify percentage composition per species per condition.

**Proteome Stratification**:
- **Stable Proteome**: |log2FC| < 0.5 (used to calculate **False Positive Rate**)
- **Variable Proteome**: |log2FC| ≥ 0.5 (used to calculate **Sensitivity/Specificity**)
""")

use_composition = st.checkbox("Define spike-in by percentage composition per species per condition")

theoretical_fc_dict = {}

if use_composition:
    species_list = sorted([s for s in df[species_col].unique() if s != 'Unknown'])
    
    st.markdown("### 📊 Percentage Composition Input")
    st.markdown(f"""
    **Conditions**: {reference_group} | {treatment_group}
    
    Enter the percentage (0-100) of each species in each condition.
    """)
    
    tab1, tab2, tab3 = st.tabs(["Preset Scenarios", "Custom Composition", "View Calculated FC"])
    
    with tab1:
        st.markdown("### 🧪 Benchmark Spike-in Scenarios (from LFQb v3.4.1)")
        
        preset_option = st.radio(
            "Select Preset Scenario:",
            options=[
                "None - Use Custom",
                "Scenario A: Balanced → Human Heavy",
                "Scenario B: Balanced → Yeast Heavy",
                "Scenario C: Balanced → EColi Heavy",
                "Scenario D: Extreme Flip"
            ]
        )
        
        composition_ref = {}
        composition_treat = {}
        
        if preset_option == "Scenario A: Balanced → Human Heavy":
            st.info("""
            **Reference**: 40% Human, 30% Yeast, 30% EColi
            **Treatment**: 70% Human, 15% Yeast, 15% EColi
            
            Expected: Human ↑, Yeast ↓, EColi ↓
            """)
            composition_ref = {'HUMAN': 40, 'YEAST': 30, 'ECOLI': 30}
            composition_treat = {'HUMAN': 70, 'YEAST': 15, 'ECOLI': 15}
            
        elif preset_option == "Scenario B: Balanced → Yeast Heavy":
            st.info("""
            **Reference**: 40% Human, 30% Yeast, 30% EColi
            **Treatment**: 15% Human, 70% Yeast, 15% EColi
            
            Expected: Yeast ↑, Human ↓, EColi ↓
            """)
            composition_ref = {'HUMAN': 40, 'YEAST': 30, 'ECOLI': 30}
            composition_treat = {'HUMAN': 15, 'YEAST': 70, 'ECOLI': 15}
            
        elif preset_option == "Scenario C: Balanced → EColi Heavy":
            st.info("""
            **Reference**: 40% Human, 30% Yeast, 30% EColi
            **Treatment**: 15% Human, 15% Yeast, 70% EColi
            
            Expected: EColi ↑, Human ↓, Yeast ↓
            """)
            composition_ref = {'HUMAN': 40, 'YEAST': 30, 'ECOLI': 30}
            composition_treat = {'HUMAN': 15, 'YEAST': 15, 'ECOLI': 70}
            
        elif preset_option == "Scenario D: Extreme Flip":
            st.info("""
            **Reference**: 70% Human, 15% Yeast, 15% EColi
            **Treatment**: 15% Human, 15% Yeast, 70% EColi
            
            Expected: Human ↓↓, EColi ↑↑, Yeast ~
            """)
            composition_ref = {'HUMAN': 70, 'YEAST': 15, 'ECOLI': 15}
            composition_treat = {'HUMAN': 15, 'YEAST': 15, 'ECOLI': 70}
        
        if composition_ref:
            st.session_state.composition_ref = composition_ref
            st.session_state.composition_treat = composition_treat
    
    with tab2:
        st.markdown("### Custom Composition per Species")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**{reference_group} (Reference)**")
            composition_ref = {}
            for species in species_list:
                pct = st.number_input(
                    f"{species} (%):",
                    value=100/len(species_list),
                    min_value=0.0,
                    max_value=100.0,
                    step=5.0,
                    key=f"ref_{species}",
                    help="Percentage composition"
                )
                composition_ref[species] = pct
            
            total = sum(composition_ref.values())
            if total > 0:
                composition_ref = {k: v*100/total for k, v in composition_ref.items()}
        
        with col2:
            st.markdown(f"**{treatment_group} (Treatment)**")
            composition_treat = {}
            for species in species_list:
                pct = st.number_input(
                    f"{species} (%):",
                    value=100/len(species_list),
                    min_value=0.0,
                    max_value=100.0,
                    step=5.0,
                    key=f"treat_{species}",
                    help="Percentage composition"
                )
                composition_treat[species] = pct
            
            total = sum(composition_treat.values())
            if total > 0:
                composition_treat = {k: v*100/total for k, v in composition_treat.items()}
        
        st.session_state.composition_ref = composition_ref
        st.session_state.composition_treat = composition_treat
    
    # ================================================================
    # CALCULATE LOG2 FOLD CHANGES FROM COMPOSITION
    # ================================================================
    with tab3:
        st.markdown("### 📈 Calculated Log2 Fold Changes from Composition")
        
        comp_ref = st.session_state.get('composition_ref', {})
        comp_treat = st.session_state.get('composition_treat', {})
        
        if comp_ref and comp_treat:
            fc_data = []
            theoretical_fc_dict = {}
            
            for species in species_list:
                ref_pct = comp_ref.get(species, 0)
                treat_pct = comp_treat.get(species, 0)
                
                if ref_pct == 0 and treat_pct == 0:
                    log2fc = 0
                elif treat_pct == 0:
                    log2fc = -10
                elif ref_pct == 0:
                    log2fc = 10
                else:
                    log2fc = np.log2(ref_pct / treat_pct)
                
                theoretical_fc_dict[species] = log2fc
                linear_fc = 2**log2fc if log2fc != 0 else 1.0
                
                # Classify as stable or variable
                is_stable = abs(log2fc) < 0.5
                proteome_type = "Stable" if is_stable else "Variable"
                
                fc_data.append({
                    'Species': species,
                    f'{reference_group} (%)': f"{ref_pct:.1f}",
                    f'{treatment_group} (%)': f"{treat_pct:.1f}",
                    'Log2FC': f"{log2fc:.3f}",
                    'Linear FC': f"{linear_fc:.2f}x",
                    'Proteome': proteome_type,
                    'Direction': '↑' if log2fc > 0 else '↓' if log2fc < 0 else '→'
                })
            
            fc_df = pd.DataFrame(fc_data)
            st.dataframe(fc_df, use_container_width=True)
            
            st.session_state.theoretical_fc_dict = theoretical_fc_dict
            
            # Summary
            n_stable = sum(1 for fc in theoretical_fc_dict.values() if abs(fc) < 0.5)
            n_variable = len(theoretical_fc_dict) - n_stable
            
            st.markdown(f"""
            **Proteome Stratification Summary**:
            - **Stable Proteome**: {n_stable} species with |log2FC| < 0.5
            - **Variable Proteome**: {n_variable} species with |log2FC| ≥ 0.5
            """)
            
            # Visualization
            st.markdown("### 📊 Composition and Fold Change Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Percentage Composition by Condition**")
                
                comp_plot_data = []
                for species in species_list:
                    comp_plot_data.append({
                        'Condition': reference_group,
                        'Species': species,
                        'Percentage': comp_ref.get(species, 0)
                    })
                    comp_plot_data.append({
                        'Condition': treatment_group,
                        'Species': species,
                        'Percentage': comp_treat.get(species, 0)
                    })
                
                comp_plot_df = pd.DataFrame(comp_plot_data)
                
                fig_comp = px.bar(
                    comp_plot_df,
                    x='Condition',
                    y='Percentage',
                    color='Species',
                    title='Percentage Composition per Condition',
                    labels={'Percentage': 'Composition (%)'},
                    barmode='stack',
                    height=500
                )
                fig_comp.update_yaxes(range=[0, 100])
                st.plotly_chart(fig_comp, use_container_width=True)
            
            with col2:
                st.markdown("**Expected Log2 Fold Changes (Proteome Stratified)**")
                
                fc_plot_data = []
                for species, log2fc in theoretical_fc_dict.items():
                    proteome = "Stable" if abs(log2fc) < 0.5 else "Variable"
                    fc_plot_data.append({
                        'Species': species,
                        'Log2FC': log2fc,
                        'Proteome': proteome
                    })
                
                fc_plot_df = pd.DataFrame(fc_plot_data)
                
                color_map = {'Stable': '#95A5A6', 'Variable': '#3498DB'}
                
                fig_fc = px.bar(
                    fc_plot_df,
                    x='Species',
                    y='Log2FC',
                    color='Proteome',
                    color_discrete_map=color_map,
                    title=f'Expected Log2FC ({reference_group} / {treatment_group})',
                    labels={'Log2FC': 'Log2 Fold Change'},
                    height=500
                )
                fig_fc.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Stable/Variable Threshold")
                fig_fc.add_hline(y=-0.5, line_dash="dash", line_color="red")
                fig_fc.add_hline(y=0, line_dash="solid", line_color="black")
                st.plotly_chart(fig_fc, use_container_width=True)
        
        else:
            st.info("👆 Enter composition percentages above to calculate expected fold changes")

st.markdown("---")

# ============================================================================
# 3. STATISTICAL PARAMETERS
# ============================================================================

st.subheader("3️⃣ Statistical Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    fc_threshold = st.slider(
        "Log2 Fold Change Threshold:",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Minimum absolute log2FC for calling significant"
    )
    
    st.markdown(f"""
    **Threshold**: ±{fc_threshold} log2FC  
    **Linear**: {2**fc_threshold:.2f}x
    """)

with col2:
    stable_proteome_threshold = st.slider(
        "Stable Proteome Threshold:",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Max |log2FC| for stable proteome (FP control)"
    )
    
    st.markdown(f"""
    **Stable**: |FC| < {stable_proteome_threshold}  
    **Variable**: |FC| ≥ {stable_proteome_threshold}
    """)

with col3:
    pval_threshold = st.selectbox(
        "FDR Threshold:",
        options=[0.01, 0.05, 0.10],
        index=1,
        format_func=lambda x: f"{x*100:.0f}%",
        help="False Discovery Rate cutoff"
    )
    
    use_fdr = st.checkbox("Use FDR", value=True, help="Benjamini-Hochberg")

st.markdown("---")

# ============================================================================
# 4. RUN ANALYSIS
# ============================================================================

st.subheader("4️⃣ Run Analysis")

if st.button("🚀 Run Analysis", type="primary"):
    with st.spinner("Running limma analysis..."):
        
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
        st.session_state.dea_stable_threshold = stable_proteome_threshold
        st.session_state.dea_pval_threshold = pval_threshold
        st.session_state.theoretical_fc_dict = st.session_state.get('theoretical_fc_dict', {})
        
        st.success("✅ Analysis complete!")

# ============================================================================
# 5. RESULTS VISUALIZATION
# ============================================================================

if 'dea_results' in st.session_state:
    results = st.session_state.dea_results
    reference_group = st.session_state.dea_ref_group
    treatment_group = st.session_state.dea_treat_group
    fc_threshold = st.session_state.dea_fc_threshold
    stable_proteome_threshold = st.session_state.dea_stable_threshold
    pval_threshold = st.session_state.dea_pval_threshold
    theoretical_fc_dict = st.session_state.get('theoretical_fc_dict', {})
    
    st.markdown("---")
    st.subheader("5️⃣ Results Summary")
    
    n_total = len(results)
    n_up = (results['regulation'] == 'up').sum()
    n_down = (results['regulation'] == 'down').sum()
    n_ns = (results['regulation'] == 'not_significant').sum()
    n_tested = n_total - (results['regulation'] == 'not_tested').sum()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", f"{n_total:,}")
    col2.metric("Up", f"{n_up:,}", delta=f"{n_up/n_tested*100:.1f}%")
    col3.metric("Down", f"{n_down:,}", delta=f"{n_down/n_tested*100:.1f}%")
    col4.metric("NS", f"{n_ns:,}")
    
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
            'log2fc': f'Log2 Fold Change',
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
    st.markdown("### 📈 MA Plot")
    
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
        title='Mean vs Log2 Fold Change',
        labels={
            'A': 'Mean Log2 Intensity',
            'M': f'Log2 Fold Change'
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
    st.markdown("### 📋 Top Differentially Abundant Proteins")
    
    sig_results = results[results['regulation'].isin(['up', 'down'])].copy()
    sig_results = sig_results.sort_values('fdr')
    
    display_cols = ['log2fc', 'mean_g1', 'mean_g2', 't_stat', 'pvalue', 'fdr', 'regulation', 'species']
    display_df = sig_results[display_cols].head(50)
    
    st.dataframe(display_df.round(4), use_container_width=True)
    
    # ================================================================
    # VALIDATION METRICS
    # ================================================================
    if len(theoretical_fc_dict) > 0:
        st.markdown("---")
        st.subheader("6️⃣ Spike-in Validation Metrics")
        
        species_mapping = dict(zip(results.index, results['species']))
        
        # === STABLE PROTEOME: FALSE POSITIVE RATE ===
        st.markdown("### 📊 Stable Proteome: False Positive Rate Control")
        st.markdown(f"Proteins with |theoretical log2FC| < {stable_proteome_threshold} (should NOT be called significant)")
        
        overall_stable, species_stable = calculate_error_metrics_stable_proteome(
            results,
            theoretical_fc_dict,
            species_mapping,
            stable_proteome_threshold,
            pval_threshold
        )
        
        if len(overall_stable) > 0:
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "False Positive Rate",
                f"{overall_stable['False_Positive_Rate']:.1%}",
                help=f"CI: {overall_stable['FPR_CI'][0]:.1%}-{overall_stable['FPR_CI'][1]:.1%}"
            )
            col2.metric("True Negative Rate", f"{overall_stable['True_Negative_Rate']:.1%}")
            col3.metric("Stable Proteins Tested", f"{overall_stable['Total_Stable_Proteins']:,}")
            
            if len(species_stable) > 0:
                st.markdown("#### Per-Species FP Rate")
                st.dataframe(species_stable.round(3), use_container_width=True)
        
        # === VARIABLE PROTEOME: SENSITIVITY/SPECIFICITY ===
        st.markdown("---")
        st.markdown("### 🎯 Variable Proteome: Sensitivity/Specificity")
        st.markdown(f"Proteins with |theoretical log2FC| ≥ {stable_proteome_threshold} (should be detected)")
        
        overall_variable, species_variable = calculate_error_metrics_variable_proteins(
            results,
            theoretical_fc_dict,
            species_mapping,
            stable_proteome_threshold,
            fc_threshold,
            pval_threshold
        )
        
        if len(overall_variable) > 0:
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric(
                "Sensitivity",
                f"{overall_variable['Sensitivity']:.1%}",
                help=f"CI: {overall_variable['Sensitivity_CI'][0]:.1%}-{overall_variable['Sensitivity_CI'][1]:.1%}"
            )
            col2.metric(
                "Specificity",
                f"{overall_variable['Specificity']:.1%}",
                help=f"CI: {overall_variable['Specificity_CI'][0]:.1%}-{overall_variable['Specificity_CI'][1]:.1%}"
            )
            col3.metric("Precision", f"{overall_variable['Precision']:.1%}")
            col4.metric("Variable Proteins", f"{overall_variable['Total_Variable_Proteins']:,}")
            
            # Confusion matrix
            st.markdown("#### Confusion Matrix (Variable Proteome)")
            cm_data = {
                'True Positives': overall_variable['True_Positives'],
                'False Positives': overall_variable['False_Positives'],
                'True Negatives': overall_variable['True_Negatives'],
                'False Negatives': overall_variable['False_Negatives']
            }
            cm_df = pd.DataFrame([cm_data])
            st.dataframe(cm_df, use_container_width=True)
            
            if len(species_variable) > 0:
                st.markdown("#### Per-Species Metrics (Variable Proteome)")
                display_cols = ['Species', 'N', 'Theo_FC', 'RMSE', 'MAE', 'Bias', 'Detection_Rate']
                st.dataframe(species_variable[[c for c in display_cols if c in species_variable.columns]].round(3), use_container_width=True)
                
                # RMSE plot
                if 'RMSE' in species_variable.columns:
                    st.markdown("#### RMSE by Species (with 95% CI)")
                    
                    fig_rmse = go.Figure()
                    
                    for idx, row in species_variable.iterrows():
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
                        title='RMSE with 95% CI (Variable Proteome)',
                        xaxis_title='Species',
                        yaxis_title='RMSE',
                        height=500
                    )
                    
                    st.plotly_chart(fig_rmse, use_container_width=True)
    
    # ================================================================
    # DOWNLOAD
    # ================================================================
    st.markdown("---")
    st.subheader("💾 Export Results")
    
    csv = results.to_csv()
    st.download_button(
        label="📥 Download Full Results",
        data=csv,
        file_name=f"dea_{reference_group}_vs_{treatment_group}.csv",
        mime="text/csv"
    )
    
    st.success(f"✅ Found {n_up + n_down:,} differentially abundant proteins")

else:
    st.info("👆 Configure parameters and click 'Run Analysis'")
