"""
pages/6_Differential_Abundance.py
Limma-style empirical Bayes moderated t-test with comprehensive validation.
Unified FP definition: p<0.01 AND |log2fc - expected| > ±0.58 for all species
Production-ready with spike-in validation and misregulation tracking
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
from scipy import stats
from scipy.stats import t as t_dist, ttest_ind
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(str(Path(__file__).parent.parent))

# ============================================================================
# COLOR SCHEME
# ============================================================================

SPECIES_COLORS = {
    "HUMAN": "#2ecc71",   # Green
    "YEAST": "#e67e22",   # Orange
    "ECOLI": "#9b59b6",   # Purple
}

# ============================================================================
# LIMMA-STYLE EMPIRICAL BAYES MODERATION
# ============================================================================

def fit_limma_empirical_bayes(
    variances: np.ndarray,
    df_residual: int,
    proportion: float = 0.01
) -> Tuple[float, float]:
    """
    Fit limma-style empirical Bayes prior for variance moderation.
    
    Args:
        variances: Array of protein-wise variances
        df_residual: Residual degrees of freedom per protein
        proportion: Proportion of genes expected to be differentially expressed
    
    Returns:
        (s0_squared, df_prior): Prior variance and prior degrees of freedom
    """
    variances = variances[~np.isnan(variances)]
    
    if len(variances) < 2:
        return 0.0, 0.0
    
    # Trim extreme values (winsorize at 1% and 99%)
    var_sorted = np.sort(variances)
    n = len(var_sorted)
    lower_idx = int(n * 0.01)
    upper_idx = int(n * 0.99)
    var_trimmed = var_sorted[lower_idx:upper_idx]
    
    # Estimate prior variance (s0^2) as median of trimmed variances
    s0_squared = np.median(var_trimmed)
    
    # Estimate prior degrees of freedom using method of moments
    # Var(s^2) = 2*s0^4 / (df + df_prior)
    var_of_vars = np.var(var_trimmed)
    
    if var_of_vars > 0 and s0_squared > 0:
        df_prior = max(1.0, (2 * s0_squared**2 / var_of_vars) - df_residual)
    else:
        df_prior = 1.0
    
    return s0_squared, df_prior


def moderate_tstatistics(
    log2fc: np.ndarray,
    variances: np.ndarray,
    n1: int,
    n2: int,
    s0_squared: float,
    df_prior: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate moderated t-statistics using empirical Bayes.
    
    Args:
        log2fc: Log2 fold-changes
        variances: Pooled variances per protein
        n1, n2: Sample sizes
        s0_squared: Prior variance
        df_prior: Prior degrees of freedom
    
    Returns:
        (moderated_t, moderated_p, moderated_df): Arrays of moderated statistics
    """
    # Degrees of freedom
    df_residual = n1 + n2 - 2
    df_total = df_residual + df_prior
    
    # Moderated variance (shrink towards prior)
    s_moderated_squared = (df_residual * variances + df_prior * s0_squared) / df_total
    
    # Standard error of log2fc
    se = np.sqrt(s_moderated_squared * (1/n1 + 1/n2))
    
    # Moderated t-statistic
    moderated_t = log2fc / se
    
    # P-values from t-distribution with moderated df
    moderated_p = 2 * (1 - t_dist.cdf(np.abs(moderated_t), df_total))
    
    moderated_df = np.full_like(moderated_t, df_total)
    
    return moderated_t, moderated_p, moderated_df


# ============================================================================
# STATISTICAL HELPERS
# ============================================================================

def perform_limma_test(
    df: pd.DataFrame,
    group1_cols: List[str],
    group2_cols: List[str],
    min_valid: int = 2,
) -> pd.DataFrame:
    """
    Perform Limma-style moderated t-test with empirical Bayes.
    Convention: log2FC = mean(group1) - mean(group2)
    """
    results = []
    variances_list = []
    log2fc_list = []
    protein_ids = []
    
    n1 = len(group1_cols)
    n2 = len(group2_cols)
    
    # First pass: collect statistics
    for protein_id in df.index:
        row = df.loc[protein_id]
        g1_vals = row[group1_cols].dropna()
        g2_vals = row[group2_cols].dropna()
        
        if len(g1_vals) < min_valid or len(g2_vals) < min_valid:
            continue
        
        mean_g1 = g1_vals.mean()
        mean_g2 = g2_vals.mean()
        log2fc = mean_g1 - mean_g2
        
        # Pooled variance
        var_g1 = g1_vals.var(ddof=1) if len(g1_vals) > 1 else 0
        var_g2 = g2_vals.var(ddof=1) if len(g2_vals) > 1 else 0
        
        n1_actual = len(g1_vals)
        n2_actual = len(g2_vals)
        
        if n1_actual + n2_actual > 2:
            pooled_var = ((n1_actual - 1) * var_g1 + (n2_actual - 1) * var_g2) / (n1_actual + n2_actual - 2)
        else:
            pooled_var = 0
        
        variances_list.append(pooled_var)
        log2fc_list.append(log2fc)
        protein_ids.append(protein_id)
        
        results.append({
            "protein_id": protein_id,
            "log2fc": log2fc,
            "mean_g1": mean_g1,
            "mean_g2": mean_g2,
            "variance": pooled_var,
            "n_g1": n1_actual,
            "n_g2": n2_actual,
        })
    
    if len(results) == 0:
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results).set_index("protein_id")
    
    # Fit empirical Bayes prior
    variances_array = np.array(variances_list)
    log2fc_array = np.array(log2fc_list)
    
    df_residual = n1 + n2 - 2
    s0_squared, df_prior = fit_limma_empirical_bayes(variances_array, df_residual)
    
    # Calculate moderated statistics
    moderated_t, moderated_p, moderated_df = moderate_tstatistics(
        log2fc_array,
        variances_array,
        n1, n2,
        s0_squared,
        df_prior
    )
    
    # Add moderated statistics
    results_df["moderated_t"] = moderated_t
    results_df["pvalue"] = moderated_p
    results_df["moderated_df"] = moderated_df
    results_df["s0_squared"] = s0_squared
    results_df["df_prior"] = df_prior
    
    # Also calculate regular Welch's t-test for comparison
    welch_pvals = []
    for protein_id in results_df.index:
        row = df.loc[protein_id]
        g1_vals = row[group1_cols].dropna()
        g2_vals = row[group2_cols].dropna()
        
        try:
            _, p_welch = ttest_ind(g1_vals, g2_vals, equal_var=False)
            welch_pvals.append(p_welch)
        except:
            welch_pvals.append(np.nan)
    
    results_df["pvalue_welch"] = welch_pvals
    
    # FDR correction (Benjamini-Hochberg)
    pvals = results_df["pvalue"].dropna().sort_values()
    n = len(pvals)
    
    if n > 0:
        ranks = np.arange(1, n + 1)
        fdr_vals = pvals.values * n / ranks
        fdr_vals = np.minimum.accumulate(fdr_vals[::-1])[::-1]
        fdr_vals = np.minimum(fdr_vals, 1.0)
        fdr_dict = dict(zip(pvals.index, fdr_vals))
        results_df["fdr"] = results_df.index.map(fdr_dict)
    else:
        results_df["fdr"] = np.nan
    
    # FDR for Welch (for comparison)
    pvals_welch = results_df["pvalue_welch"].dropna().sort_values()
    if len(pvals_welch) > 0:
        ranks_welch = np.arange(1, len(pvals_welch) + 1)
        fdr_welch = pvals_welch.values * len(pvals_welch) / ranks_welch
        fdr_welch = np.minimum.accumulate(fdr_welch[::-1])[::-1]
        fdr_welch = np.minimum(fdr_welch, 1.0)
        fdr_welch_dict = dict(zip(pvals_welch.index, fdr_welch))
        results_df["fdr_welch"] = results_df.index.map(fdr_welch_dict)
    else:
        results_df["fdr_welch"] = np.nan
    
    return results_df


def classify_regulation(
    log2fc: float,
    pvalue: float,
    fc_threshold: float = 0.0,
    pval_threshold: float = 0.05,
) -> str:
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


def calculate_asymmetry(values: np.ndarray, expected: float) -> float:
    """
    Calculate asymmetry around expected value.
    Asymmetry = median(observed) / expected
    """
    if len(values) == 0 or expected == 0:
        return np.nan
    median_obs = np.median(values)
    return abs(median_obs / expected)


def compute_species_metrics(
    results_df: pd.DataFrame,
    true_fc_dict: Dict[str, float],
    species_col_series: pd.Series,
    stable_thr: float = 0.5,
    fc_tolerance: float = 0.58,
    p_threshold: float = 0.01,
) -> Tuple[Dict, pd.DataFrame, Dict, pd.DataFrame, Dict, Dict, pd.DataFrame]:
    """
    Calculate metrics for variable and stable proteomes + asymmetry.
    
    FALSE POSITIVES (UNIFIED DEFINITION FOR ALL SPECIES):
    FP = p < p_threshold AND |log2fc - expected| > fc_tolerance
         (called significant but OUTSIDE ±0.58 from theoretical)
    
    Returns:
        (variable_overall, variable_per_species, stable_overall, stable_per_species,
         asymmetry_dict, error_dict, fp_var_per_species)
    """
    res = results_df.copy()
    res["species"] = res.index.map(species_col_series)
    res["true_log2fc"] = res["species"].map(true_fc_dict)
    
    res = res[res["regulation"] != "not_tested"].copy()
    res = res.dropna(subset=["true_log2fc", "species"])
    
    if res.empty:
        return {}, pd.DataFrame(), {}, pd.DataFrame(), {}, {}, pd.DataFrame()
    
    # === ASYMMETRY CALCULATION ===
    asymmetry_dict = {}
    for sp in res["species"].unique():
        sp_df = res[res["species"] == sp].copy()
        expected_fc = true_fc_dict.get(sp, 0.0)
        if abs(expected_fc) >= stable_thr:
            asym = calculate_asymmetry(sp_df["log2fc"].values, expected_fc)
            asymmetry_dict[sp] = asym
    
    # === ERROR TRACKING ===
    error_dict = {}
    
    # === VARIABLE PROTEOME (|expected FC| >= stable_thr) ===
    var_df = res[np.abs(res["true_log2fc"]) >= stable_thr].copy()
    
    var_overall = {}
    var_species_rows = []
    
    if not var_df.empty:
        var_df["observed_regulated"] = var_df["regulation"].isin(["up", "down"])
        var_df["true_regulated"] = np.abs(var_df["true_log2fc"]) >= stable_thr
        var_df["significant"] = var_df["pvalue"] < p_threshold
        var_df["within_tolerance"] = np.abs(var_df["log2fc"] - var_df["true_log2fc"]) <= fc_tolerance
        
        # CONFUSION MATRIX (for variable species)
        tp = int((var_df["true_regulated"] & var_df["observed_regulated"]).sum())
        fn = int((var_df["true_regulated"] & ~var_df["observed_regulated"]).sum())
        tn = int((~var_df["true_regulated"] & ~var_df["observed_regulated"]).sum())
        fp = int((~var_df["true_regulated"] & var_df["observed_regulated"]).sum())
        
        # Metrics
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        var_overall = {
            "Total": len(var_df),
            "TP": tp,
            "FN": fn,
            "TN": tn,
            "FP": fp,
            "Sensitivity": sens,
            "Specificity": spec,
            "Precision": prec,
        }
        
        # Per-species metrics (variable)
        for sp in var_df["species"].unique():
            sp_df = var_df[var_df["species"] == sp].copy()
            theo = true_fc_dict.get(sp, 0.0)
            error_log2 = sp_df["log2fc"] - theo
            mae_log2 = error_log2.abs().mean()
            
            if theo != 0:
                mape = (error_log2.abs() / abs(theo) * 100).mean()
            else:
                mape = np.nan
            
            var_species_rows.append({
                "Species": sp,
                "N": len(sp_df),
                "Expected_log2FC": f"{theo:.2f}",
                "RMSE": f"{np.sqrt((error_log2**2).mean()):.3f}",
                "MAE": f"{mae_log2:.3f}",
                "MAPE_%": f"{mape:.1f}" if not np.isnan(mape) else "N/A",
                "Bias": f"{error_log2.mean():.3f}",
                "Detection_%": f"{sp_df['observed_regulated'].mean()*100:.1f}",
            })
    
    # === STABLE PROTEOME (|expected FC| < stable_thr) ===
    stab_df = res[np.abs(res["true_log2fc"]) < stable_thr].copy()
    
    stab_overall = {}
    stab_species_rows = []
    
    if not stab_df.empty:
        stab_df["significant"] = stab_df["pvalue"] < p_threshold
        stab_df["outside_tolerance"] = np.abs(stab_df["log2fc"] - stab_df["true_log2fc"]) > fc_tolerance
        
        # UNIFIED FP: p<threshold AND |error| > ±fc_tolerance
        true_fp = int((stab_df["significant"] & stab_df["outside_tolerance"]).sum())
        tn = int((~stab_df["significant"] | ~stab_df["outside_tolerance"]).sum())
        
        total = len(stab_df)
        fpr = true_fp / total if total > 0 else 0.0
        
        stab_overall = {"Total": total, "FP": true_fp, "TN": tn, "FPR": fpr}
        
        # Per-species metrics (stable)
        for sp in stab_df["species"].unique():
            sp_df = stab_df[stab_df["species"] == sp].copy()
            sp_df["significant"] = sp_df["pvalue"] < p_threshold
            sp_df["outside_tolerance"] = np.abs(sp_df["log2fc"] - sp_df["true_log2fc"]) > fc_tolerance
            
            fp_s = int((sp_df["significant"] & sp_df["outside_tolerance"]).sum())
            tn_s = int((~sp_df["significant"] | ~sp_df["outside_tolerance"]).sum())
            mae_log2 = sp_df["log2fc"].abs().mean()
            
            stab_species_rows.append({
                "Species": sp,
                "N": len(sp_df),
                "FP": fp_s,
                "TN": tn_s,
                "FPR_%": f"{fp_s/len(sp_df)*100:.1f}" if len(sp_df) > 0 else "0.0",
                "MAE": f"{mae_log2:.3f}",
            })
            
            error_dict[f"FP_{sp}"] = fp_s
    
    # === MISREGULATED PROTEINS (Variable species with wrong magnitude) ===
    fp_var_species_rows = []
    
    if not var_df.empty:
        for sp in var_df["species"].unique():
            if sp != "HUMAN":  # Only for variable species
                sp_df = var_df[var_df["species"] == sp].copy()
                sp_df["significant"] = sp_df["pvalue"] < p_threshold
                sp_df["outside_tolerance"] = np.abs(sp_df["log2fc"] - sp_df["true_log2fc"]) > fc_tolerance
                
                # MISREGULATED: p<threshold AND outside ±fc_tolerance (wrong magnitude)
                misreg_s = int((sp_df["significant"] & sp_df["outside_tolerance"]).sum())
                accurate_s = int((sp_df["significant"] & ~sp_df["outside_tolerance"]).sum())
                total_var_s = len(sp_df)
                
                fp_var_species_rows.append({
                    "Species": sp,
                    "Total_Proteins": total_var_s,
                    "Accurate": accurate_s,
                    "Misregulated": misreg_s,
                    "Accuracy_%": f"{accurate_s/total_var_s*100:.1f}" if total_var_s > 0 else "0.0",
                })
                
                error_dict[f"Misreg_{sp}"] = misreg_s
    
    return (
        var_overall,
        pd.DataFrame(var_species_rows),
        stab_overall,
        pd.DataFrame(stab_species_rows),
        asymmetry_dict,
        error_dict,
        pd.DataFrame(fp_var_species_rows),
    )


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(page_title="Differential Abundance", page_icon="🔬", layout="wide")
st.title("🔬 Differential Abundance Analysis (Limma-Style)")
st.markdown("Empirical Bayes moderated t-test with spike-in validation")
st.markdown("---")

# ============================================================================
# DATA VALIDATION
# ============================================================================

if "df_imputed" not in st.session_state or st.session_state.df_imputed is None:
    st.error("❌ No imputed data. Complete **Missing Value Imputation** first.")
    st.stop()

df = st.session_state.df_imputed.copy()
numeric_cols: List[str] = st.session_state.numeric_cols
species_col: str = st.session_state.species_col
sample_to_condition: Dict[str, str] = st.session_state.sample_to_condition

conditions = sorted(set(sample_to_condition.values()))
cond_samples: Dict[str, List[str]] = {}
for s, c in sample_to_condition.items():
    if s in numeric_cols:
        cond_samples.setdefault(c, []).append(s)

st.info(f"📊 **Data**: {df.shape[0]:,} proteins × {len(numeric_cols)} samples | **Conditions**: {', '.join(conditions)}")

# ============================================================================
# 1. COMPARISON SETUP
# ============================================================================

st.subheader("1️⃣ Comparison Setup")

col1, col2 = st.columns(2)
with col1:
    ref_cond = st.selectbox("Condition A (reference)", options=conditions, index=0)
with col2:
    available_treat = [c for c in conditions if c != ref_cond]
    treat_cond = st.selectbox(
        "Condition B (treatment)",
        options=available_treat,
        index=0 if len(available_treat) > 0 else 0
    )

if ref_cond == treat_cond:
    st.error("❌ Choose two different conditions.")
    st.stop()

ref_samples = cond_samples[ref_cond]
treat_samples = cond_samples[treat_cond]

st.markdown(f"**Log2FC convention**: log2({ref_cond} / {treat_cond}) — positive = higher in {ref_cond}")
st.markdown("---")

# ============================================================================
# 2. SPIKE-IN COMPOSITION
# ============================================================================

st.subheader("2️⃣ Spike-in Composition (Optional)")

use_comp = st.checkbox("Provide % composition per species", value=False)

theoretical_fc_temp: Dict[str, float] = {}
species_values = sorted([s for s in df[species_col].unique() if isinstance(s, str) and s != 'Unknown'])

if use_comp:
    st.markdown("Enter % composition (auto-normalized to 100% per condition)")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{ref_cond} (A)**")
        comp_a = {}
        for sp in species_values:
            val = st.number_input(
                f"{sp} (%) in {ref_cond}",
                min_value=0.0,
                max_value=100.0,
                value=100.0/max(len(species_values), 1),
                step=5.0,
                key=f"a_{sp}"
            )
            comp_a[sp] = val
        ta = sum(comp_a.values()) or 1.0
        comp_a = {k: v*100/ta for k, v in comp_a.items()}
    
    with c2:
        st.markdown(f"**{treat_cond} (B)**")
        comp_b = {}
        for sp in species_values:
            val = st.number_input(
                f"{sp} (%) in {treat_cond}",
                min_value=0.0,
                max_value=100.0,
                value=100.0/max(len(species_values), 1),
                step=5.0,
                key=f"b_{sp}"
            )
            comp_b[sp] = val
        tb = sum(comp_b.values()) or 1.0
        comp_b = {k: v*100/tb for k, v in comp_b.items()}
    
    rows = []
    for sp in species_values:
        pa, pb = comp_a.get(sp, 0.0), comp_b.get(sp, 0.0)
        
        if pa == 0 and pb == 0:
            log2fc = 0.0
        elif pb == 0:
            log2fc = 10.0
        elif pa == 0:
            log2fc = -10.0
        else:
            log2fc = float(np.log2(pa/pb))
        
        theoretical_fc_temp[sp] = log2fc
        rows.append({
            "Species": sp,
            f"{ref_cond} (%)": f"{pa:.1f}",
            f"{treat_cond} (%)": f"{pb:.1f}",
            "Log2FC": f"{log2fc:.3f}",
            "Linear_FC": f"{2**log2fc:.2f}x"
        })
    
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    
    if st.button("💾 Save Expected FC", type="primary"):
        st.session_state.dea_theoretical_fc = theoretical_fc_temp.copy()
        st.success(f"✅ Saved for {len(theoretical_fc_temp)} species!")
    
    saved_fc = st.session_state.get('dea_theoretical_fc', {})
    if saved_fc:
        st.info(f"✓ **Saved**: {', '.join(f'{k}={v:.2f}' for k, v in saved_fc.items())}")
    else:
        st.warning("⚠️ Not saved yet.")

st.markdown("---")

# ============================================================================
# 3. STATISTICAL SETTINGS
# ============================================================================

st.subheader("3️⃣ Statistical Settings")

c1, c2, c3 = st.columns(3)

with c1:
    p_thr = st.selectbox(
        "Significance threshold",
        options=[0.001, 0.01, 0.05],
        index=1,
        format_func=lambda x: f"p < {x}"
    )

with c2:
    use_fdr = st.checkbox("Use FDR correction (BH)", value=True)

with c3:
    test_method = st.selectbox(
        "Test method",
        options=["Limma (EB)", "Welch's t-test", "Both"],
        index=0
    )

stable_thr = 0.5
fc_tolerance = 0.58

st.caption(
    f"**Definitions**: Stable = |expected log2FC| < {stable_thr} · "
    f"**FP (unified)** = p < {p_thr} AND |log2fc - expected| > ±{fc_tolerance}"
)
st.markdown("---")

# ============================================================================
# 4. RUN ANALYSIS
# ============================================================================

st.subheader("4️⃣ Run Analysis")

if st.button("🚀 Run Differential Abundance Test", type="primary"):
    with st.spinner("⏳ Running Limma-style moderated t-test..."):
        
        # Log2 transform if needed
        df_num = df[numeric_cols]
        if (df_num > 50).any().any():
            df_log2 = np.log2(df_num + 1.0)
        else:
            df_log2 = df_num.copy()
        
        # Run Limma test
        results = perform_limma_test(df_log2, ref_samples, treat_samples)
        
        if len(results) == 0:
            st.error("❌ No proteins passed filtering.")
            st.stop()
        
        # Choose test column
        test_col = "fdr" if use_fdr else "pvalue"
        
        # Classification
        results["regulation"] = results.apply(
            lambda row: classify_regulation(row["log2fc"], row[test_col], 0.0, p_thr),
            axis=1
        )
        
        results["neg_log10_p"] = -np.log10(results[test_col].replace(0, 1e-300))
        results["species"] = results.index.map(df[species_col])
        
        # Save to session state
        st.session_state.dea_results = results
        st.session_state.dea_ref = ref_cond
        st.session_state.dea_treat = treat_cond
        st.session_state.dea_p_thr = p_thr
        st.session_state.dea_use_fdr = use_fdr
        st.session_state.dea_test_method = test_method
    
    st.success("✅ Analysis complete!")

# ============================================================================
# 5. RESULTS
# ============================================================================

if "dea_results" in st.session_state:
    res = st.session_state.dea_results
    ref_cond = st.session_state.dea_ref
    treat_cond = st.session_state.dea_treat
    p_thr = st.session_state.dea_p_thr
    use_fdr = st.session_state.get('dea_use_fdr', True)
    test_method = st.session_state.get('dea_test_method', 'Limma (EB)')
    theoretical_fc = st.session_state.get('dea_theoretical_fc', {})
    
    st.markdown("---")
    st.subheader("5️⃣ Results Overview")
    
    # Basic stats
    n_total = len(res)
    n_quant = int((res["regulation"] != "not_tested").sum())
    quant_rate = n_quant / n_total * 100 if n_total > 0 else 0
    
    # Calculate validation metrics
    sens_pct = 0.0
    spec_pct = 0.0
    true_positives = 0
    de_fdr = 0.0
    
    if theoretical_fc:
        species_series = df[species_col]
        
        res_all = res[res["regulation"] != "not_tested"].copy()
        res_all["species"] = res_all.index.map(species_series)
        res_all["true_log2fc"] = res_all["species"].map(theoretical_fc)
        res_all = res_all.dropna(subset=["true_log2fc"])
        
        if not res_all.empty:
            res_all["true_regulated"] = np.abs(res_all["true_log2fc"]) >= stable_thr
            res_all["observed_regulated"] = res_all["regulation"].isin(["up", "down"])
            
            tp = int((res_all["true_regulated"] & res_all["observed_regulated"]).sum())
            fn = int((res_all["true_regulated"] & ~res_all["observed_regulated"]).sum())
            tn = int((~res_all["true_regulated"] & ~res_all["observed_regulated"]).sum())
            fp = int((~res_all["true_regulated"] & res_all["observed_regulated"]).sum())
            
            sens_pct = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
            spec_pct = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
            true_positives = tp
            de_fdr = fp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Quantifiable Proteins", f"{n_quant:,}")
        st.caption(f"{quant_rate:.1f}% of total")
    
    with col2:
        st.metric("Sensitivity", f"{sens_pct:.1f}%")
        st.caption("Detection rate")
    
    with col3:
        st.metric("Specificity", f"{spec_pct:.1f}%")
        st.caption("True negative rate")
    
    with col4:
        st.metric("True Positives", f"{true_positives:,}")
        st.caption(f"deFDR: {de_fdr:.2f}%")
    
    # Asymmetry metrics
    if theoretical_fc:
        asym_dict = {}
        for sp in theoretical_fc.keys():
            sp_df = res[res["species"] == sp].dropna(subset=["log2fc"])
            if len(sp_df) > 0 and abs(theoretical_fc[sp]) >= stable_thr:
                asym = calculate_asymmetry(sp_df["log2fc"].values, theoretical_fc[sp])
                asym_dict[sp] = asym
        
        if asym_dict:
            st.markdown("**Asymmetry**: " + ", ".join([f"{k} {v:.2f}" for k, v in asym_dict.items()]))
    
    st.markdown("---")
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    
    # === PLOT 1: FACETED MA PLOT ===
    st.markdown("### 📊 MA Plot (Faceted by Species)")
    
    ma = res[res["regulation"] != "not_tested"].copy()
    ma["A"] = (ma["mean_g1"] + ma["mean_g2"]) / 2
    ma = ma.dropna(subset=['A', 'log2fc', 'species'])
    
    species_list = sorted(ma["species"].unique())
    fig_facet = make_subplots(
        rows=1,
        cols=len(species_list),
        subplot_titles=species_list,
        shared_yaxes=True,
        horizontal_spacing=0.05
    )
    
    for i, sp in enumerate(species_list, 1):
        sp_data = ma[ma["species"] == sp]
        color = SPECIES_COLORS.get(sp, "#95a5a6")
        
        sp_sig = sp_data[sp_data["regulation"] != "not_significant"]
        sp_nonsig = sp_data[sp_data["regulation"] == "not_significant"]
        
        # Non-significant (grayed)
        fig_facet.add_trace(
            go.Scatter(
                x=sp_nonsig["A"],
                y=sp_nonsig["log2fc"],
                mode='markers',
                marker=dict(size=3, color="lightgray", opacity=0.3),
                name=f"{sp} (ns)",
                showlegend=False,
                hovertemplate=f"{sp} (ns)<br>A=%{{x:.2f}}<br>log2FC=%{{y:.3f}}<extra></extra>"
            ),
            row=1, col=i
        )
        
        # Significant
        sp_up = sp_sig[sp_sig["regulation"] == "up"]
        sp_down = sp_sig[sp_sig["regulation"] == "down"]
        
        fig_facet.add_trace(
            go.Scatter(
                x=sp_up["A"],
                y=sp_up["log2fc"],
                mode='markers',
                marker=dict(size=4, color=color, opacity=0.8, symbol='circle'),
                name=f"{sp} (↑)",
                showlegend=(i == 1),
                hovertemplate=f"{sp} ↑<br>A=%{{x:.2f}}<br>log2FC=%{{y:.3f}}<extra></extra>"
            ),
            row=1, col=i
        )
        
        fig_facet.add_trace(
            go.Scatter(
                x=sp_down["A"],
                y=sp_down["log2fc"],
                mode='markers',
                marker=dict(size=4, color=color, opacity=0.8, symbol='diamond'),
                name=f"{sp} (↓)",
                showlegend=(i == 1),
                hovertemplate=f"{sp} ↓<br>A=%{{x:.2f}}<br>log2FC=%{{y:.3f}}<extra></extra>"
            ),
            row=1, col=i
        )
        
        # Expected FC lines
        if theoretical_fc and sp in theoretical_fc:
            expected_fc = theoretical_fc[sp]
            
            fig_facet.add_hline(
                y=expected_fc,
                line_dash="dash",
                line_color=color,
                line_width=2,
                row=1, col=i
            )
            
            fig_facet.add_hline(
                y=expected_fc + fc_tolerance,
                line_dash="dot",
                line_color=color,
                line_width=1,
                opacity=0.3,
                row=1, col=i
            )
            fig_facet.add_hline(
                y=expected_fc - fc_tolerance,
                line_dash="dot",
                line_color=color,
                line_width=1,
                opacity=0.3,
                row=1, col=i
            )
        
        # Zero line
        fig_facet.add_hline(y=0, line_color="red", line_width=1, opacity=0.5, row=1, col=i)
    
    fig_facet.update_xaxes(title_text="Mean Intensity", row=1, col=2)
    fig_facet.update_yaxes(title_text=f"log2({ref_cond}:{treat_cond})", row=1, col=1)
    fig_facet.update_layout(height=500, showlegend=False)
    
    st.plotly_chart(fig_facet, use_container_width=True)
    
    # === PLOT 2: COMBINED MA PLOT ===
    st.markdown("### 📈 MA Plot (Combined)")
    
    ma_plot_data = ma.copy()
    ma_plot_data["Status"] = ma_plot_data.apply(
        lambda x: "Not Sig." if x["regulation"] == "not_significant" else ("↑ Up" if x["regulation"] == "up" else "↓ Down"),
        axis=1
    )
    
    fig_ma = px.scatter(
        ma_plot_data,
        x="A",
        y="log2fc",
        color="species",
        symbol="Status",
        color_discrete_map=SPECIES_COLORS,
        category_orders={"Status": ["Not Sig.", "↓ Down", "↑ Up"]},
        opacity=0.7,
        labels={"A": "Mean Intensity", "log2fc": f"log2({ref_cond}:{treat_cond})"},
        height=600,
    )
    
    # Gray out non-significant
    for trace in fig_ma.data:
        if "Not Sig." in trace.name:
            trace.marker.opacity = 0.2
            trace.marker.color = "lightgray"
    
    fig_ma.add_hline(y=0.0, line_color="red", line_width=1, opacity=0.5)
    
    if theoretical_fc:
        for species, expected_fc in theoretical_fc.items():
            color = SPECIES_COLORS.get(species, "#95a5a6")
            
            fig_ma.add_hline(
                y=expected_fc,
                line_dash="dash",
                line_width=2,
                line_color=color,
                opacity=0.7,
                annotation_text=f"{species}",
                annotation_position="right",
            )
            
            fig_ma.add_hline(y=expected_fc + fc_tolerance, line_dash="dot", line_width=1, line_color=color, opacity=0.2)
            fig_ma.add_hline(y=expected_fc - fc_tolerance, line_dash="dot", line_width=1, line_color=color, opacity=0.2)
    
    st.plotly_chart(fig_ma, use_container_width=True)
    
    # === PLOT 3: DENSITY PLOT ===
    st.markdown("### 📊 Log2FC Density Plot")
    
    density_data = res[res["regulation"] != "not_tested"].dropna(subset=["log2fc", "species"])
    
    fig_density = go.Figure()
    
    for sp in sorted(density_data["species"].unique()):
        sp_data = density_data[density_data["species"] == sp]["log2fc"]
        color = SPECIES_COLORS.get(sp, "#95a5a6")
        
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(sp_data)
        x_range = np.linspace(sp_data.min(), sp_data.max(), 200)
        density = kde(x_range)
        
        fig_density.add_trace(go.Scatter(
            x=x_range,
            y=density,
            mode='lines',
            name=sp,
            fill='tozeroy',
            opacity=0.6,
            line=dict(width=2, color=color),
            fillcolor=color
        ))
        
        if theoretical_fc and sp in theoretical_fc:
            expected_fc = theoretical_fc[sp]
            fig_density.add_vline(
                x=expected_fc,
                line_dash="dash",
                line_width=2,
                line_color=color,
                annotation_text=f"{sp}",
                annotation_position="top"
            )
    
    fig_density.update_layout(
        xaxis_title=f"log2({ref_cond}:{treat_cond})",
        yaxis_title="Density",
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig_density, use_container_width=True)
    
    # === PLOT 4: VOLCANO PLOT ===
    st.markdown("### 🌋 Volcano Plot")
    
    volc = res[res["regulation"] != "not_tested"].dropna(subset=['neg_log10_p', 'log2fc'])
    volc["Status"] = volc.apply(
        lambda x: "Significant" if x["regulation"] != "not_significant" else "Not Significant",
        axis=1
    )
    
    fig_v = px.scatter(
        volc,
        x="log2fc",
        y="neg_log10_p",
        color="species",
        opacity=0.7,
        color_discrete_map=SPECIES_COLORS,
        labels={
            "log2fc": f"log2({ref_cond}:{treat_cond})",
            "neg_log10_p": "-log10(FDR)" if use_fdr else "-log10(p)"
        },
        height=600,
    )
    
    # Gray out non-significant
    for i, trace in enumerate(fig_v.data):
        mask = volc[volc["species"] == trace.name]["Status"] == "Not Significant"
        if mask.any():
            trace.marker.opacity = np.where(mask, 0.15, 0.7)
    
    fig_v.add_hline(
        y=-np.log10(p_thr),
        line_dash="dash",
        line_color="gray",
        line_width=2,
        annotation_text=f"p={p_thr}"
    )
    
    if theoretical_fc:
        fig_v.add_vline(x=fc_tolerance, line_dash="dot", line_color="gray", line_width=1, opacity=0.5)
        fig_v.add_vline(x=-fc_tolerance, line_dash="dot", line_color="gray", line_width=1, opacity=0.5)
    
    st.plotly_chart(fig_v, use_container_width=True)
    
    # ========================================================================
    # VALIDATION SECTION
    # ========================================================================
    
    if theoretical_fc:
        st.markdown("---")
        st.subheader("6️⃣ Spike-in Validation")
        
        st.info(f"✓ **Using theoretical FC**: {', '.join(f'{k}={v:.2f}' for k, v in theoretical_fc.items())}")
        st.caption(f"**FP definition (unified)**: p < {p_thr} AND |log2fc - expected| > ±{fc_tolerance}")
        
        species_series = df[species_col]
        var_ov, var_sp, stab_ov, stab_sp, asym_dict, error_dict, fp_var_sp = compute_species_metrics(
            res, theoretical_fc, species_series, stable_thr, fc_tolerance, p_thr
        )
        
        # === VALIDATION TABLE ===
        validation_rows = []
        
        initial_total = len(df)
        
        # Asymmetry
        for sp in ["HUMAN", "ECOLI", "YEAST"]:
            asym_val = asym_dict.get(sp, np.nan)
            validation_rows.append({
                "Metric": "Asymmetry",
                "Species": sp,
                "Value": f"{asym_val:.2f}" if not np.isnan(asym_val) else "N/A",
                "Category": "Quality"
            })
        
        # False Positives
        for sp in ["HUMAN", "ECOLI", "YEAST"]:
            fp_count = error_dict.get(f"FP_{sp}", 0)
            misreg_count = error_dict.get(f"Misreg_{sp}", 0)
            
            total = 0
            fpr = 0
            
            if sp in stab_sp["Species"].values:
                total = int(stab_sp[stab_sp["Species"] == sp]["N"].values[0])
                fpr = fp_count / total * 100 if total > 0 else 0
            elif sp in fp_var_sp["Species"].values:
                total = int(fp_var_sp[fp_var_sp["Species"] == sp]["Total_Proteins"].values[0])
                fpr = fp_count / total * 100 if total > 0 else 0
            
            if total > 0:
                validation_rows.append({
                    "Metric": "False Positives" if sp == "HUMAN" else "Misregulated",
                    "Species": sp,
                    "Value": f"{fp_count + misreg_count:,} / {total:,} ({fpr:.1f}%)",
                    "Category": "Error"
                })
        
        # Detection/Accuracy (variable species)
        for sp in ["ECOLI", "YEAST"]:
            if sp in var_sp["Species"].values:
                det_pct = float(var_sp[var_sp["Species"] == sp]["Detection_%"].values[0].rstrip("%"))
                mae = float(var_sp[var_sp["Species"] == sp]["MAE"].values[0])
                
                validation_rows.append({
                    "Metric": "Detection",
                    "Species": sp,
                    "Value": f"{det_pct:.1f}%",
                    "Category": "Sensitivity"
                })
                
                validation_rows.append({
                    "Metric": "MAE (log2)",
                    "Species": sp,
                    "Value": f"{mae:.3f}",
                    "Category": "Accuracy"
                })
        
        # Overall metrics
        if var_ov:
            validation_rows.append({
                "Metric": "Sensitivity (Variable)",
                "Species": "Overall",
                "Value": f"{var_ov['Sensitivity']:.1%}",
                "Category": "Sensitivity"
            })
            
            validation_rows.append({
                "Metric": "Specificity (Variable)",
                "Species": "Overall",
                "Value": f"{var_ov['Specificity']:.1%}",
                "Category": "Specificity"
            })
            
            validation_rows.append({
                "Metric": "Precision",
                "Species": "Overall",
                "Value": f"{var_ov['Precision']:.1%}",
                "Category": "Precision"
            })
        
        if stab_ov:
            validation_rows.append({
                "Metric": "FPR (Stable)",
                "Species": "HUMAN",
                "Value": f"{stab_ov['FPR']:.1%}",
                "Category": "Specificity"
            })
        
        # deFDR
        if true_positives > 0:
            validation_rows.append({
                "Metric": "deFDR",
                "Species": "Overall",
                "Value": f"{de_fdr:.2f}%",
                "Category": "Error"
            })
        
        validation_df = pd.DataFrame(validation_rows)
        
        st.markdown("### 📊 Validation Metrics")
        st.dataframe(
            validation_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Metric": st.column_config.TextColumn(width=150),
                "Species": st.column_config.TextColumn(width=100),
                "Value": st.column_config.TextColumn(width=150),
                "Category": st.column_config.TextColumn(width=100),
            }
        )
        
        st.download_button(
            "📥 Download Validation Metrics",
            data=validation_df.to_csv(index=False).encode("utf-8"),
            file_name=f"validation_metrics_{ref_cond}_vs_{treat_cond}.csv",
            mime="text/csv",
        )
    
    # ========================================================================
    # METHOD COMPARISON (if Both selected)
    # ========================================================================
    
    if test_method == "Both":
        st.markdown("---")
        st.subheader("7️⃣ Method Comparison: Limma vs Welch")
        
        comparison_df = res[["pvalue", "pvalue_welch", "fdr", "fdr_welch", "log2fc"]].copy()
        comparison_df = comparison_df.dropna()
        
        col1, col2 = st.columns(2)
        
        with col1:
            corr_p = comparison_df["pvalue"].corr(comparison_df["pvalue_welch"])
            st.metric("P-value Correlation", f"{corr_p:.3f}")
        
        with col2:
            corr_fdr = comparison_df["fdr"].corr(comparison_df["fdr_welch"])
            st.metric("FDR Correlation", f"{corr_fdr:.3f}")
        
        # Scatter plot
        fig_comp = px.scatter(
            comparison_df,
            x="pvalue_welch",
            y="pvalue",
            hover_data=["log2fc"],
            labels={"pvalue_welch": "Welch p-value", "pvalue": "Limma p-value"},
            title="P-value Comparison",
            height=500
        )
        fig_comp.update_xaxes(type="log")
        fig_comp.update_yaxes(type="log")
        
        st.plotly_chart(fig_comp, use_container_width=True)
    
    # ========================================================================
    # INDIVIDUAL PROTEIN TABLE
    # ========================================================================
    
    st.markdown("---")
    
    with st.expander("🔬 View Individual Protein Results", expanded=False):
        st.markdown("### Individual Protein Data")
        st.caption(f"Complete results for all {len(res):,} proteins")
        
        results_table = res.copy()
        results_table["species"] = results_table.index.map(df[species_col])
        results_table = results_table.reset_index()
        results_table = results_table.rename(columns={"protein_id": "Protein_ID"})
        
        results_table = results_table.round({
            "log2fc": 3,
            "pvalue": 6,
            "fdr": 6,
            "pvalue_welch": 6,
            "fdr_welch": 6,
            "mean_g1": 2,
            "mean_g2": 2,
            "neg_log10_p": 2,
            "moderated_t": 3,
        })
        
        display_cols = [
            "Protein_ID", "species", "log2fc", "pvalue", "fdr",
            "mean_g1", "mean_g2", "regulation", "moderated_t", "moderated_df"
        ]
        
        if test_method == "Both":
            display_cols.extend(["pvalue_welch", "fdr_welch"])
        
        results_table = results_table[[c for c in display_cols if c in results_table.columns]]
        
        st.dataframe(
            results_table,
            use_container_width=True,
            height=600,
            column_config={
                "Protein_ID": st.column_config.TextColumn(width=100),
                "species": st.column_config.TextColumn(width=80),
                "log2fc": st.column_config.NumberColumn(format="%.3f"),
                "pvalue": st.column_config.NumberColumn(format="%.2e"),
                "fdr": st.column_config.NumberColumn(format="%.2e"),
                "regulation": st.column_config.TextColumn(width=80),
            }
        )
        
        st.download_button(
            "📥 Download Individual Proteins (CSV)",
            data=results_table.to_csv(index=False).encode("utf-8"),
            file_name=f"dea_proteins_{ref_cond}_vs_{treat_cond}.csv",
            mime="text/csv",
            key="individual_download"
        )

else:
    st.info("👆 Configure settings and run analysis above")

st.markdown("---")
st.success("✅ **Limma-style differential abundance testing** with empirical Bayes moderation")
st.caption("Unified FP definition applied across all species for robust validation")
