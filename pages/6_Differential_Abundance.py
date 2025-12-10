"""
pages/6_Differential_Abundance.py
Welch's t-test + Limma-style EB with comprehensive visualization and spike-in validation.
Unified FP definition: p<0.01 AND |log2fc - expected| > ±0.58 for all species
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, gaussian_kde
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(str(Path(__file__).parent.parent))

# ============================================================================
# COLOR SCHEME
# ============================================================================

SPECIES_COLORS = {
    "HUMAN": "#2ecc71",  # Green
    "YEAST": "#e67e22",  # Orange
    "ECOLI": "#9b59b6",  # Purple
}

# ============================================================================
# LIMMA-STYLE EMPIRICAL BAYES
# ============================================================================

def limma_EB_fit(variances: np.ndarray) -> Tuple[float, float]:
    """
    Fit empirical Bayes hyperparameters (d0, s02) to variance estimates.
    Simplified approach: method of moments.
    
    Returns: (d0, s02) where posterior variance = (d0*s02 + n*sample_var) / (d0 + n)
    """
    if len(variances) < 2:
        return 10.0, np.median(variances) if len(variances) > 0 else 1.0
    
    mean_var = np.mean(variances)
    var_of_vars = np.var(variances)
    
    # Method of moments: fit prior variance and degrees of freedom
    # d0: prior degrees of freedom (higher = more shrinkage)
    # s02: prior variance
    
    if var_of_vars > 0:
        d0 = 2 * mean_var**2 / var_of_vars  # Prior df
        s02 = mean_var
    else:
        d0 = 10.0
        s02 = mean_var
    
    d0 = max(1.0, d0)  # Ensure positive
    s02 = max(1e-6, s02)
    
    return d0, s02


def limma_moderated_ttest(
    group1: np.ndarray,
    group2: np.ndarray,
    all_variances: np.ndarray,
    d0: float = 10.0,
    s02: float = 1.0,
) -> Tuple[float, float]:
    """
    Limma-style moderated t-test with empirical Bayes shrinkage.
    Shrinks sample variances toward global estimate using EB hyperparameters.
    
    Returns: (t_stat, p_value)
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan, 1.0
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    
    # Pooled sample variance
    sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    
    # EB shrinkage: moderated pooled variance
    # s2_star = (d0*s02 + df*sp2) / (d0 + df)
    df = n1 + n2 - 2
    sp2_moderated = (d0 * s02 + df * sp2) / (d0 + df)
    
    # Standard error with moderated variance
    se = np.sqrt(sp2_moderated * (1/n1 + 1/n2))
    if se == 0:
        return np.nan, 1.0
    
    # t-statistic
    t_stat = (mean1 - mean2) / se
    
    # Degrees of freedom: moderated df
    df_moderated = d0 + df
    
    # p-value from moderated t-distribution
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df_moderated))
    
    return t_stat, p_val


def perform_dea(
    df: pd.DataFrame,
    group1_cols: List[str],
    group2_cols: List[str],
    use_limma: bool = True,
    min_valid: int = 2,
) -> pd.DataFrame:
    """
    Perform differential expression analysis with optional Limma-style EB.
    Log2FC = mean(group1) - mean(group2) (Limma convention)
    """
    results = []
    variances_all = []
    
    # First pass: collect all variances for EB fitting
    for protein_id in df.index:
        row = df.loc[protein_id]
        g1_vals = row[group1_cols].dropna().values
        g2_vals = row[group2_cols].dropna().values
        
        if len(g1_vals) >= min_valid:
            variances_all.append(np.var(g1_vals, ddof=1))
        if len(g2_vals) >= min_valid:
            variances_all.append(np.var(g2_vals, ddof=1))
    
    # Fit EB hyperparameters
    if use_limma and len(variances_all) > 10:
        d0, s02 = limma_EB_fit(np.array(variances_all))
    else:
        d0, s02 = 10.0, np.median(variances_all) if variances_all else 1.0
    
    # Second pass: compute statistics
    for protein_id in df.index:
        row = df.loc[protein_id]
        g1_vals = row[group1_cols].dropna().values
        g2_vals = row[group2_cols].dropna().values
        
        if len(g1_vals) < min_valid or len(g2_vals) < min_valid:
            results.append({
                "protein_id": protein_id,
                "log2fc": np.nan,
                "pvalue": np.nan,
                "t_stat": np.nan,
                "mean_g1": np.nan,
                "mean_g2": np.nan,
                "n_g1": len(g1_vals),
                "n_g2": len(g2_vals),
                "var_g1": np.nan,
                "var_g2": np.nan,
            })
            continue
        
        mean_g1 = np.mean(g1_vals)
        mean_g2 = np.mean(g2_vals)
        log2fc = mean_g1 - mean_g2
        
        # Standard t-test
        t_stat_std, p_std = ttest_ind(g1_vals, g2_vals, equal_var=False)
        
        # Limma moderated t-test
        if use_limma:
            t_stat, p_val = limma_moderated_ttest(
                g1_vals, g2_vals, np.array(variances_all), d0, s02
            )
        else:
            t_stat, p_val = t_stat_std, p_std
        
        var_g1 = np.var(g1_vals, ddof=1)
        var_g2 = np.var(g2_vals, ddof=1)
        
        results.append({
            "protein_id": protein_id,
            "log2fc": log2fc,
            "pvalue": p_val,
            "t_stat": t_stat,
            "mean_g1": mean_g1,
            "mean_g2": mean_g2,
            "n_g1": len(g1_vals),
            "n_g2": len(g2_vals),
            "var_g1": var_g1,
            "var_g2": var_g2,
        })
    
    results_df = pd.DataFrame(results).set_index("protein_id")
    
    # FDR Correction (Benjamini-Hochberg)
    pvals = results_df["pvalue"].dropna().values
    n_tests = len(pvals)
    
    if n_tests > 0:
        sorted_idx = np.argsort(pvals)
        sorted_p = pvals[sorted_idx]
        ranks = np.arange(1, n_tests + 1)
        fdr_vals = sorted_p * n_tests / ranks
        fdr_vals = np.minimum.accumulate(fdr_vals[::-1])[::-1]
        fdr_vals = np.minimum(fdr_vals, 1.0)
        
        fdr_dict = dict(zip(results_df["pvalue"].dropna().index, fdr_vals))
        results_df["fdr"] = results_df["pvalue"].map(fdr_dict)
    else:
        results_df["fdr"] = np.nan
    
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
    """Calculate asymmetry: median(observed) / expected"""
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
    
    Unified FP definition (ALL SPECIES): p<p_threshold AND |log2fc - expected| > fc_tolerance
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
        
        # Confusion matrix
        tp = int((var_df["true_regulated"] & var_df["observed_regulated"]).sum())
        fn = int((var_df["true_regulated"] & ~var_df["observed_regulated"]).sum())
        tn = int((~var_df["true_regulated"] & ~var_df["observed_regulated"]).sum())
        fp = int((~var_df["true_regulated"] & var_df["observed_regulated"]).sum())
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        var_overall = {
            "Total": len(var_df),
            "TP": tp,
            "FN": fn,
            "Sensitivity": sens,
            "Specificity": spec,
            "Precision": prec,
        }
        
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
        
        # True FP: p<threshold AND |error| > ±0.58
        true_fp = int((stab_df["significant"] & stab_df["outside_tolerance"]).sum())
        tn = int((~stab_df["significant"] | ~stab_df["outside_tolerance"]).sum())
        
        total = len(stab_df)
        fpr = true_fp / total if total > 0 else 0.0
        
        stab_overall = {"Total": total, "FP": true_fp, "TN": tn, "FPR": fpr}
        
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
    
    # === FALSE POSITIVES IN VARIABLE PROTEOME ===
    fp_var_species_rows = []
    
    if not var_df.empty:
        for sp in var_df["species"].unique():
            sp_df = var_df[var_df["species"] == sp].copy()
            sp_df["significant"] = sp_df["pvalue"] < p_threshold
            sp_df["outside_tolerance"] = np.abs(sp_df["log2fc"] - sp_df["true_log2fc"]) > fc_tolerance
            
            # FP: p<threshold AND outside ±0.58
            fp_s = int((sp_df["significant"] & sp_df["outside_tolerance"]).sum())
            accurate_s = int((sp_df["significant"] & ~sp_df["outside_tolerance"]).sum())
            total_var_s = len(sp_df)
            
            fp_var_species_rows.append({
                "Species": sp,
                "Total_Detected": total_var_s,
                "Accurate": accurate_s,
                "FP_Wrong_Mag": fp_s,
                "Accuracy_%": f"{accurate_s/total_var_s*100:.1f}" if total_var_s > 0 else "0.0",
            })
            
            error_dict[f"FP_{sp}"] = fp_s
    
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
st.title("🔬 Differential Abundance Analysis (DEA)")
st.markdown("Welch's t-test + Limma-style EB with spike-in validation")
st.markdown("---")

# ============================================================================
# DATA
# ============================================================================

if "df_imputed" not in st.session_state or st.session_state.df_imputed is None:
    st.error("❌ No imputed data. Complete **📊 Post-Imputation EDA** first.")
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

st.info(f"📊 **Data**: {df.shape[0]:,} proteins × {len(numeric_cols)} samples · **Conditions**: {', '.join(conditions)}")

# ============================================================================
# 1. COMPARISON SETUP
# ============================================================================

st.subheader("1️⃣ Comparison Setup (A vs B)")

col1, col2 = st.columns(2)
with col1:
    ref_cond = st.selectbox("Condition A (reference)", options=conditions, index=0)
with col2:
    treat_cond = st.selectbox("Condition B (treatment)", options=[c for c in conditions if c != ref_cond], index=0 if len(conditions) > 1 else 0)

if ref_cond == treat_cond:
    st.error("❌ Choose two different conditions.")
    st.stop()

ref_samples = cond_samples[ref_cond]
treat_samples = cond_samples[treat_cond]

st.markdown(f"- **Log2FC** = {ref_cond} - {treat_cond} (positive = higher in {ref_cond})")
st.markdown("---")

# ============================================================================
# 2. SPIKE-IN COMPOSITION
# ============================================================================

st.subheader("2️⃣ Spike-in Composition (Optional)")

use_comp = st.checkbox("✓ Provide % composition per species", value=False)

theoretical_fc_temp: Dict[str, float] = {}
species_values = sorted([s for s in df[species_col].unique() if isinstance(s, str) and s.strip()])

if use_comp:
    st.markdown("Enter **% composition** (auto-normalized to 100% per condition)")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{ref_cond} (A)**")
        comp_a = {}
        for sp in species_values:
            val = st.number_input(
                f"{sp} (%)",
                min_value=0.0,
                max_value=100.0,
                value=100.0 / max(len(species_values), 1),
                step=5.0,
                key=f"a_{sp}"
            )
            comp_a[sp] = val
        ta = sum(comp_a.values()) or 1.0
        comp_a = {k: v * 100 / ta for k, v in comp_a.items()}
    
    with c2:
        st.markdown(f"**{treat_cond} (B)**")
        comp_b = {}
        for sp in species_values:
            val = st.number_input(
                f"{sp} (%)",
                min_value=0.0,
                max_value=100.0,
                value=100.0 / max(len(species_values), 1),
                step=5.0,
                key=f"b_{sp}"
            )
            comp_b[sp] = val
        tb = sum(comp_b.values()) or 1.0
        comp_b = {k: v * 100 / tb for k, v in comp_b.items()}
    
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
            log2fc = float(np.log2(pa / pb))
        theoretical_fc_temp[sp] = log2fc
        rows.append({
            "Species": sp,
            f"{ref_cond} (%)": f"{pa:.1f}",
            f"{treat_cond} (%)": f"{pb:.1f}",
            "Log2FC": f"{log2fc:.3f}",
            "Linear_FC": f"{2**log2fc:.2f}x"
        })
    
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    
    if st.button("💾 Save Expected FC", type="primary"):
        st.session_state.dea_theoretical_fc = theoretical_fc_temp.copy()
        st.success(f"✅ Saved for {len(theoretical_fc_temp)} species!")
    
    saved_fc = st.session_state.get('dea_theoretical_fc', {})
    if saved_fc:
        st.info(f"✓ **Saved**: {', '.join(f'{k}={v:.2f}' for k, v in saved_fc.items())}")
    else:
        st.warning("⚠️ Not saved yet")

st.markdown("---")

# ============================================================================
# 3. STATISTICAL SETTINGS
# ============================================================================

st.subheader("3️⃣ Statistical Settings")

c1, c2, c3 = st.columns(3)
with c1:
    use_limma = st.checkbox("✓ Use Limma-style EB", value=True, help="Empirical Bayes shrinkage of variances")
with c2:
    use_fdr = st.checkbox("✓ Use FDR correction (BH)", value=True)
with c3:
    p_thr = st.selectbox("P-value threshold", options=[0.01, 0.05, 0.1], index=0)

stable_thr = 0.5
fc_tolerance = 0.58

st.caption(
    f"**Stable proteome**: |expected log2FC| < {stable_thr}  \n"
    f"**FP definition (unified)**: p < {p_thr} AND |log2fc - expected| > ±{fc_tolerance}"
)

st.markdown("---")

# ============================================================================
# 4. RUN ANALYSIS
# ============================================================================

st.subheader("4️⃣ Run Analysis")

if st.button("🚀 Run DEA (Welch's t-test + Limma EB)", type="primary"):
    with st.spinner("⏳ Running analysis..."):
        # Log2 transform if needed
        df_num = df[numeric_cols]
        if (df_num > 50).any().any():
            df_test = np.log2(df_num + 1.0)
        else:
            df_test = df_num.copy()
        
        # Run DEA
        results = perform_dea(df_test, ref_samples, treat_samples, use_limma=use_limma)
        
        # Classify regulation
        test_col = "fdr" if use_fdr else "pvalue"
        results["regulation"] = results.apply(
            lambda row: classify_regulation(row["log2fc"], row[test_col], 0.0, p_thr),
            axis=1
        )
        results["neg_log10_p"] = -np.log10(results[test_col].replace(0, 1e-300))
        results["species"] = results.index.map(df[species_col])
        
        st.session_state.dea_results = results
        st.session_state.dea_ref = ref_cond
        st.session_state.dea_treat = treat_cond
        st.session_state.dea_p_thr = p_thr
        st.session_state.dea_use_fdr = use_fdr
        st.session_state.dea_use_limma = use_limma
    
    st.success("✅ Analysis complete!")

# ============================================================================
# 5. RESULTS DISPLAY
# ============================================================================

if "dea_results" in st.session_state:
    res = st.session_state.dea_results
    ref_cond = st.session_state.dea_ref
    treat_cond = st.session_state.dea_treat
    p_thr = st.session_state.dea_p_thr
    use_fdr = st.session_state.get('dea_use_fdr', True)
    use_limma_flag = st.session_state.get('dea_use_limma', True)
    theoretical_fc = st.session_state.get('dea_theoretical_fc', {})
    
    st.markdown("---")
    st.subheader("5️⃣ Results Overview")
    
    n_total = len(res)
    n_quant = int((res["regulation"] != "not_tested").sum())
    quant_rate = n_quant / n_total * 100 if n_total > 0 else 0
    
    test_col = "fdr" if use_fdr else "pvalue"
    test_label = "FDR" if use_fdr else "p-value"
    
    # Validation metrics
    sens_pct = 0.0
    spec_pct = 0.0
    true_positives = 0
    de_fdr = 0.0
    total_fp = 0
    
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
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Quantified", f"{n_quant:,}")
    with col2:
        st.metric("Quant. Rate", f"{quant_rate:.0f}%")
    with col3:
        st.metric("Method", "Limma+EB" if use_limma_flag else "Welch")
    with col4:
        st.metric("Threshold", f"p={p_thr}")
    with col5:
        st.metric("Test", test_label)
    
    if theoretical_fc:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sensitivity", f"{sens_pct:.1f}%")
        with col2:
            st.metric("Specificity", f"{spec_pct:.1f}%")
        with col3:
            st.metric("True Positives", true_positives)
        with col4:
            st.metric("deFDR", f"{de_fdr:.2f}%")
    
    st.markdown("---")
    
    # ============================================================================
    # PLOTS
    # ============================================================================
    
    st.subheader("📊 Visualizations")
    
    # === PLOT 1: FACETED MA PLOT ===
    st.markdown("### 1️⃣ MA Plot (Faceted by Species)")
    
    ma = res[res["regulation"] != "not_tested"].copy()
    ma["A"] = (ma["mean_g1"] + ma["mean_g2"]) / 2
    ma = ma.dropna(subset=['A', 'log2fc', 'species'])
    
    species_list = sorted(ma["species"].unique())
    fig_facet = make_subplots(
        rows=1, cols=len(species_list),
        subplot_titles=species_list,
        shared_yaxes=True,
        horizontal_spacing=0.05
    )
    
    for i, sp in enumerate(species_list, 1):
        sp_data = ma[ma["species"] == sp]
        color = SPECIES_COLORS.get(sp, "#95a5a6")
        
        sp_sig = sp_data[sp_data["regulation"] != "not_significant"]
        sp_nonsig = sp_data[sp_data["regulation"] == "not_significant"]
        
        # Non-sig (grayed)
        fig_facet.add_trace(
            go.Scatter(
                x=sp_nonsig["A"],
                y=sp_nonsig["log2fc"],
                mode='markers',
                marker=dict(size=3, color="lightgray", opacity=0.3),
                showlegend=False,
                hovertemplate=f"{sp} (ns)<br>A=%{{x:.2f}}<br>log2FC=%{{y:.3f}}<extra></extra>"
            ),
            row=1, col=i
        )
        
        # Up-regulated
        sp_up = sp_sig[sp_sig["regulation"] == "up"]
        fig_facet.add_trace(
            go.Scatter(
                x=sp_up["A"],
                y=sp_up["log2fc"],
                mode='markers',
                marker=dict(size=4, color=color, opacity=0.8, symbol='circle'),
                showlegend=(i==1),
                name=f"{sp} (↑)",
                hovertemplate=f"{sp} ↑<br>A=%{{x:.2f}}<br>log2FC=%{{y:.3f}}<extra></extra>"
            ),
            row=1, col=i
        )
        
        # Down-regulated
        sp_down = sp_sig[sp_sig["regulation"] == "down"]
        fig_facet.add_trace(
            go.Scatter(
                x=sp_down["A"],
                y=sp_down["log2fc"],
                mode='markers',
                marker=dict(size=4, color=color, opacity=0.8, symbol='diamond'),
                showlegend=(i==1),
                name=f"{sp} (↓)",
                hovertemplate=f"{sp} ↓<br>A=%{{x:.2f}}<br>log2FC=%{{y:.3f}}<extra></extra>"
            ),
            row=1, col=i
        )
        
        # Expected FC line + tolerance band
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
    
    fig_facet.update_xaxes(title_text="log2(Mean Intensity)", row=1, col=1)
    fig_facet.update_yaxes(title_text=f"log2FC ({ref_cond}/{treat_cond})", row=1, col=1)
    fig_facet.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig_facet, use_container_width=True)
    
    # === PLOT 2: REGULAR MA PLOT ===
    st.markdown("### 2️⃣ MA Plot (Combined)")
    
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
        labels={"A": "log2(Mean Intensity)", "log2fc": f"log2FC ({ref_cond}/{treat_cond})"},
        height=600,
    )
    
    for trace in fig_ma.data:
        if trace.name == "Not Sig.":
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
            
            fig_ma.add_hline(
                y=expected_fc + fc_tolerance,
                line_dash="dot",
                line_width=1,
                line_color=color,
                opacity=0.2,
            )
            fig_ma.add_hline(
                y=expected_fc - fc_tolerance,
                line_dash="dot",
                line_width=1,
                line_color=color,
                opacity=0.2,
            )
    
    st.plotly_chart(fig_ma, use_container_width=True)
    
    # === PLOT 3: DENSITY PLOT ===
    st.markdown("### 3️⃣ Density Plot (Log2FC Distribution)")
    
    density_data = res[res["regulation"] != "not_tested"].dropna(subset=["log2fc", "species"])
    
    fig_density = go.Figure()
    
    for sp in sorted(density_data["species"].unique()):
        sp_data = density_data[density_data["species"] == sp]["log2fc"].values
        color = SPECIES_COLORS.get(sp, "#95a5a6")
        
        if len(sp_data) > 2:
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
                    annotation_text=f"{sp}: {expected_fc:.2f}"
                )
    
    fig_density.update_layout(
        xaxis_title=f"log2FC ({ref_cond}/{treat_cond})",
        yaxis_title="Density",
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig_density, use_container_width=True)
    
    # === PLOT 4: VOLCANO PLOT ===
    st.markdown("### 4️⃣ Volcano Plot")
    
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
        labels={"log2fc": f"log2FC ({ref_cond}/{treat_cond})", "neg_log10_p": f"-log10({test_label})"},
        height=600,
    )
    
    for trace in fig_v.data:
        mask = volc[volc["species"] == trace.name]["Status"] == "Not Significant"
        if mask.any():
            trace.marker.opacity = np.where(mask, 0.15, 0.7)
    
    fig_v.add_hline(y=-np.log10(p_thr), line_dash="dash", line_color="gray", line_width=2, annotation_text=f"p={p_thr}")
    
    if theoretical_fc:
        fig_v.add_vline(x=fc_tolerance, line_dash="dot", line_color="gray", line_width=1, opacity=0.5, annotation_text=f"±{fc_tolerance}")
        fig_v.add_vline(x=-fc_tolerance, line_dash="dot", line_color="gray", line_width=1, opacity=0.5)
    
    st.plotly_chart(fig_v, use_container_width=True)
    
    # === PLOT 5: EFFECT SIZE DISTRIBUTION ===
    st.markdown("### 5️⃣ Effect Size (Log2FC) Distribution")
    
    fig_effect = px.histogram(
        res[res["regulation"] != "not_tested"].dropna(subset=['log2fc']),
        x='log2fc',
        color='species',
        color_discrete_map=SPECIES_COLORS,
        nbinsx=50,
        title='Log2FC Distribution',
        labels={'log2fc': f'log2FC ({ref_cond}/{treat_cond})'},
        height=500,
    )
    
    fig_effect.add_vline(x=0, line_color="red", line_width=2)
    
    st.plotly_chart(fig_effect, use_container_width=True)
    
    # === PLOT 6: P-VALUE DISTRIBUTION ===
    st.markdown("### 6️⃣ P-Value Distribution")
    
    fig_pval = px.histogram(
        res[res["regulation"] != "not_tested"].dropna(subset=[test_col]),
        x=test_col,
        nbinsx=50,
        title=f'{test_label} Distribution',
        labels={test_col: test_label},
        height=500,
    )
    
    fig_pval.add_vline(x=p_thr, line_dash="dash", line_color="red", line_width=2, annotation_text=f"{test_label}={p_thr}")
    
    st.plotly_chart(fig_pval, use_container_width=True)
    
    # ============================================================================
    # VALIDATION METRICS (IF SPIKE-IN PROVIDED)
    # ============================================================================
    
    if theoretical_fc:
        st.markdown("---")
        st.subheader("6️⃣ Spike-in Validation Metrics")
        
        st.info(f"**Theoretical FC**: {', '.join(f'{k}={v:.2f}' for k, v in theoretical_fc.items())}")
        st.caption(f"**Unified FP definition**: p<{p_thr} AND |log2fc - expected| > ±{fc_tolerance}")
        
        species_series = df[species_col]
        var_ov, var_sp, stab_ov, stab_sp, asym_dict, error_dict, fp_var_sp = compute_species_metrics(
            res, theoretical_fc, species_series, stable_thr, fc_tolerance, p_thr
        )
        
        # Build validation table
        validation_rows = []
        initial_total = len(df)
        
        # Asymmetry
        for sp in ["HUMAN", "YEAST", "ECOLI"]:
            asym_val = asym_dict.get(sp, np.nan)
            validation_rows.append({
                "Metric": "Asymmetry",
                "Species": sp,
                "Value": f"{asym_val:.2f}" if not np.isnan(asym_val) else "N/A",
                "Category": "Quality"
            })
        
        # FP metrics
        for sp in ["HUMAN", "YEAST", "ECOLI"]:
            fp_count = error_dict.get(f"FP_{sp}", 0)
            total = 0
            
            if sp in stab_sp["Species"].values:
                total = int(stab_sp[stab_sp["Species"] == sp]["N"].values[0])
            elif sp in fp_var_sp["Species"].values:
                total = int(fp_var_sp[fp_var_sp["Species"] == sp]["Total_Detected"].values[0])
            
            if total > 0:
                fpr = fp_count / total * 100
                validation_rows.append({
                    "Metric": "False Positives",
                    "Species": sp,
                    "Value": f"{fp_count:,} / {total:,} ({fpr:.1f}%)",
                    "Category": "Error"
                })
        
        # Detection (variable species)
        for sp in ["YEAST", "ECOLI"]:
            if sp in var_sp["Species"].values:
                det_pct = float(var_sp[var_sp["Species"] == sp]["Detection_%"].values[0].rstrip("%"))
                validation_rows.append({
                    "Metric": "Detection",
                    "Species": sp,
                    "Value": f"{det_pct:.1f}%",
                    "Category": "Sensitivity"
                })
                
                mae = float(var_sp[var_sp["Species"] == sp]["MAE"].values[0])
                validation_rows.append({
                    "Metric": "MAE (log2)",
                    "Species": sp,
                    "Value": f"{mae:.3f}",
                    "Category": "Accuracy"
                })
        
        # Overall metrics
        if var_ov:
            validation_rows.append({
                "Metric": "Sensitivity",
                "Species": "Overall",
                "Value": f"{var_ov['Sensitivity']:.1%}",
                "Category": "Sensitivity"
            })
            
            validation_rows.append({
                "Metric": "Specificity",
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
                "Species": "Overall",
                "Value": f"{stab_ov['FPR']:.1%}",
                "Category": "Specificity"
            })
        
        validation_df = pd.DataFrame(validation_rows)
        
        st.markdown("### Validation Metrics Table")
        st.dataframe(validation_df, use_container_width=True, hide_index=True)
        
        st.download_button(
            "📥 Download Validation Metrics",
            data=validation_df.to_csv(index=False).encode("utf-8"),
            file_name=f"validation_{ref_cond}_vs_{treat_cond}.csv",
            mime="text/csv",
        )
    
    # ============================================================================
    # INDIVIDUAL PROTEIN RESULTS
    # ============================================================================
    
    st.markdown("---")
    
    with st.expander("🔬 View Individual Protein Results", expanded=False):
        st.markdown("### Detailed Results Table")
        
        results_table = res.copy().reset_index()
        results_table = results_table.rename(columns={"protein_id": "Protein"})
        
        results_table = results_table.round({
            "log2fc": 3,
            "pvalue": 6,
            "fdr": 6,
            "mean_g1": 2,
            "mean_g2": 2,
            "neg_log10_p": 2
        })
        
        display_cols = [
            "Protein", "species", "log2fc", "pvalue", "fdr",
            "mean_g1", "mean_g2", "regulation", "neg_log10_p", "n_g1", "n_g2"
        ]
        
        results_table = results_table[[c for c in display_cols if c in results_table.columns]]
        
        st.dataframe(results_table, use_container_width=True, height=600)
        
        st.download_button(
            "📥 Download Individual Results (CSV)",
            data=results_table.to_csv(index=False).encode("utf-8"),
            file_name=f"dea_proteins_{ref_cond}_vs_{treat_cond}.csv",
            mime="text/csv",
        )
    
    st.markdown("---")
    st.success("✅ Differential Abundance Analysis Complete!")

else:
    st.info("👆 Configure comparison and run analysis")
