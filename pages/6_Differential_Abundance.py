"""
pages/6_Differential_Abundance.py - FIXED
Comprehensive Differential Abundance Analysis (DEA)
- Welch's t-test + Limma-style Empirical Bayes
- Parametric & non-parametric tests
- Spike-in validation with unified FP definition
- Effect sizes, visualizations, interpretation
- BUGFIX: Empty species handling + deprecated Streamlit warnings
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, gaussian_kde, t as t_dist
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(str(Path(__file__).parent.parent))

# ============================================================================
# COLOR SCHEME
# ============================================================================

SPECIES_COLORS = {
    "HUMAN": "#2ecc71",    # Green
    "YEAST": "#e67e22",    # Orange
    "ECOLI": "#9b59b6",    # Purple
}

# ============================================================================
# LIMMA-STYLE EMPIRICAL BAYES
# ============================================================================

def limma_EB_fit(variances: np.ndarray) -> Tuple[float, float]:
    """
    Fit empirical Bayes hyperparameters (d0, s02) to variance estimates.
    Simplified method of moments approach.
    
    Returns: (d0, s02) where posterior variance = (d0*s02 + n*sample_var) / (d0 + n)
    """
    if len(variances) < 2:
        return 10.0, np.median(variances) if len(variances) > 0 else 1.0
    
    mean_var = np.mean(variances)
    var_of_vars = np.var(variances)
    
    if var_of_vars > 0:
        d0 = 2 * mean_var**2 / var_of_vars
        s02 = mean_var
    else:
        d0 = 10.0
        s02 = mean_var
    
    d0 = max(1.0, d0)
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
    
    sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    
    df = n1 + n2 - 2
    sp2_moderated = (d0 * s02 + df * sp2) / (d0 + df)
    
    se = np.sqrt(sp2_moderated * (1/n1 + 1/n2))
    if se == 0:
        return np.nan, 1.0
    
    t_stat = (mean1 - mean2) / se
    df_moderated = d0 + df
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df_moderated))
    
    return t_stat, p_val


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Calculate Benjamini-Hochberg FDR correction"""
    n = len(pvalues)
    sorted_idx = np.argsort(pvalues)
    sorted_p = pvalues[sorted_idx]
    
    adjusted_p = np.zeros(n)
    adjusted_p[sorted_idx] = sorted_p * n / (np.arange(1, n + 1))
    
    for i in range(n - 2, -1, -1):
        adjusted_p[i] = min(adjusted_p[i], adjusted_p[i + 1])
    
    adjusted_p = np.minimum(adjusted_p, 1.0)
    
    return adjusted_p


def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0
    
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return cohens_d


def perform_dea(
    df: pd.DataFrame,
    group1_cols: List[str],
    group2_cols: List[str],
    use_limma: bool = True,
    use_parametric: bool = True,
    use_nonparametric: bool = True,
    min_valid: int = 2,
) -> pd.DataFrame:
    """
    Perform differential expression analysis.
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
                "p_ttest": np.nan,
                "p_mannwhitney": np.nan,
                "t_stat": np.nan,
                "u_stat": np.nan,
                "cohens_d": np.nan,
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
        cohens_d = calculate_cohens_d(g1_vals, g2_vals)
        
        var_g1 = np.var(g1_vals, ddof=1)
        var_g2 = np.var(g2_vals, ddof=1)
        
        # Parametric test
        if use_parametric:
            if use_limma:
                t_stat, p_ttest = limma_moderated_ttest(
                    g1_vals, g2_vals, np.array(variances_all), d0, s02
                )
            else:
                t_stat, p_ttest = ttest_ind(g1_vals, g2_vals, equal_var=False)
        else:
            t_stat, p_ttest = np.nan, np.nan
        
        # Non-parametric test
        if use_nonparametric:
            try:
                u_stat, p_mw = mannwhitneyu(g1_vals, g2_vals)
            except Exception:
                u_stat, p_mw = np.nan, np.nan
        else:
            u_stat, p_mw = np.nan, np.nan
        
        results.append({
            "protein_id": protein_id,
            "log2fc": log2fc,
            "p_ttest": p_ttest,
            "p_mannwhitney": p_mw,
            "t_stat": t_stat,
            "u_stat": u_stat,
            "cohens_d": cohens_d,
            "mean_g1": mean_g1,
            "mean_g2": mean_g2,
            "n_g1": len(g1_vals),
            "n_g2": len(g2_vals),
            "var_g1": var_g1,
            "var_g2": var_g2,
        })
    
    results_df = pd.DataFrame(results).set_index("protein_id")
    
    # FDR Correction
    if use_parametric:
        pvals = results_df["p_ttest"].dropna().values
        if len(pvals) > 0:
            fdr_vals = benjamini_hochberg(pvals)
            fdr_dict = dict(zip(results_df["p_ttest"].dropna().index, fdr_vals))
            results_df["q_ttest"] = results_df["p_ttest"].map(fdr_dict)
        else:
            results_df["q_ttest"] = np.nan
    
    if use_nonparametric:
        pvals = results_df["p_mannwhitney"].dropna().values
        if len(pvals) > 0:
            fdr_vals = benjamini_hochberg(pvals)
            fdr_dict = dict(zip(results_df["p_mannwhitney"].dropna().index, fdr_vals))
            results_df["q_mannwhitney"] = results_df["p_mannwhitney"].map(fdr_dict)
        else:
            results_df["q_mannwhitney"] = np.nan
    
    return results_df


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
    test_col: str = "p_ttest",
) -> Tuple[Dict, pd.DataFrame, Dict, pd.DataFrame, Dict, Dict, pd.DataFrame]:
    """
    Calculate validation metrics for variable and stable proteomes.
    
    Unified FP definition (ALL SPECIES): p<p_threshold AND |log2fc - expected| > fc_tolerance
    """
    res = results_df.copy()
    res["species"] = res.index.map(species_col_series)
    res["true_log2fc"] = res["species"].map(true_fc_dict)
    
    res = res[res["regulation"] != "not_tested"].copy()
    res = res.dropna(subset=["true_log2fc", "species"])
    
    if res.empty:
        return {}, pd.DataFrame(), {}, pd.DataFrame(), {}, {}, pd.DataFrame()
    
    # === ASYMMETRY ===
    asymmetry_dict = {}
    for sp in res["species"].unique():
        sp_df = res[res["species"] == sp].copy()
        expected_fc = true_fc_dict.get(sp, 0.0)
        if abs(expected_fc) >= stable_thr:
            asym = calculate_asymmetry(sp_df["log2fc"].values, expected_fc)
            asymmetry_dict[sp] = asym
    
    error_dict = {}
    
    # === VARIABLE PROTEOME ===
    var_df = res[np.abs(res["true_log2fc"]) >= stable_thr].copy()
    var_overall = {}
    var_species_rows = []
    
    if not var_df.empty:
        var_df["observed_regulated"] = var_df["regulation"].isin(["up", "down"])
        var_df["true_regulated"] = np.abs(var_df["true_log2fc"]) >= stable_thr
        var_df["within_tolerance"] = np.abs(var_df["log2fc"] - var_df["true_log2fc"]) <= fc_tolerance
        
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
    
    # === STABLE PROTEOME ===
    stab_df = res[np.abs(res["true_log2fc"]) < stable_thr].copy()
    stab_overall = {}
    stab_species_rows = []
    
    if not stab_df.empty:
        stab_df["significant"] = stab_df[test_col] < p_threshold
        stab_df["outside_tolerance"] = np.abs(stab_df["log2fc"] - stab_df["true_log2fc"]) > fc_tolerance
        
        true_fp = int((stab_df["significant"] & stab_df["outside_tolerance"]).sum())
        tn = int((~stab_df["significant"] | ~stab_df["outside_tolerance"]).sum())
        
        total = len(stab_df)
        fpr = true_fp / total if total > 0 else 0.0
        
        stab_overall = {"Total": total, "FP": true_fp, "TN": tn, "FPR": fpr}
        
        for sp in stab_df["species"].unique():
            sp_df = stab_df[stab_df["species"] == sp].copy()
            sp_df["significant"] = sp_df[test_col] < p_threshold
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
            sp_df["significant"] = sp_df[test_col] < p_threshold
            sp_df["outside_tolerance"] = np.abs(sp_df["log2fc"] - sp_df["true_log2fc"]) > fc_tolerance
            
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
st.markdown("Welch's t-test + Limma-style EB + Non-parametric tests + Spike-in validation")
st.markdown("---")

# ============================================================================
# DATA VALIDATION
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
# SIDEBAR CONFIG
# ============================================================================

st.sidebar.subheader("⚙️ Configuration")

# Comparison setup
ref_cond = st.sidebar.selectbox("Condition A (reference)", options=conditions, index=0)
treat_cond = st.sidebar.selectbox(
    "Condition B (treatment)",
    options=[c for c in conditions if c != ref_cond],
    index=0 if len(conditions) > 1 else 0
)

if ref_cond == treat_cond:
    st.error("❌ Choose two different conditions.")
    st.stop()

ref_samples = cond_samples[ref_cond]
treat_samples = cond_samples[treat_cond]

st.sidebar.markdown("---")

# Statistical tests
col1, col2 = st.sidebar.columns(2)
with col1:
    use_limma = st.checkbox("✓ Limma EB", value=True, help="Empirical Bayes variance shrinkage")
with col2:
    use_fdr = st.checkbox("✓ FDR (BH)", value=True)

col1, col2 = st.sidebar.columns(2)
with col1:
    use_parametric = st.checkbox("✓ Parametric", value=True, help="Welch's t-test")
with col2:
    use_nonparametric = st.checkbox("✓ Non-param", value=True, help="Mann-Whitney U")

p_thr = st.sidebar.slider("P-value threshold", 0.001, 0.1, 0.05, 0.001)
fc_thr = st.sidebar.slider("Log2 FC threshold (|x|)", 0.0, 3.0, 0.5, 0.1)

st.sidebar.markdown("---")

# Filtering
min_intensity = st.sidebar.slider("Min mean intensity", 0, 20, 0, 1)
min_var_pct = st.sidebar.slider("Min variance percentile", 0, 50, 25, 5)

st.sidebar.markdown("---")

# Visualizations
viz_options = st.sidebar.multiselect(
    "Visualizations:",
    [
        "Volcano Plot",
        "MA Plot (Faceted)",
        "MA Plot (Combined)",
        "Density Plot",
        "Box Plots (Top Proteins)",
        "Heatmap",
        "Cohen's d Distribution",
        "P-value Distribution",
        "Parametric vs Non-parametric"
    ],
    default=["Volcano Plot", "MA Plot (Combined)"]
)

# Spike-in
use_spike_in = st.sidebar.checkbox("✓ Spike-in Validation", value=False)

if use_spike_in:
    st.sidebar.markdown("**Spike-in Composition**")
    species_values = sorted([s for s in df[species_col].unique() if isinstance(s, str) and s.strip()])
    
    theoretical_fc_sidebar: Dict[str, float] = {}
    for sp in species_values:
        comp_a = st.sidebar.slider(f"{sp} in {ref_cond} (%)", 0.0, 100.0, 100.0/max(len(species_values),1), 5.0, key=f"sidebar_a_{sp}")
        comp_b = st.sidebar.slider(f"{sp} in {treat_cond} (%)", 0.0, 100.0, 100.0/max(len(species_values),1), 5.0, key=f"sidebar_b_{sp}")
        
        if comp_a > 0 and comp_b > 0:
            theoretical_fc_sidebar[sp] = np.log2(comp_a / comp_b)
    
    if st.sidebar.button("💾 Save Spike-in FC"):
        st.session_state.dea_theoretical_fc = theoretical_fc_sidebar.copy()
        st.sidebar.success("✅ Saved!")

st.sidebar.markdown("---")

# ============================================================================
# RUN ANALYSIS
# ============================================================================

st.subheader("1️⃣ Running Analysis")

if st.button("🚀 Run DEA", type="primary"):
    with st.spinner("⏳ Processing..."):
        # Log2 transform
        df_num = df[numeric_cols]
        if (df_num > 50).any().any():
            df_test = np.log2(df_num + 1.0)
        else:
            df_test = df_num.copy()
        
        # Filter
        mask_intensity = (df_test[ref_samples + treat_samples].mean(axis=1) >= min_intensity)
        
        var_pct_threshold = np.percentile(
            df_test[ref_samples + treat_samples].var(axis=1, ddof=1),
            min_var_pct
        )
        mask_variance = (df_test[ref_samples + treat_samples].var(axis=1, ddof=1) >= var_pct_threshold)
        
        df_filtered = df_test[mask_intensity & mask_variance]
        
        # Run DEA
        results = perform_dea(
            df_filtered,
            ref_samples,
            treat_samples,
            use_limma=use_limma,
            use_parametric=use_parametric,
            use_nonparametric=use_nonparametric
        )
        
        # Determine test column
        if use_parametric:
            test_col = "q_ttest" if use_fdr else "p_ttest"
        else:
            test_col = "q_mannwhitney" if use_fdr else "p_mannwhitney"
        
        # Classify regulation
        results["regulation"] = results.apply(
            lambda row: "up" if (row[test_col] < p_thr and row["log2fc"] > fc_thr)
            else ("down" if (row[test_col] < p_thr and row["log2fc"] < -fc_thr)
            else "not_significant"),
            axis=1
        )
        results["regulation"] = results.apply(
            lambda row: "not_tested" if pd.isna(row["log2fc"]) else row["regulation"],
            axis=1
        )
        results["neg_log10_p"] = -np.log10(results[test_col].replace(0, 1e-300))
        results["species"] = results.index.map(df[species_col])
        results["mean_intensity"] = (results["mean_g1"] + results["mean_g2"]) / 2
        
        st.session_state.dea_results = results
        st.session_state.dea_ref = ref_cond
        st.session_state.dea_treat = treat_cond
        st.session_state.dea_p_thr = p_thr
        st.session_state.dea_fc_thr = fc_thr
        st.session_state.dea_test_col = test_col
        st.session_state.dea_use_fdr = use_fdr
        st.session_state.dea_use_limma = use_limma
        st.session_state.dea_use_parametric = use_parametric
    
    st.success("✅ Analysis complete!")

# ============================================================================
# RESULTS
# ============================================================================

if "dea_results" in st.session_state:
    res = st.session_state.dea_results
    ref_cond = st.session_state.dea_ref
    treat_cond = st.session_state.dea_treat
    p_thr = st.session_state.dea_p_thr
    fc_thr = st.session_state.dea_fc_thr
    test_col = st.session_state.dea_test_col
    use_fdr = st.session_state.dea_use_fdr
    use_limma_flag = st.session_state.get('dea_use_limma', True)
    use_param = st.session_state.get('dea_use_parametric', True)
    theoretical_fc = st.session_state.get('dea_theoretical_fc', {})
    
    test_label = "Welch's t-test" if use_param else "Mann-Whitney U"
    pval_threshold_label = f"{p_thr:.3f}"
    
    st.markdown("---")
    st.subheader("2️⃣ Results Summary")
    
    n_total = len(res)
    n_tested = int((res["regulation"] != "not_tested").sum())
    n_sig_up = int((res["regulation"] == "up").sum())
    n_sig_down = int((res["regulation"] == "down").sum())
    n_sig = n_sig_up + n_sig_down
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Proteins", f"{n_total:,}")
    with col2:
        st.metric("Tested", f"{n_tested:,}")
    with col3:
        st.metric("Significant", f"{n_sig:,}")
    with col4:
        st.metric("↑ Up", f"{n_sig_up:,}")
    with col5:
        st.metric("↓ Down", f"{n_sig_down:,}")
    with col6:
        st.metric("Method", f"{'Limma' if use_limma_flag else 'Welch'}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Test", test_label)
    with col2:
        st.metric("Correction", "FDR" if use_fdr else "Raw p")
    with col3:
        st.metric("p-threshold", pval_threshold_label)
    with col4:
        st.metric("|logFC|", f">{fc_thr:.1f}")
    
    # Validation metrics if spike-in provided
    if theoretical_fc:
        species_series = df[species_col]
        res_all = res[res["regulation"] != "not_tested"].copy()
        res_all["species"] = res_all.index.map(species_series)
        res_all["true_log2fc"] = res_all["species"].map(theoretical_fc)
        res_all = res_all.dropna(subset=["true_log2fc"])
        
        if not res_all.empty:
            stable_thr = 0.5
            res_all["true_regulated"] = np.abs(res_all["true_log2fc"]) >= stable_thr
            res_all["observed_regulated"] = res_all["regulation"].isin(["up", "down"])
            
            tp = int((res_all["true_regulated"] & res_all["observed_regulated"]).sum())
            fn = int((res_all["true_regulated"] & ~res_all["observed_regulated"]).sum())
            tn = int((~res_all["true_regulated"] & ~res_all["observed_regulated"]).sum())
            fp = int((~res_all["true_regulated"] & res_all["observed_regulated"]).sum())
            
            sens = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
            prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Sensitivity", f"{sens:.1f}%")
            with col2:
                st.metric("Specificity", f"{spec:.1f}%")
            with col3:
                st.metric("Precision", f"{prec:.1f}%")
            with col4:
                st.metric("True Positives", tp)
    
    st.markdown("---")
    
    # ============================================================================
    # VISUALIZATIONS
    # ============================================================================
    
    st.subheader("3️⃣ Visualizations")
    
    # === VOLCANO PLOT ===
    if "Volcano Plot" in viz_options:
        st.markdown("### 🌋 Volcano Plot")
        
        volc = res[res["regulation"] != "not_tested"].dropna(subset=['neg_log10_p', 'log2fc'])
        
        if len(volc) > 0:
            fig_v = px.scatter(
                volc,
                x="log2fc",
                y="neg_log10_p",
                color="regulation",
                color_discrete_map={"up": "#e74c3c", "down": "#3498db", "not_significant": "#bdc3c7"},
                hover_data=["species", "mean_intensity", "cohens_d"],
                labels={"log2fc": f"log2({ref_cond}/{treat_cond})", "neg_log10_p": "-log10(p)"},
                height=600,
            )
            
            fig_v.add_hline(y=-np.log10(p_thr), line_dash="dash", line_color="gray", line_width=2)
            fig_v.add_vline(x=fc_thr, line_dash="dash", line_color="gray", line_width=1, opacity=0.5)
            fig_v.add_vline(x=-fc_thr, line_dash="dash", line_color="gray", line_width=1, opacity=0.5)
            
            st.plotly_chart(fig_v, width="stretch")
        else:
            st.warning("⚠️ No data for volcano plot after filtering")
    
    # === MA PLOT (FACETED) ===
    if "MA Plot (Faceted)" in viz_options:
        st.markdown("### 📊 MA Plot (Faceted by Species)")
        
        ma = res[res["regulation"] != "not_tested"].copy()
        ma = ma.dropna(subset=['log2fc', 'species'])
        
        species_list = sorted(ma["species"].unique())
        
        # BUGFIX: Check if species_list is empty
        if len(species_list) > 0:
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
                
                # Non-sig
                if len(sp_nonsig) > 0:
                    fig_facet.add_trace(
                        go.Scatter(
                            x=sp_nonsig["mean_intensity"],
                            y=sp_nonsig["log2fc"],
                            mode='markers',
                            marker=dict(size=3, color="lightgray", opacity=0.2),
                            name="ns",
                            showlegend=False,
                            hovertemplate=f"{sp}<br>Intensity=%{{x:.2f}}<br>log2FC=%{{y:.3f}}<extra></extra>"
                        ),
                        row=1, col=i
                    )
                
                # Up
                sp_up = sp_sig[sp_sig["regulation"] == "up"]
                if len(sp_up) > 0:
                    fig_facet.add_trace(
                        go.Scatter(
                            x=sp_up["mean_intensity"],
                            y=sp_up["log2fc"],
                            mode='markers',
                            marker=dict(size=4, color=color, opacity=0.8, symbol='circle'),
                            name="↑",
                            showlegend=(i==1),
                            hovertemplate=f"{sp} ↑<br>Intensity=%{{x:.2f}}<br>log2FC=%{{y:.3f}}<extra></extra>"
                        ),
                        row=1, col=i
                    )
                
                # Down
                sp_down = sp_sig[sp_sig["regulation"] == "down"]
                if len(sp_down) > 0:
                    fig_facet.add_trace(
                        go.Scatter(
                            x=sp_down["mean_intensity"],
                            y=sp_down["log2fc"],
                            mode='markers',
                            marker=dict(size=4, color=color, opacity=0.8, symbol='diamond'),
                            name="↓",
                            showlegend=(i==1),
                            hovertemplate=f"{sp} ↓<br>Intensity=%{{x:.2f}}<br>log2FC=%{{y:.3f}}<extra></extra>"
                        ),
                        row=1, col=i
                    )
                
                # Expected FC lines if spike-in
                if theoretical_fc and sp in theoretical_fc:
                    expected_fc = theoretical_fc[sp]
                    fc_tolerance = 0.58
                    
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
                
                fig_facet.add_hline(y=0, line_color="red", line_width=1, opacity=0.5, row=1, col=i)
            
            fig_facet.update_xaxes(title_text="log10(Intensity)", row=1, col=1)
            fig_facet.update_yaxes(title_text=f"log2({ref_cond}/{treat_cond})", row=1, col=1)
            fig_facet.update_layout(height=500, showlegend=True)
            
            st.plotly_chart(fig_facet, width="stretch")
        else:
            st.warning("⚠️ No species data for faceted MA plot")
    
    # === MA PLOT (COMBINED) ===
    if "MA Plot (Combined)" in viz_options:
        st.markdown("### 📈 MA Plot (Combined)")
        
        ma_plot = res[res["regulation"] != "not_tested"].dropna(subset=['log2fc'])
        
        if len(ma_plot) > 0:
            fig_ma = px.scatter(
                ma_plot,
                x="mean_intensity",
                y="log2fc",
                color="regulation",
                color_discrete_map={"up": "#e74c3c", "down": "#3498db", "not_significant": "#bdc3c7"},
                hover_data=["species", "cohens_d"],
                labels={"mean_intensity": "Mean Intensity", "log2fc": f"log2({ref_cond}/{treat_cond})"},
                height=600,
            )
            
            fig_ma.add_hline(y=0.0, line_color="red", line_width=1, opacity=0.5)
            fig_ma.add_hline(y=fc_thr, line_dash="dash", line_color="green", opacity=0.3)
            fig_ma.add_hline(y=-fc_thr, line_dash="dash", line_color="green", opacity=0.3)
            
            st.plotly_chart(fig_ma, width="stretch")
        else:
            st.warning("⚠️ No data for MA plot")
    
    # === DENSITY PLOT ===
    if "Density Plot" in viz_options:
        st.markdown("### 📊 Density Plot (log2FC Distribution)")
        
        density_data = res[res["regulation"] != "not_tested"].dropna(subset=["log2fc", "species"])
        
        if len(density_data) > 0:
            fig_density = go.Figure()
            
            for sp in sorted(density_data["species"].unique()):
                sp_data = density_data[density_data["species"] == sp]["log2fc"]
                if len(sp_data) > 1:
                    color = SPECIES_COLORS.get(sp, "#95a5a6")
                    
                    kde = gaussian_kde(sp_data)
                    x_range = np.linspace(sp_data.min(), sp_data.max(), 200)
                    density_vals = kde(x_range)
                    
                    fig_density.add_trace(go.Scatter(
                        x=x_range,
                        y=density_vals,
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
                            annotation_text=sp,
                            annotation_position="top"
                        )
            
            fig_density.update_layout(
                xaxis_title=f"log2({ref_cond}/{treat_cond})",
                yaxis_title="Density",
                height=500
            )
            
            st.plotly_chart(fig_density, width="stretch")
        else:
            st.warning("⚠️ No data for density plot")
    
    # === BOX PLOTS ===
    if "Box Plots (Top Proteins)" in viz_options:
        st.markdown("### 📦 Box Plots - Top Differentially Abundant Proteins")
        
        res_sorted = res.sort_values(test_col)
        n_top = min(9, len(res_sorted))
        
        if n_top > 0:
            top_proteins = res_sorted.head(n_top).index.values
            
            n_cols = min(3, n_top)
            n_rows = (n_top + n_cols - 1) // n_cols
            
            fig_box = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=top_proteins,
                specs=[[{'type': 'box'} for _ in range(n_cols)] for _ in range(n_rows)]
            )
            
            for idx, protein in enumerate(top_proteins):
                row = idx // n_cols + 1
                col = idx % n_cols + 1
                
                data_ref = df.loc[protein, ref_samples].values
                data_treat = df.loc[protein, treat_samples].values
                
                fig_box.add_trace(
                    go.Box(y=data_ref, name=ref_cond, marker_color='#3498db', showlegend=(idx==0)),
                    row=row, col=col
                )
                fig_box.add_trace(
                    go.Box(y=data_treat, name=treat_cond, marker_color='#e74c3c', showlegend=(idx==0)),
                    row=row, col=col
                )
            
            fig_box.update_yaxes(title_text="Intensity")
            fig_box.update_layout(height=300*n_rows, showlegend=True)
            
            st.plotly_chart(fig_box, width="stretch")
        else:
            st.warning("⚠️ No top proteins for box plots")
    
    # === HEATMAP ===
    if "Heatmap" in viz_options:
        st.markdown("### 🔥 Heatmap - Top Differentially Abundant Proteins")
        
        res_sorted = res.sort_values(test_col)
        n_top = min(20, len(res_sorted))
        
        if n_top > 0:
            top_proteins = res_sorted.head(n_top).index.values
            
            heatmap_data = df.loc[top_proteins, ref_samples + treat_samples].copy()
            
            # Z-score normalize
            heatmap_z = heatmap_data.T.apply(lambda x: (x - x.mean()) / (x.std() + 1e-6) if x.std() > 0 else x).T
            
            fig_heat = go.Figure(data=go.Heatmap(
                z=heatmap_z.values,
                x=heatmap_z.columns,
                y=heatmap_z.index,
                colorscale='RdBu_r',
                zmid=0,
                colorbar=dict(title="Z-score")
            ))
            
            fig_heat.update_layout(
                title=f'Top {n_top} Proteins (Z-score normalized)',
                xaxis_title='Sample',
                yaxis_title='Protein',
                height=600
            )
            
            st.plotly_chart(fig_heat, width="stretch")
        else:
            st.warning("⚠️ No proteins for heatmap")
    
    # === COHEN'S D DISTRIBUTION ===
    if "Cohen's d Distribution" in viz_options:
        st.markdown("### 📊 Effect Size Distribution (Cohen's d)")
        
        cohens_data = res[res["cohens_d"].notna()]
        
        if len(cohens_data) > 0:
            fig_cohens = px.histogram(
                cohens_data,
                x='cohens_d',
                nbins=50,
                title="Cohen's d Distribution",
                labels={'cohens_d': "Cohen's d"},
                height=500
            )
            fig_cohens.add_vline(x=0.2, line_dash="dash", annotation_text="Small (0.2)", line_color="gray")
            fig_cohens.add_vline(x=0.5, line_dash="dash", annotation_text="Medium (0.5)", line_color="orange")
            fig_cohens.add_vline(x=0.8, line_dash="dash", annotation_text="Large (0.8)", line_color="red")
            
            st.plotly_chart(fig_cohens, width="stretch")
        else:
            st.warning("⚠️ No Cohen's d data")
    
    # === P-VALUE DISTRIBUTION ===
    if "P-value Distribution" in viz_options:
        st.markdown("### 📊 P-value Distribution")
        
        pval_data = res[res[test_col].notna()]
        
        if len(pval_data) > 0:
            fig_pval = px.histogram(
                pval_data,
                x=test_col,
                nbins=50,
                title=f"{test_label} P-value Distribution",
                labels={test_col: "P-value" if not use_fdr else "FDR (q-value)"},
                height=500
            )
            fig_pval.add_vline(x=p_thr, line_dash="dash", annotation_text=f"Threshold={p_thr}", line_color="red")
            
            st.plotly_chart(fig_pval, width="stretch")
        else:
            st.warning("⚠️ No p-value data")
    
    # === PARAMETRIC VS NON-PARAMETRIC COMPARISON ===
    if "Parametric vs Non-parametric" in viz_options and use_parametric and use_nonparametric:
        st.markdown("### 🔍 Parametric vs Non-parametric Comparison")
        
        comp_data = res[res['p_ttest'].notna() & res['p_mannwhitney'].notna()]
        
        if len(comp_data) > 0:
            corr = comp_data[['p_ttest', 'p_mannwhitney']].corr().iloc[0, 1]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                sig_ttest = ((comp_data['q_ttest'] < p_thr) & (comp_data['log2fc'].abs() > fc_thr)).sum()
                st.metric("Significant (t-test)", sig_ttest)
            with col2:
                sig_mw = ((comp_data['q_mannwhitney'] < p_thr) & (comp_data['log2fc'].abs() > fc_thr)).sum()
                st.metric("Significant (MW)", sig_mw)
            with col3:
                st.metric("P-value Correlation", f"{corr:.3f}")
            
            fig_comp = px.scatter(
                comp_data,
                x='p_ttest',
                y='p_mannwhitney',
                hover_data=['species', 'log2fc'],
                title="P-value Comparison",
                labels={'p_ttest': "Welch's p", 'p_mannwhitney': 'Mann-Whitney p'},
                height=600
            )
            fig_comp.update_xaxes(type='log')
            fig_comp.update_yaxes(type='log')
            
            st.plotly_chart(fig_comp, width="stretch")
        else:
            st.warning("⚠️ Insufficient data for comparison")
    
    # ============================================================================
    # SPIKE-IN VALIDATION
    # ============================================================================
    
    if theoretical_fc:
        st.markdown("---")
        st.subheader("4️⃣ Spike-in Validation")
        
        st.info(f"✓ Using: {', '.join(f'{k}={v:.2f}' for k, v in theoretical_fc.items())}")
        
        species_series = df[species_col]
        stable_thr = 0.5
        fc_tolerance = 0.58
        
        var_ov, var_sp, stab_ov, stab_sp, asym_dict, error_dict, fp_var_sp = compute_species_metrics(
            res, theoretical_fc, species_series, stable_thr, fc_tolerance, p_thr, test_col
        )
        
        # Validation metrics table
        validation_rows = []
        
        for sp in theoretical_fc.keys():
            asym_val = asym_dict.get(sp, np.nan)
            if not np.isnan(asym_val):
                validation_rows.append({
                    "Metric": "Asymmetry",
                    "Species": sp,
                    "Value": f"{asym_val:.2f}",
                    "Category": "Quality"
                })
        
        for sp in theoretical_fc.keys():
            fp_count = error_dict.get(f"FP_{sp}", 0)
            if fp_count > 0:
                validation_rows.append({
                    "Metric": "False Positives",
                    "Species": sp,
                    "Value": f"{fp_count}",
                    "Category": "Error"
                })
        
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
        
        if validation_rows:
            val_df = pd.DataFrame(validation_rows)
            st.dataframe(val_df, use_container_width=True, hide_index=True)
    
    # ============================================================================
    # RESULTS TABLE
    # ============================================================================
    
    st.markdown("---")
    st.subheader("5️⃣ Detailed Results")
    
    with st.expander("📋 View All Results", expanded=False):
        display_cols = [
            "log2fc", "cohens_d", "mean_g1", "mean_g2",
            "p_ttest", "q_ttest", "p_mannwhitney", "q_mannwhitney",
            "species", "regulation"
        ]
        display_cols = [c for c in display_cols if c in res.columns]
        
        display_df = res[display_cols].copy()
        display_df = display_df.round(4)
        display_df = display_df.sort_values(test_col)
        
        st.dataframe(display_df, use_container_width=True, height=600)
        
        csv_data = display_df.to_csv(index=True).encode('utf-8')
        st.download_button(
            "📥 Download Results (CSV)",
            data=csv_data,
            file_name=f"dea_{ref_cond}_vs_{treat_cond}.csv",
            mime="text/csv"
        )
    
    # ============================================================================
    # INTERPRETATION GUIDE
    # ============================================================================
    
    st.markdown("---")
    st.subheader("📖 Interpretation Guide")
    
    with st.expander("**Cohen's d Effect Size**"):
        st.markdown("""
        - |d| < 0.2: **Small** effect
        - 0.2 ≤ |d| < 0.5: **Small-Medium**
        - 0.5 ≤ |d| < 0.8: **Medium-Large**
        - |d| ≥ 0.8: **Large** effect
        """)
    
    with st.expander("**Log Fold-Change**"):
        st.markdown(f"""
        - logFC = 1 → 2-fold change
        - logFC = 2 → 4-fold change
        - logFC = -1 → 0.5-fold (50% decrease)
        
        **Current threshold**: |logFC| > {fc_thr}
        """)
    
    with st.expander("**Statistical Thresholds**"):
        st.markdown(f"""
        - **P-value**: < {p_thr}
        - **Correction**: {'FDR (Benjamini-Hochberg)' if use_fdr else 'Raw p-value'}
        - **Test**: {test_label} {'+ Limma EB' if use_limma_flag else ''}
        """)
    
    with st.expander("**Unified False Positive Definition** (Spike-in)"):
        st.markdown("""
        For **ALL species**:
        - p < 0.01 **AND** |log2fc - expected| > ±0.58
        
        This identifies proteins that are:
        - Statistically significant **BUT**
        - Have magnitude errors > ±0.58 (∼1.5 fold away from true value)
        """)
    
    st.success("✅ Analysis complete! Export results for downstream analysis.")

else:
    st.info("👆 Configure comparison and run analysis above")
