"""
pages/6_Differential_Abundance.py
Welch's t-test DA with comprehensive visualization and spike-in validation.
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

from scipy.stats import t as t_dist, ttest_ind
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------
# COLOR SCHEME
# ---------------------------------------------------------------------

SPECIES_COLORS = {
    "HUMAN": "#2ecc71",  # Green
    "YEAST": "#e67e22",  # Orange
    "ECOLI": "#9b59b6",  # Purple
}

# ---------------------------------------------------------------------
# STATISTICAL HELPERS
# ---------------------------------------------------------------------


def perform_ttest(
    df: pd.DataFrame,
    group1_cols: List[str],
    group2_cols: List[str],
    min_valid: int = 2,
) -> pd.DataFrame:
    """Perform Welch's t-test. Limma convention: log2FC = mean(group1) - mean(group2)"""
    results = []
    
    for protein_id in df.index:
        row = df.loc[protein_id]
        g1_vals = row[group1_cols].dropna()
        g2_vals = row[group2_cols].dropna()
        
        if len(g1_vals) < min_valid or len(g2_vals) < min_valid:
            results.append({
                "protein_id": protein_id,
                "log2fc": np.nan,
                "pvalue": np.nan,
                "mean_g1": np.nan,
                "mean_g2": np.nan,
                "n_g1": len(g1_vals),
                "n_g2": len(g2_vals),
            })
            continue
        
        mean_g1 = g1_vals.mean()
        mean_g2 = g2_vals.mean()
        log2fc = mean_g1 - mean_g2
        
        try:
            t_stat, pval = ttest_ind(g1_vals, g2_vals, equal_var=False)
        except Exception:
            t_stat, pval = np.nan, np.nan
        
        results.append({
            "protein_id": protein_id,
            "log2fc": log2fc,
            "pvalue": pval,
            "mean_g1": mean_g1,
            "mean_g2": mean_g2,
            "n_g1": len(g1_vals),
            "n_g2": len(g2_vals),
        })
    
    results_df = pd.DataFrame(results).set_index("protein_id")
    
    # FDR Correction
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
    
    False Positives (ALL SPECIES): p<p_threshold AND |log2fc - expected| > fc_tolerance
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
        
        # CORRECT CONFUSION MATRIX
        tp = int((var_df["true_regulated"] & var_df["observed_regulated"]).sum())
        fn = int((var_df["true_regulated"] & ~var_df["observed_regulated"]).sum())
        tn = int((~var_df["true_regulated"] & ~var_df["observed_regulated"]).sum())
        fp = int((~var_df["true_regulated"] & var_df["observed_regulated"]).sum())
        
        # Correct calculations
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TN / (TN + FP) - CORRECT
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
        
        # True FP: p<threshold AND |error| > ±0.58 (significant but wrong magnitude)
        true_fp = int((stab_df["significant"] & stab_df["outside_tolerance"]).sum())
        
        # Correct TN: not significant OR within ±0.58
        tn = int((~stab_df["significant"] | ~stab_df["outside_tolerance"]).sum())
        
        total = len(stab_df)
        fpr = true_fp / total if total > 0 else 0.0
        
        stab_overall = {"Total": total, "FP": true_fp, "TN": tn, "FPR": fpr}
        
        for sp in stab_df["species"].unique():
            sp_df = stab_df[stab_df["species"] == sp].copy()
            sp_df["significant"] = sp_df["pvalue"] < p_threshold
            sp_df["outside_tolerance"] = np.abs(sp_df["log2fc"] - sp_df["true_log2fc"]) > fc_tolerance
            
            # True FP for this species
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
    
    # === FALSE POSITIVES IN VARIABLE PROTEOME (p<threshold but wrong magnitude) ===
    # For ECOLI/YEAST: p<p_threshold BUT |log2fc - expected| > fc_tolerance
    fp_var_species_rows = []
    
    if not var_df.empty:
        for sp in var_df["species"].unique():
            if sp != "HUMAN":  # Only for variable species
                sp_df = var_df[var_df["species"] == sp].copy()
                sp_df["significant"] = sp_df["pvalue"] < p_threshold
                sp_df["outside_tolerance"] = np.abs(sp_df["log2fc"] - sp_df["true_log2fc"]) > fc_tolerance
                
                # FP: p<threshold AND outside ±0.58 (wrong magnitude)
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


# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------

st.set_page_config(page_title="Differential Abundance", page_icon="🔬", layout="wide")
st.title("🔬 Differential Abundance Analysis")
st.markdown("Welch's t-test on A vs B with spike-in validation.")
st.markdown("---")

# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------

if "df_imputed" not in st.session_state or st.session_state.df_imputed is None:
    st.error("No imputed data. Complete Missing Value Imputation first.")
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

st.info(f"Data: {df.shape[0]:,} proteins × {len(numeric_cols)} samples · Conditions: {', '.join(conditions)}")

# ---------------------------------------------------------------------
# 1. COMPARISON
# ---------------------------------------------------------------------

st.subheader("1️⃣ Comparison Setup (A vs B)")

col1, col2 = st.columns(2)
with col1:
    ref_cond = st.selectbox("Condition A (reference)", options=conditions, index=0)
with col2:
    treat_cond = st.selectbox("Condition B (treatment)", options=[c for c in conditions if c != ref_cond], index=0 if len(conditions) > 1 else 0)

if ref_cond == treat_cond:
    st.error("Choose two different conditions.")
    st.stop()

ref_samples = cond_samples[ref_cond]
treat_samples = cond_samples[treat_cond]

st.markdown(f"- Log2FC = **{ref_cond}/{treat_cond}** (positive = higher in A)")
st.markdown("---")

# ---------------------------------------------------------------------
# 2. SPIKE-IN
# ---------------------------------------------------------------------

st.subheader("2️⃣ Spike-in Composition (optional)")

use_comp = st.checkbox("Provide % composition per species", value=False)

theoretical_fc_temp: Dict[str, float] = {}
species_values = sorted([s for s in df[species_col].unique() if isinstance(s, str) and s != 'Unknown'])

if use_comp:
    st.markdown("Enter % composition (normalized to 100% per condition).")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{ref_cond} (A)**")
        comp_a = {}
        for sp in species_values:
            val = st.number_input(f"{sp} (%) in {ref_cond}", min_value=0.0, max_value=100.0, value=100.0/max(len(species_values),1), step=5.0, key=f"a_{sp}")
            comp_a[sp] = val
        ta = sum(comp_a.values()) or 1.0
        comp_a = {k: v*100/ta for k, v in comp_a.items()}
    
    with c2:
        st.markdown(f"**{treat_cond} (B)**")
        comp_b = {}
        for sp in species_values:
            val = st.number_input(f"{sp} (%) in {treat_cond}", min_value=0.0, max_value=100.0, value=100.0/max(len(species_values),1), step=5.0, key=f"b_{sp}")
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
        rows.append({"Species": sp, f"{ref_cond} (%)": f"{pa:.1f}", f"{treat_cond} (%)": f"{pb:.1f}", "Log2FC": f"{log2fc:.3f}", "Linear_FC": f"{2**log2fc:.2f}x"})
    
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    
    if st.button("💾 Save Expected FC", type="primary"):
        st.session_state.dea_theoretical_fc = theoretical_fc_temp.copy()
        st.success(f"✅ Saved for {len(theoretical_fc_temp)} species!")
    
    saved_fc = st.session_state.get('dea_theoretical_fc', {})
    if saved_fc:
        st.info(f"✓ Saved: {', '.join(f'{k}={v:.2f}' for k,v in saved_fc.items())}")
    else:
        st.warning("⚠️ Not saved yet.")

st.markdown("---")

# ---------------------------------------------------------------------
# 3. SETTINGS
# ---------------------------------------------------------------------

st.subheader("3️⃣ Statistical Settings")

c1, c2 = st.columns(2)
with c1:
    p_thr = st.selectbox("FDR threshold", options=[0.01, 0.05], index=1, format_func=lambda x: f"{x*100:.0f}%")
with c2:
    use_fdr = st.checkbox("Use FDR correction (BH)", value=True)

stable_thr = 0.5
fc_tolerance = 0.58  # FP window
st.caption(f"Stable: |expected log2FC| < {stable_thr} · FP definition: p<0.01 AND |log2fc - expected| > ±{fc_tolerance}")
st.markdown("---")

# ---------------------------------------------------------------------
# 4. RUN
# ---------------------------------------------------------------------

st.subheader("4️⃣ Run Analysis")

if st.button("🚀 Run Welch's t-test", type="primary"):
    with st.spinner("Running..."):
        df_num = df[numeric_cols]
        if (df_num > 50).any().any():
            df_log2 = np.log2(df_num + 1.0)
        else:
            df_log2 = df_num.copy()
        
        results = perform_ttest(df_log2, ref_samples, treat_samples)
        test_col = "fdr" if use_fdr else "pvalue"
        
        results["regulation"] = results.apply(lambda row: classify_regulation(row["log2fc"], row[test_col], 0.0, p_thr), axis=1)
        results["neg_log10_p"] = -np.log10(results[test_col].replace(0, 1e-300))
        results["species"] = results.index.map(df[species_col])
        
        st.session_state.dea_results = results
        st.session_state.dea_ref = ref_cond
        st.session_state.dea_treat = treat_cond
        st.session_state.dea_p_thr = p_thr
        st.session_state.dea_use_fdr = use_fdr
    
    st.success("✅ Done!")

# ---------------------------------------------------------------------
# 5. RESULTS
# ---------------------------------------------------------------------

if "dea_results" in st.session_state:
    res = st.session_state.dea_results
    ref_cond = st.session_state.dea_ref
    treat_cond = st.session_state.dea_treat
    p_thr = st.session_state.dea_p_thr
    use_fdr = st.session_state.get('dea_use_fdr', True)
    theoretical_fc = st.session_state.get('dea_theoretical_fc', {})
    
    st.markdown("---")
    st.subheader("5️⃣ Results Overview")
    
    n_total = len(res)
    n_quant = int((res["regulation"] != "not_tested").sum())
    quant_rate = n_quant / n_total * 100 if n_total > 0 else 0
    n_ids = n_total
    
    # Calculate validation metrics across ALL species if theoretical FC available
    sens_pct = 0.0
    spec_pct = 0.0
    true_positives = 0
    de_fdr = 0.0
    total_fp = 0
    
    if theoretical_fc:
        species_series = df[species_col]
        
        # Get ALL tested proteins with species info
        res_all = res[res["regulation"] != "not_tested"].copy()
        res_all["species"] = res_all.index.map(species_series)
        res_all["true_log2fc"] = res_all["species"].map(theoretical_fc)
        
        # Only keep proteins with valid theoretical FC
        res_all = res_all.dropna(subset=["true_log2fc"])
        
        if not res_all.empty:
            # Classify based on theoretical FC
            res_all["true_regulated"] = np.abs(res_all["true_log2fc"]) >= stable_thr
            res_all["observed_regulated"] = res_all["regulation"].isin(["up", "down"])
            
            # Calculate confusion matrix across ALL species
            tp = int((res_all["true_regulated"] & res_all["observed_regulated"]).sum())
            fn = int((res_all["true_regulated"] & ~res_all["observed_regulated"]).sum())
            tn = int((~res_all["true_regulated"] & ~res_all["observed_regulated"]).sum())
            fp = int((~res_all["true_regulated"] & res_all["observed_regulated"]).sum())
            
            # Calculate metrics
            sens_pct = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
            spec_pct = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
            true_positives = tp
            de_fdr = fp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    
    # Header with all stats
    st.markdown(
        f"**{n_quant:,} Quant. ({quant_rate:.0f}%), {n_ids:,} IDs**  \n"
        f"**Sens. {sens_pct:.2f} %, Spec. {spec_pct:.2f} %**  \n"
        f"**{true_positives:,} true positives, {de_fdr:.2f}% deFDR**"
    )
    
    # Calculate asymmetry if theoretical FC available
    if theoretical_fc:
        asym_dict = {}
        for sp in theoretical_fc.keys():
            sp_df = res[res["species"] == sp].dropna(subset=["log2fc"])
            if len(sp_df) > 0 and abs(theoretical_fc[sp]) >= stable_thr:
                asym = calculate_asymmetry(sp_df["log2fc"].values, theoretical_fc[sp])
                asym_dict[sp] = asym
        
        if asym_dict:
            asymmetry_text = ", ".join([f"Asym. {k} {v:.2f}" for k, v in asym_dict.items()])
            st.markdown(f"**{asymmetry_text}**")
    
    # === PLOT 1: FACETED SCATTER (MA PLOT) - ALL POINTS ===
    st.markdown("### 📊 MA Plot (Faceted by Species)")
    
    ma = res[res["regulation"] != "not_tested"].copy()
    ma["A"] = (ma["mean_g1"] + ma["mean_g2"]) / 2
    ma = ma.dropna(subset=['A', 'log2fc', 'species'])
    
    # Create faceted plot
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
        
        # Significant points
        sp_sig = sp_data[sp_data["regulation"] != "not_significant"]
        sp_nonsig = sp_data[sp_data["regulation"] == "not_significant"]
        
        # Non-significant (grayed out)
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
        
        # Significant by regulation
        sp_up = sp_sig[sp_sig["regulation"] == "up"]
        sp_down = sp_sig[sp_sig["regulation"] == "down"]
        
        # Up-regulated (circles)
        fig_facet.add_trace(
            go.Scatter(
                x=sp_up["A"],
                y=sp_up["log2fc"],
                mode='markers',
                marker=dict(size=4, color=color, opacity=0.8, symbol='circle'),
                name=f"{sp} (↑)",
                showlegend=(i==1),
                hovertemplate=f"{sp} ↑<br>A=%{{x:.2f}}<br>log2FC=%{{y:.3f}}<extra></extra>"
            ),
            row=1, col=i
        )
        
        # Down-regulated (diamonds)
        fig_facet.add_trace(
            go.Scatter(
                x=sp_down["A"],
                y=sp_down["log2fc"],
                mode='markers',
                marker=dict(size=4, color=color, opacity=0.8, symbol='diamond'),
                name=f"{sp} (↓)",
                showlegend=(i==1),
                hovertemplate=f"{sp} ↓<br>A=%{{x:.2f}}<br>log2FC=%{{y:.3f}}<extra></extra>"
            ),
            row=1, col=i
        )
        
        # Add expected FC line if available
        if theoretical_fc and sp in theoretical_fc:
            expected_fc = theoretical_fc[sp]
            
            fig_facet.add_hline(
                y=expected_fc,
                line_dash="dash",
                line_color=color,
                line_width=2,
                row=1, col=i
            )
            
            # Add tolerance band (±0.58)
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
    
    fig_facet.update_xaxes(title_text="log2(B)", row=1, col=2)
    fig_facet.update_yaxes(title_text=f"log2(A:B)", row=1, col=1)
    fig_facet.update_layout(height=500, title_text="", showlegend=False)
    st.plotly_chart(fig_facet, use_container_width=True)
    
    # === PLOT 2: REGULAR SCATTER (MA PLOT) - ALL POINTS ===
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
        labels={"A": "log2(B)", "log2fc": f"log2({ref_cond}/{treat_cond})"},
        height=600,
    )
    
    # Gray out non-significant
    for trace in fig_ma.data:
        if trace.name == "Not Sig.":
            trace.marker.opacity = 0.2
            trace.marker.color = "lightgray"
    
    fig_ma.add_hline(y=0.0, line_color="red", line_width=1, opacity=0.5)
    
    if theoretical_fc:
        for species, expected_fc in theoretical_fc.items():
            color = SPECIES_COLORS.get(species, "#95a5a6")
            
            # Expected FC line
            fig_ma.add_hline(
                y=expected_fc,
                line_dash="dash",
                line_width=2,
                line_color=color,
                opacity=0.7,
                annotation_text=f"{species}",
                annotation_position="right",
            )
            
            # Tolerance band (±0.58)
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
    st.markdown("### 📊 Density Plot")
    
    density_data = res[res["regulation"] != "not_tested"].dropna(subset=["log2fc", "species"])
    
    fig_density = go.Figure()
    
    for sp in sorted(density_data["species"].unique()):
        sp_data = density_data[density_data["species"] == sp]["log2fc"]
        color = SPECIES_COLORS.get(sp, "#95a5a6")
        
        # Smooth with KDE
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
        
        # Add expected FC line
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
        yaxis_title="density",
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig_density, use_container_width=True)
    
    # === PLOT 4: VOLCANO - GRAY OUT NON-SIG + DASHED ±0.58 ===
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
        labels={"log2fc": f"log2 FC ({ref_cond}/{treat_cond})", "neg_log10_p": "-log10(FDR)" if use_fdr else "-log10(p)"},
        height=600,
    )
    
    # Gray out non-significant
    for i, trace in enumerate(fig_v.data):
        mask = volc[volc["species"] == trace.name]["Status"] == "Not Significant"
        if mask.any():
            trace.marker.opacity = np.where(mask, 0.15, 0.7)
    
    # FDR threshold line
    fig_v.add_hline(y=-np.log10(p_thr), line_dash="dash", line_color="gray", line_width=2, annotation_text=f"p={p_thr}")
    
    # Add ±0.58 dashed lines if theoretical FC available
    if theoretical_fc:
        fig_v.add_vline(x=fc_tolerance, line_dash="dot", line_color="gray", line_width=1, opacity=0.5)
        fig_v.add_vline(x=-fc_tolerance, line_dash="dot", line_color="gray", line_width=1, opacity=0.5)
    
    st.plotly_chart(fig_v, use_container_width=True)
    
    # ===== VALIDATION SECTION =====
    if theoretical_fc:
        st.markdown("---")
        st.subheader("6️⃣ Spike-in Validation")
        
        st.info(f"✓ Using: {', '.join(f'{k}={v:.2f}' for k,v in theoretical_fc.items())}")
        st.caption(f"FP definition: p<{p_thr} AND |log2fc - expected| > ±{fc_tolerance}")
        
        species_series = df[species_col]
        var_ov, var_sp, stab_ov, stab_sp, asym_dict, error_dict, fp_var_sp = compute_species_metrics(
            res, theoretical_fc, species_series, stable_thr, fc_tolerance, p_thr
        )
        
        # === UNIFIED VALIDATION TABLE ===
        validation_rows = []
        
        # Get initial total from uploaded data
        initial_total = len(df)
        
        # Asymmetry row
        for sp in ["HUMAN", "ECOLI", "YEAST"]:
            asym_val = asym_dict.get(sp, np.nan)
            validation_rows.append({
                "Metric": f"Asymmetry",
                "Species": sp,
                "Value": f"{asym_val:.2f}" if not np.isnan(asym_val) else "N/A",
                "Category": "Quality"
            })
        
        # FP metrics
        for sp in ["HUMAN", "ECOLI", "YEAST"]:
            fp_count = error_dict.get(f"FP_{sp}", 0)
            total = 0
            fpr = 0
            
            if sp in stab_sp["Species"].values:
                total = int(stab_sp[stab_sp["Species"] == sp]["N"].values[0])
                fpr = fp_count / total * 100 if total > 0 else 0
            elif sp in fp_var_sp["Species"].values:
                total = int(fp_var_sp[fp_var_sp["Species"] == sp]["Total_Detected"].values[0])
                fpr = fp_count / total * 100 if total > 0 else 0
            
            if total > 0:
                validation_rows.append({
                    "Metric": "False Positives",
                    "Species": sp,
                    "Value": f"{fp_count:,} / {total:,} ({fpr:.1f}%)",
                    "Category": "Error"
                })
        
        # Detection/Accuracy
        for sp in ["ECOLI", "YEAST"]:
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
        
        # Stable metrics (HUMAN FPR)
        if stab_ov:
            validation_rows.append({
                "Metric": "FPR (Stable)",
                "Species": "HUMAN",
                "Value": f"{stab_ov['FPR']:.1%}",
                "Category": "Specificity"
            })
        
        # Overall metrics - FIX SPECIFICITY CALCULATION
        if var_ov:
            # Sensitivity: TP / (TP + FN)
            validation_rows.append({
                "Metric": "Sensitivity (Variable)",
                "Species": "Overall",
                "Value": f"{var_ov['Sensitivity']:.1%}",
                "Category": "Sensitivity"
            })
            
            # Specificity: TN / (TN + FP) - correctly calculated
            spec_value = var_ov['Specificity']
            validation_rows.append({
                "Metric": "Specificity (Variable)",
                "Species": "Overall",
                "Value": f"{spec_value:.1%}",
                "Category": "Specificity"
            })
            
            # Precision: TP / (TP + FP)
            validation_rows.append({
                "Metric": "Precision",
                "Species": "Overall",
                "Value": f"{var_ov['Precision']:.1%}",
                "Category": "Precision"
            })
            
            # Sample counts
            validation_rows.append({
                "Metric": "Quantifiable Proteins",
                "Species": "Overall",
                "Value": f"{var_ov['Total']:,} / {initial_total:,}",
                "Category": "Data"
            })
        
        # deFDR
        if true_positives > 0 and total_fp > 0:
            defdr = total_fp / (true_positives + total_fp) * 100
            validation_rows.append({
                "Metric": "deFDR",
                "Species": "Overall",
                "Value": f"{defdr:.2f}%",
                "Category": "Error"
            })
        
        validation_df = pd.DataFrame(validation_rows)
        
        st.markdown("### 📊 Validation Metrics")
        st.dataframe(
            validation_df,
            use_container_width=True,
            column_config={
                "Metric": st.column_config.TextColumn(width=150),
                "Species": st.column_config.TextColumn(width=100),
                "Value": st.column_config.TextColumn(width=150),
                "Category": st.column_config.TextColumn(width=100),
            }
        )
        
        # Download validation metrics
        st.download_button(
            "📥 Download Validation Metrics",
            data=validation_df.to_csv(index=False).encode("utf-8"),
            file_name=f"validation_metrics_{ref_cond}_vs_{treat_cond}.csv",
            mime="text/csv",
        )

    
    # ===== INDIVIDUAL PROTEIN TABLE (LAZY LOADING WITH EXPANDER) =====
    st.markdown("---")
    
    with st.expander("🔬 View Individual Protein Results", expanded=False):
        st.markdown("### Individual Protein Data")
        st.caption(f"Complete results for all {len(res):,} proteins")
        
        # Prepare comprehensive table
        results_table = res.copy()
        results_table["species"] = results_table.index.map(df[species_col])
        results_table = results_table.reset_index()
        results_table = results_table.rename(columns={"protein_id": "Protein_ID"})
        
        # Round numeric columns
        results_table = results_table.round({
            "log2fc": 3,
            "pvalue": 6,
            "fdr": 6,
            "mean_g1": 2,
            "mean_g2": 2,
            "neg_log10_p": 2
        })
        
        # Reorder columns
        display_cols = [
            "Protein_ID", "species", "log2fc", "pvalue", "fdr", 
            "mean_g1", "mean_g2", "regulation", "neg_log10_p", "n_g1", "n_g2"
        ]
        
        results_table = results_table[[c for c in display_cols if c in results_table.columns]]
        
        # Display with sorting/filtering
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
        
        # Download button
        st.download_button(
            "📥 Download Individual Proteins (CSV)",
            data=results_table.to_csv(index=False).encode("utf-8"),
            file_name=f"dea_proteins_{ref_cond}_vs_{treat_cond}.csv",
            mime="text/csv",
            key="individual_download"
        )
else:
    st.info("👆 Configure and run analysis")
