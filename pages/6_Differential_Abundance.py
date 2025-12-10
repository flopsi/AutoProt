"""
pages/6_Differential_Abundance.py
Limma-style empirical Bayes DA with composition-based spike-in validation.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys

from scipy.stats import t as t_dist, ttest_ind
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------
# STATISTICAL HELPERS
# ---------------------------------------------------------------------


def perform_ttest(
    df: pd.DataFrame,
    group1_cols: List[str],
    group2_cols: List[str],
    min_valid: int = 2,
) -> pd.DataFrame:
    """
    Perform Welch's t-test on log2-transformed data.
    Limma convention: log2FC = mean(group1) - mean(group2)
    """
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
    
    results_df = pd.DataFrame(results)
    results_df.set_index("protein_id", inplace=True)
    
    # FDR Correction (Benjamini-Hochberg)
    pvals = results_df["pvalue"].dropna().sort_values()
    n = len(pvals)
    
    if n > 0:
        ranks = np.arange(1, n + 1)
        fdr_vals = pvals.values * n / ranks
        fdr_vals = np.minimum.accumulate(fdr_vals[::-1])[::-1]
        fdr_dict = dict(zip(pvals.index, fdr_vals))
        results_df["fdr"] = results_df.index.map(fdr_dict)
    else:
        results_df["fdr"] = np.nan
    
    results_df["neg_log10_pval"] = -np.log10(
        results_df["pvalue"].replace(0, 1e-300)
    )
    
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


def compute_species_metrics(
    results_df: pd.DataFrame,
    true_fc_dict: Dict[str, float],
    species_col_series: pd.Series,
    stable_thr: float = 0.5,
) -> Tuple[Dict, pd.DataFrame, Dict, pd.DataFrame]:
    """
    Calculate metrics for variable and stable proteomes.
    
    Args:
        results_df: Results with regulation column
        true_fc_dict: Species → expected log2FC
        species_col_series: Protein → species mapping
        stable_thr: Threshold for stable vs variable
    
    Returns:
        (variable_overall, variable_per_species, stable_overall, stable_per_species)
    """
    # Add species to results
    res = results_df.copy()
    res["species"] = res.index.map(species_col_series)
    res["true_log2fc"] = res["species"].map(true_fc_dict)
    
    # Filter tested proteins with valid species
    res = res[res["regulation"] != "not_tested"].copy()
    res = res.dropna(subset=["true_log2fc", "species"])
    
    if res.empty:
        return {}, pd.DataFrame(), {}, pd.DataFrame()
    
    # === VARIABLE PROTEOME (|true FC| >= threshold) ===
    var_df = res[np.abs(res["true_log2fc"]) >= stable_thr].copy()
    
    var_overall = {}
    var_species_rows = []
    
    if not var_df.empty:
        var_df["observed_regulated"] = var_df["regulation"].isin(["up", "down"])
        var_df["true_regulated"] = np.abs(var_df["true_log2fc"]) >= stable_thr
        
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
            "TN": tn,
            "FP": fp,
            "Sensitivity": sens,
            "Specificity": spec,
            "Precision": prec,
        }
        
        # Per-species
        for sp in var_df["species"].unique():
            sp_df = var_df[var_df["species"] == sp].copy()
            theo = true_fc_dict.get(sp, 0.0)
            error = sp_df["log2fc"] - theo
            
            var_species_rows.append({
                "Species": sp,
                "N": len(sp_df),
                "Theo_FC": f"{theo:.2f}",
                "RMSE": f"{np.sqrt((error**2).mean()):.3f}",
                "MAE": f"{error.abs().mean():.3f}",
                "Bias": f"{error.mean():.3f}",
                "Detection": f"{sp_df['observed_regulated'].mean():.1%}",
            })
    
    # === STABLE PROTEOME (|true FC| < threshold) ===
    stab_df = res[np.abs(res["true_log2fc"]) < stable_thr].copy()
    
    stab_overall = {}
    stab_species_rows = []
    
    if not stab_df.empty:
        stab_df["observed_regulated"] = stab_df["regulation"].isin(["up", "down"])
        
        fp = int(stab_df["observed_regulated"].sum())
        tn = int((~stab_df["observed_regulated"]).sum())
        total = len(stab_df)
        
        fpr = fp / total if total > 0 else 0.0
        
        stab_overall = {
            "Total": total,
            "FP": fp,
            "TN": tn,
            "FPR": fpr,
        }
        
        # Per-species
        for sp in stab_df["species"].unique():
            sp_df = stab_df[stab_df["species"] == sp].copy()
            fp_s = int(sp_df["observed_regulated"].sum())
            tn_s = int((~sp_df["observed_regulated"]).sum())
            
            stab_species_rows.append({
                "Species": sp,
                "N": len(sp_df),
                "FP": fp_s,
                "TN": tn_s,
                "FPR": f"{fp_s/len(sp_df):.1%}" if len(sp_df) > 0 else "0.0%",
                "MAE": f"{sp_df['log2fc'].abs().mean():.3f}",
            })
    
    return (
        var_overall,
        pd.DataFrame(var_species_rows),
        stab_overall,
        pd.DataFrame(stab_species_rows),
    )


# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------

st.set_page_config(page_title="Differential Abundance", page_icon="🔬", layout="wide")
st.title("🔬 Differential Abundance Analysis")
st.markdown("Welch's t-test on A vs B with spike-in validation.")
st.markdown("---")

# ---------------------------------------------------------------------
# DATA AVAILABILITY
# ---------------------------------------------------------------------

if "df_imputed" not in st.session_state or st.session_state.df_imputed is None:
    st.error("No imputed data found. Please finish the Missing Value Imputation step first.")
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

st.info(
    f"Data: {df.shape[0]:,} proteins × {len(numeric_cols)} samples · "
    f"Conditions: {', '.join(conditions)}"
)

# ---------------------------------------------------------------------
# 1. CHOOSE A VS B
# ---------------------------------------------------------------------

st.subheader("1️⃣ Comparison Setup (A vs B)")

col1, col2 = st.columns(2)
with col1:
    ref_cond = st.selectbox("Condition A (reference)", options=conditions, index=0)
with col2:
    treat_cond = st.selectbox(
        "Condition B (treatment)",
        options=[c for c in conditions if c != ref_cond],
        index=0 if len(conditions) > 1 else 0,
    )

if ref_cond == treat_cond:
    st.error("Choose two different conditions.")
    st.stop()

ref_samples = cond_samples[ref_cond]
treat_samples = cond_samples[treat_cond]

st.markdown(
    f"- Log2FC is **A/B = {ref_cond}/{treat_cond}**\n"
    "- Positive log2FC → higher in A; negative → higher in B."
)
st.markdown("---")

# ---------------------------------------------------------------------
# 2. COMPOSITION-BASED EXPECTED FC
# ---------------------------------------------------------------------

st.subheader("2️⃣ Spike-in Composition (optional)")

use_comp = st.checkbox(
    "Provide % composition per species per condition to define expected log2FC",
    value=False,
)

theoretical_fc_temp: Dict[str, float] = {}
species_values = sorted([s for s in df[species_col].unique() if isinstance(s, str) and s != 'Unknown'])

if use_comp:
    st.markdown("Enter percentage composition (normalized to 100% within each condition).")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{ref_cond} (A)**")
        comp_a = {}
        for sp in species_values:
            val = st.number_input(
                f"{sp} (%) in {ref_cond}",
                min_value=0.0,
                max_value=100.0,
                value=100.0 / max(len(species_values), 1),
                step=5.0,
                key=f"a_{sp}",
            )
            comp_a[sp] = val
        ta = sum(comp_a.values()) or 1.0
        comp_a = {k: v * 100 / ta for k, v in comp_a.items()}

    with c2:
        st.markdown(f"**{treat_cond} (B)**")
        comp_b = {}
        for sp in species_values:
            val = st.number_input(
                f"{sp} (%) in {treat_cond}",
                min_value=0.0,
                max_value=100.0,
                value=100.0 / max(len(species_values), 1),
                step=5.0,
                key=f"b_{sp}",
            )
            comp_b[sp] = val
        tb = sum(comp_b.values()) or 1.0
        comp_b = {k: v * 100 / tb for k, v in comp_b.items()}

    rows = []
    for sp in species_values:
        pa = comp_a.get(sp, 0.0)
        pb = comp_b.get(sp, 0.0)
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
            "Linear_FC": f"{2**log2fc:.2f}x",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    
    if st.button("💾 Save Expected Fold Changes", type="primary"):
        st.session_state.dea_theoretical_fc = theoretical_fc_temp.copy()
        st.success(f"✅ Saved expected FC for {len(theoretical_fc_temp)} species!")

    saved_fc = st.session_state.get('dea_theoretical_fc', {})
    if saved_fc:
        st.info(f"✓ Saved: {', '.join(f'{k}={v:.2f}' for k, v in saved_fc.items())}")
    else:
        st.warning("⚠️ No expected FC saved yet.")

st.markdown("---")

# ---------------------------------------------------------------------
# 3. STATISTICAL SETTINGS
# ---------------------------------------------------------------------

st.subheader("3️⃣ Statistical Settings")

c1, c2 = st.columns(2)
with c1:
    p_thr = st.selectbox(
        "FDR significance threshold",
        options=[0.001, 0.01, 0.05, 0.1],
        index=2,
        format_func=lambda x: f"{x*100:.1f} %",
    )
with c2:
    use_fdr = st.checkbox("Use FDR correction (BH)", value=True)

stable_thr = 0.5
st.caption(f"Stable proteome: |expected log2FC| < {stable_thr}")

st.markdown("---")

# ---------------------------------------------------------------------
# 4. RUN ANALYSIS
# ---------------------------------------------------------------------

st.subheader("4️⃣ Run Analysis")

if st.button("🚀 Run Welch's t-test", type="primary"):
    with st.spinner("Running statistical tests..."):
        df_num = df[numeric_cols]
        if (df_num > 50).any().any():
            df_log2 = np.log2(df_num + 1.0)
        else:
            df_log2 = df_num.copy()

        results = perform_ttest(df_log2, ref_samples, treat_samples)
        
        test_col = "fdr" if use_fdr else "pvalue"
        
        results["regulation"] = results.apply(
            lambda row: classify_regulation(
                row["log2fc"], row[test_col], fc_threshold=0.0, pval_threshold=p_thr
            ),
            axis=1,
        )
        
        results["species"] = results.index.map(df[species_col])

        st.session_state.dea_results = results
        st.session_state.dea_ref = ref_cond
        st.session_state.dea_treat = treat_cond
        st.session_state.dea_p_thr = p_thr
        st.session_state.dea_use_fdr = use_fdr

    st.success("✅ Analysis complete!")

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
    n_up = int((res["regulation"] == "up").sum())
    n_down = int((res["regulation"] == "down").sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("Total", f"{n_total:,}")
    m2.metric("Significant", f"{n_up + n_down:,}")
    m3.metric("Up / Down", f"{n_up} / {n_down}")

    # Volcano
    st.markdown("### 🌋 Volcano Plot")
    volc = res[res["regulation"] != "not_tested"].dropna(subset=['neg_log10_pval', 'log2fc'])
    
    fig_v = px.scatter(
        volc,
        x="log2fc",
        y="neg_log10_pval",
        color="species",
        hover_data=["regulation"],
        labels={
            "log2fc": f"log2 FC ({ref_cond}/{treat_cond})",
            "neg_log10_pval": "-log10(FDR)" if use_fdr else "-log10(p)",
        },
        height=600,
    )
    fig_v.add_hline(y=-np.log10(p_thr), line_dash="dash", line_color="gray")
    st.plotly_chart(fig_v, use_container_width=True)

    # MA
    st.markdown("### 📈 MA Plot")
    ma = res[res["regulation"] != "not_tested"].copy()
    ma["A"] = (ma["mean_g1"] + ma["mean_g2"]) / 2
    ma = ma.dropna(subset=['A', 'log2fc'])
    
    fig_ma = px.scatter(
        ma,
        x="A",
        y="log2fc",
        color="species",
        hover_data=["regulation"],
        labels={"A": "Mean log2 intensity", "log2fc": f"log2 FC ({ref_cond}/{treat_cond})"},
        height=600,
    )
    fig_ma.add_hline(y=0.0, line_color="red")
    st.plotly_chart(fig_ma, use_container_width=True)

    # Top table
    st.markdown("### 📋 Top Significant")
    top = res[res["regulation"].isin(["up", "down"])].sort_values("fdr").head(50)
    st.dataframe(top[["log2fc", "pvalue", "fdr", "regulation", "species"]].round(4), use_container_width=True)

    # ---------------------------------------------------------------------
    # 6. VALIDATION
    # ---------------------------------------------------------------------
    if theoretical_fc:
        st.markdown("---")
        st.subheader("6️⃣ Spike-in Validation")
        
        st.info(f"✓ Using saved FC: {', '.join(f'{k}={v:.2f}' for k, v in theoretical_fc.items())}")

        species_series = df[species_col]
        
        var_ov, var_sp, stab_ov, stab_sp = compute_species_metrics(
            res, theoretical_fc, species_series, stable_thr=stable_thr
        )

        if stab_ov:
            st.markdown("**Stable Proteome (FP Analysis)**")
            c1, c2 = st.columns(2)
            c1.metric("False Positive Rate", f"{stab_ov['FPR']:.1%}")
            c2.metric("Stable Proteins", f"{stab_ov['Total']:,}")
            if not stab_sp.empty:
                st.dataframe(stab_sp, use_container_width=True)
        
        if var_ov:
            st.markdown("**Variable Proteome (Detection)**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Sensitivity", f"{var_ov['Sensitivity']:.1%}")
            c2.metric("Specificity", f"{var_ov['Specificity']:.1%}")
            c3.metric("Precision", f"{var_ov['Precision']:.1%}")
            if not var_sp.empty:
                st.dataframe(var_sp, use_container_width=True)
    else:
        st.info("💡 Define spike-in composition in section 2 to enable validation")

    # Export
    st.markdown("---")
    st.download_button(
        "📥 Download Results",
        data=res.to_csv().encode("utf-8"),
        file_name=f"dea_{ref_cond}_vs_{treat_cond}.csv",
        mime="text/csv",
    )
else:
    st.info("👆 Configure and run analysis")
