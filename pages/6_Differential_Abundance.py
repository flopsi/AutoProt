"""
pages/6_Differential_Abundance.py
Welch's t-test DA with comprehensive visualization and spike-in validation.
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
) -> Tuple[Dict, pd.DataFrame, Dict, pd.DataFrame, Dict]:
    """
    Calculate metrics for variable and stable proteomes + asymmetry.
    
    Returns:
        (variable_overall, variable_per_species, stable_overall, stable_per_species, asymmetry_dict)
    """
    res = results_df.copy()
    res["species"] = res.index.map(species_col_series)
    res["true_log2fc"] = res["species"].map(true_fc_dict)
    
    res = res[res["regulation"] != "not_tested"].copy()
    res = res.dropna(subset=["true_log2fc", "species"])
    
    if res.empty:
        return {}, pd.DataFrame(), {}, pd.DataFrame(), {}
    
    # === ASYMMETRY CALCULATION ===
    asymmetry_dict = {}
    for sp in res["species"].unique():
        sp_df = res[res["species"] == sp].copy()
        expected_fc = true_fc_dict.get(sp, 0.0)
        if abs(expected_fc) >= stable_thr:
            asym = calculate_asymmetry(sp_df["log2fc"].values, expected_fc)
            asymmetry_dict[sp] = asym
    
    # === VARIABLE PROTEOME ===
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
        stab_df["observed_regulated"] = stab_df["regulation"].isin(["up", "down"])
        fp = int(stab_df["observed_regulated"].sum())
        tn = int((~stab_df["observed_regulated"]).sum())
        total = len(stab_df)
        fpr = fp / total if total > 0 else 0.0
        
        stab_overall = {"Total": total, "FP": fp, "TN": tn, "FPR": fpr}
        
        for sp in stab_df["species"].unique():
            sp_df = stab_df[stab_df["species"] == sp].copy()
            fp_s = int(sp_df["observed_regulated"].sum())
            tn_s = int((~sp_df["observed_regulated"]).sum())
            mae_log2 = sp_df["log2fc"].abs().mean()
            
            stab_species_rows.append({
                "Species": sp,
                "N": len(sp_df),
                "FP": fp_s,
                "TN": tn_s,
                "FPR_%": f"{fp_s/len(sp_df)*100:.1f}" if len(sp_df) > 0 else "0.0",
                "MAE": f"{mae_log2:.3f}",
            })
    
    return (
        var_overall,
        pd.DataFrame(var_species_rows),
        stab_overall,
        pd.DataFrame(stab_species_rows),
        asymmetry_dict,
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
st.caption(f"Stable: |expected log2FC| < {stable_thr} · Errors on log2 scale")
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
    
    # === PLOT 1: FACETED SCATTER (MA PLOT) ===
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
        
        # Calculate mean and std for outlier detection
        mean_fc = sp_data["log2fc"].mean()
        std_fc = sp_data["log2fc"].std()
        
        # Filter outliers: only show points within ±2 SD
        outlier_mask = np.abs(sp_data["log2fc"] - mean_fc) <= 2 * std_fc
        sp_data_filtered = sp_data[outlier_mask]
        
        fig_facet.add_trace(
            go.Scatter(
                x=sp_data_filtered["A"],
                y=sp_data_filtered["log2fc"],
                mode='markers',
                marker=dict(size=3, color=color, opacity=0.6),
                name=sp,
                showlegend=False,
                hovertemplate=f"{sp}<br>A=%{{x:.2f}}<br>log2FC=%{{y:.3f}}<extra></extra>"
            ),
            row=1, col=i
        )
        
        # Add expected FC line if available
        if theoretical_fc and sp in theoretical_fc:
            expected_fc = theoretical_fc[sp]
            if abs(expected_fc) >= stable_thr:
                fig_facet.add_hline(
                    y=expected_fc,
                    line_dash="dash",
                    line_color=color,
                    line_width=2,
                    row=1, col=i
                )
        
        # Zero line
        fig_facet.add_hline(y=0, line_color="red", line_width=1, opacity=0.5, row=1, col=i)
        
        # Add boxplot showing only ±2 SD range
        fig_facet.add_trace(
            go.Box(
                y=sp_data_filtered["log2fc"],
                name=sp,
                marker_color=color,
                showlegend=False,
                width=0.3,
                boxmean='sd',
                boxpoints=False  # Don't show individual outlier points
            ),
            row=1, col=i
        )
    
    fig_facet.update_xaxes(title_text="log2(B)", row=1, col=2)
    fig_facet.update_yaxes(title_text=f"log2(A:B)", row=1, col=1)
    fig_facet.update_layout(height=500, title_text="", showlegend=False)
    st.plotly_chart(fig_facet, use_container_width=True)
    
    # === PLOT 2: REGULAR SCATTER (MA PLOT) ===
    st.markdown("### 📈 MA Plot (Combined)")
    
    fig_ma = px.scatter(
        ma,
        x="A",
        y="log2fc",
        color="species",
        color_discrete_map=SPECIES_COLORS,
        hover_data=["regulation"],
        labels={"A": "log2(B)", "log2fc": f"log2({ref_cond}/{treat_cond})"},
        height=600,
    )
    
    fig_ma.add_hline(y=0.0, line_color="red", line_width=1, opacity=0.5)
    
    if theoretical_fc:
        for species, expected_fc in theoretical_fc.items():
            if abs(expected_fc) >= stable_thr:
                fig_ma.add_hline(
                    y=expected_fc,
                    line_dash="dot",
                    line_width=2,
                    line_color=SPECIES_COLORS.get(species, "#95a5a6"),
                    opacity=0.7,
                    annotation_text=f"{species} (exp={expected_fc:.2f})",
                    annotation_position="right",
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
    
    # Volcano
    st.markdown("### 🌋 Volcano Plot")
    volc = res[res["regulation"] != "not_tested"].dropna(subset=['neg_log10_p', 'log2fc'])
    
    fig_v = px.scatter(
        volc,
        x="log2fc",
        y="neg_log10_p",
        color="species",
        color_discrete_map=SPECIES_COLORS,
        hover_data=["regulation"],
        labels={"log2fc": f"log2 FC ({ref_cond}/{treat_cond})", "neg_log10_p": "-log10(FDR)" if use_fdr else "-log10(p)"},
        height=600,
    )
    fig_v.add_hline(y=-np.log10(p_thr), line_dash="dash", line_color="gray")
    st.plotly_chart(fig_v, use_container_width=True)
    
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
        
        st.info(f"✓ Using: {', '.join(f'{k}={v:.2f}' for k,v in theoretical_fc.items())}")
        
        species_series = df[species_col]
        var_ov, var_sp, stab_ov, stab_sp, asym_dict = compute_species_metrics(res, theoretical_fc, species_series, stable_thr)
        
        # Display asymmetry
        if asym_dict:
            st.markdown("**Asymmetry (median/expected)**")
            asym_cols = st.columns(len(asym_dict))
            for i, (sp, asym) in enumerate(asym_dict.items()):
                asym_cols[i].metric(sp, f"{asym:.2f}")
        
        if stab_ov:
            st.markdown("**Stable Proteome (FP)**")
            c1, c2 = st.columns(2)
            c1.metric("FPR", f"{stab_ov['FPR']:.1%}")
            c2.metric("N", f"{stab_ov['Total']:,}")
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
                st.caption("**RMSE/MAE**: log2 units | **MAPE**: % error vs expected | **Bias**: systematic over/under-estimation")
    else:
        st.info("💡 Enable spike-in composition for validation")
    
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
