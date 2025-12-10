"""pages/6_Differential_Abundance.py - CORRECTED & SIMPLIFIED
Comprehensive Differential Abundance Analysis (DEA)
- Welch's t-test with proper log2 fold-change calculation
- FDR correction (Benjamini-Hochberg)
- ROC and precision-recall curves
- Per-species error metrics
- Spike-in validation
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys
from scipy.stats import ttest_ind, mannwhitneyu
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(str(Path(__file__).parent.parent))

# ============================================================================
# T-TEST & FDR CALCULATION
# ============================================================================

def perform_ttest(
    df: pd.DataFrame,
    group1_cols: list,
    group2_cols: list,
    min_valid: int = 2,
) -> pd.DataFrame:
    """
    Perform Welch's t-test on intensity data.
    
    Log2FC = log2(mean(group1) / mean(group2))
    Positive FC = higher in group1
    
    Args:
        df: Raw intensity data (NOT log-transformed)
        group1_cols: Column names for reference/control
        group2_cols: Column names for treatment
        min_valid: Minimum non-missing values required per group
        
    Returns:
        DataFrame with log2fc, pvalue, fdr, means, n_valid
    """
    results = []
    
    # Loop over each protein
    for protein_id in df.index:
        row = df.loc[protein_id]
        g1_vals = row[group1_cols].dropna().values
        g2_vals = row[group2_cols].dropna().values
        
        # Check minimum requirement
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
        
        # Calculate means in LINEAR space
        mean_g1 = np.mean(g1_vals)
        mean_g2 = np.mean(g2_vals)
        
        # Prevent divide by zero
        if mean_g2 == 0:
            log2fc = np.nan
        else:
            # CORRECT: log2(mean1/mean2) - fold change in linear space
            log2fc = np.log2(mean_g1 / mean_g2)
        
        # Perform Welch's t-test (unequal variances)
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
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df.set_index("protein_id", inplace=True)
    
    # FDR Correction (Benjamini-Hochberg)
    pvals = results_df["pvalue"].dropna()
    
    if len(pvals) > 0:
        sorted_pvals = pvals.sort_values()
        n = len(sorted_pvals)
        ranks = np.arange(1, n + 1)
        
        # BH correction
        fdr_vals = sorted_pvals.values * n / ranks
        
        # Ensure monotonicity (increasing from right to left)
        fdr_vals = np.minimum.accumulate(fdr_vals[::-1])[::-1]
        
        # Cap at 1.0
        fdr_vals = np.minimum(fdr_vals, 1.0)
        
        # Map back
        fdr_dict = dict(zip(sorted_pvals.index, fdr_vals))
        results_df["fdr"] = results_df.index.map(fdr_dict)
    else:
        results_df["fdr"] = np.nan
    
    # Add -log10(p) for volcano plots
    results_df["neg_log10_pval"] = -np.log10(
        results_df["pvalue"].replace(0, 1e-300)
    )
    
    return results_df


# ============================================================================
# CLASSIFICATION
# ============================================================================

def classify_regulation(
    log2fc: float,
    pvalue: float,
    fc_threshold: float = 1.0,
    pval_threshold: float = 0.05,
) -> str:
    """
    Classify protein regulation status.
    """
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


# ============================================================================
# ERROR METRICS
# ============================================================================

def calculate_error_metrics(
    results_df: pd.DataFrame,
    true_fc_dict: Dict[str, float],
    fc_threshold: float = 1.0,
    pval_threshold: float = 0.05,
) -> Dict:
    """
    Calculate confusion matrix and performance metrics.
    """
    results_df = results_df.copy()
    
    # Map true fold changes
    results_df["true_log2fc"] = results_df.index.map(
        lambda x: true_fc_dict.get(x, 0.0)
    )
    
    # Classify true regulation
    results_df["true_regulated"] = results_df["true_log2fc"].apply(
        lambda x: abs(x) > fc_threshold
    )
    
    # Observed regulation
    results_df["observed_regulated"] = results_df["regulation"].isin(["up", "down"])
    
    # Confusion matrix
    TP = ((results_df["true_regulated"]) & (results_df["observed_regulated"])).sum()
    FP = ((~results_df["true_regulated"]) & (results_df["observed_regulated"])).sum()
    TN = ((~results_df["true_regulated"]) & (~results_df["observed_regulated"])).sum()
    FN = ((results_df["true_regulated"]) & (~results_df["observed_regulated"])).sum()
    
    # Metrics
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    return {
        "TP": int(TP),
        "FP": int(FP),
        "TN": int(TN),
        "FN": int(FN),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "fpr": fpr,
    }


# ============================================================================
# PER-SPECIES METRICS
# ============================================================================

def compute_species_rmse(
    results_df: pd.DataFrame,
    true_fc_dict: Dict[str, float],
    species_col: pd.Series,
    fc_threshold: float = 1.0,
    pval_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Calculate RMSE and other metrics per species.
    """
    species_metrics = []
    
    results_df = results_df.copy()
    results_df["species"] = results_df.index.map(species_col)
    
    for species in results_df["species"].unique():
        if pd.isna(species):
            continue
        
        # Get results for this species
        species_results = results_df[results_df["species"] == species].copy()
        
        if len(species_results) == 0:
            continue
        
        # Get theoretical FC for this species
        theo_fc = true_fc_dict.get(species, 0.0)
        
        # Calculate error
        species_results["error"] = species_results["log2fc"] - theo_fc
        
        n_proteins = len(species_results)
        rmse = np.sqrt((species_results["error"] ** 2).mean())
        mae = species_results["error"].abs().mean()
        bias = species_results["error"].mean()
        
        # Detection rate
        if "regulation" not in species_results.columns:
            species_results["regulation"] = species_results.apply(
                lambda row: classify_regulation(
                    row["log2fc"], row["pvalue"], fc_threshold, pval_threshold
                ),
                axis=1,
            )
        
        n_detected = (
            species_results["regulation"].isin(["up", "down"]).sum()
        )
        detection_rate = n_detected / n_proteins if n_proteins > 0 else 0
        
        species_metrics.append({
            "Species": species,
            "N": n_proteins,
            "Theo FC": f"{theo_fc:.2f}",
            "RMSE": f"{rmse:.3f}",
            "MAE": f"{mae:.3f}",
            "Detection": f"{detection_rate:.1%}",
            "Bias": f"{bias:.3f}",
        })
    
    return pd.DataFrame(species_metrics)


# ============================================================================
# ROC CURVE
# ============================================================================

def compute_roc_curve(
    results_df: pd.DataFrame,
    true_fc_dict: Dict[str, float],
    fc_threshold: float = 1.0,
) -> Tuple[list, list, list]:
    """
    Compute ROC curve by varying p-value threshold.
    """
    results_df = results_df.copy()
    results_df["true_regulated"] = results_df.index.map(
        lambda x: abs(true_fc_dict.get(x, 0.0)) > fc_threshold
    )
    
    results_df = results_df.dropna(subset=["pvalue"])
    
    fpr_list = []
    tpr_list = []
    
    n_neg = (~results_df["true_regulated"]).sum()
    n_pos = results_df["true_regulated"].sum()
    
    if n_neg == 0 or n_pos == 0:
        return [0, 1], [0, 1], [1, 0]
    
    # Vary threshold
    for pval_threshold in np.linspace(1, 0, 50):
        tp = (results_df["true_regulated"] & (results_df["pvalue"] < pval_threshold)).sum()
        fp = ((~results_df["true_regulated"]) & (results_df["pvalue"] < pval_threshold)).sum()
        
        tpr = tp / n_pos
        fpr = fp / n_neg
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    return fpr_list, tpr_list, np.linspace(1, 0, 50)


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(page_title="Differential Abundance", page_icon="🔬", layout="wide")
st.title("🔬 Differential Abundance Analysis (DEA)")
st.markdown("Welch's t-test + FDR correction + Spike-in validation")
st.markdown("---")

# ============================================================================
# DATA VALIDATION
# ============================================================================

if "df_imputed" not in st.session_state or st.session_state.df_imputed is None:
    st.error("❌ No imputed data. Complete **📊 Post-Imputation EDA** first.")
    st.stop()

df = st.session_state.df_imputed.copy()
numeric_cols = st.session_state.get("numeric_cols", [])
species_col_name = st.session_state.get("species_col", None)
sample_to_condition = st.session_state.get("sample_to_condition", {})

if not numeric_cols or not species_col_name or not sample_to_condition:
    st.error("❌ Missing required session state. Complete **📊 Post-Imputation EDA** first.")
    st.stop()

# Get species column
if species_col_name in df.columns:
    species_col = df[species_col_name]
else:
    st.error(f"❌ Species column '{species_col_name}' not found in data.")
    st.stop()

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

# Statistical thresholds
fc_thr = st.sidebar.slider("Log2 FC threshold (|x|)", 0.0, 3.0, 1.0, 0.1)
p_thr = st.sidebar.slider("P-value threshold", 0.001, 0.1, 0.05, 0.001)
use_fdr = st.sidebar.checkbox("Use FDR-adjusted p-values", value=True)

st.sidebar.markdown("---")

# Data filtering
min_intensity = st.sidebar.slider("Min mean intensity", 0.0, 20.0, 0.0, 1.0)
min_valid_per_group = st.sidebar.slider("Min values per group", 1, 5, 2, 1)

st.sidebar.markdown("---")

# Visualization options
viz_options = st.sidebar.multiselect(
    "Visualizations:",
    [
        "Volcano Plot",
        "MA Plot",
        "Box Plots (Top)",
        "Heatmap",
        "P-value Distribution",
        "ROC Curve",
    ],
    default=["Volcano Plot", "MA Plot"]
)

st.sidebar.markdown("---")

# Spike-in validation
use_spike_in = st.sidebar.checkbox("Enable Spike-in Validation", value=False)

if use_spike_in:
    st.sidebar.markdown("**Spike-in Fold Changes (log2)**")
    species_values = sorted([s for s in df[species_col_name].unique() if isinstance(s, str) and s.strip()])
    
    theoretical_fc: Dict[str, float] = {}
    for sp in species_values:
        fc_input = st.sidebar.number_input(f"{sp} (log2 FC):", value=0.0, step=0.1, key=f"fc_{sp}")
        theoretical_fc[sp] = fc_input
    
    if st.sidebar.button("💾 Save Spike-in"):
        st.session_state.dea_theoretical_fc = theoretical_fc.copy()
        st.sidebar.success("✅ Saved!")
else:
    theoretical_fc = st.session_state.get("dea_theoretical_fc", {})

st.sidebar.markdown("---")

# ============================================================================
# RUN ANALYSIS
# ============================================================================

st.subheader("1️⃣ Running Analysis")

if st.button("🚀 Run DEA", type="primary"):
    with st.spinner("⏳ Processing..."):
        # Prepare data
        df_work = df[numeric_cols].copy()
        
        # Filter by intensity
        mean_intensity = df_work[ref_samples + treat_samples].mean(axis=1)
        mask_intensity = mean_intensity >= min_intensity
        
        df_filtered = df_work[mask_intensity].copy()
        
        st.write(f"✓ After intensity filter: {df_filtered.shape[0]} proteins")
        
        # Run t-test
        results = perform_ttest(
            df_filtered,
            ref_samples,
            treat_samples,
            min_valid=min_valid_per_group
        )
        
        # Determine p-value column
        pval_col = "fdr" if use_fdr else "pvalue"
        
        # Classify
        results["regulation"] = results.apply(
            lambda row: classify_regulation(
                row["log2fc"],
                row[pval_col],
                fc_thr,
                p_thr
            ),
            axis=1
        )
        
        # Add species
        results["species"] = results.index.map(species_col)
        
        st.session_state.dea_results = results
        st.session_state.dea_ref_cond = ref_cond
        st.session_state.dea_treat_cond = treat_cond
        st.session_state.dea_fc_thr = fc_thr
        st.session_state.dea_p_thr = p_thr
        st.session_state.dea_pval_col = pval_col
        
    st.success("✅ Analysis complete!")

# ============================================================================
# RESULTS
# ============================================================================

if "dea_results" in st.session_state:
    res = st.session_state.dea_results
    ref_cond = st.session_state.dea_ref_cond
    treat_cond = st.session_state.dea_treat_cond
    fc_thr = st.session_state.dea_fc_thr
    p_thr = st.session_state.dea_p_thr
    pval_col = st.session_state.dea_pval_col
    theoretical_fc = st.session_state.get("dea_theoretical_fc", {})
    
    st.markdown("---")
    st.subheader("2️⃣ Results Summary")
    
    n_total = len(res)
    n_tested = int((res["regulation"] != "not_tested").sum())
    n_up = int((res["regulation"] == "up").sum())
    n_down = int((res["regulation"] == "down").sum())
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total", f"{n_total:,}")
    with col2:
        st.metric("Tested", f"{n_tested:,}")
    with col3:
        st.metric("↑ Up", f"{n_up:,}")
    with col4:
        st.metric("↓ Down", f"{n_down:,}")
    with col5:
        st.metric("Sig.", f"{n_up + n_down:,}")
    
    # Per-species metrics
    if len(res) > 0:
        st.markdown("---")
        st.subheader("3️⃣ Per-Species Metrics")
        
        species_metrics_df = compute_species_rmse(
            res,
            theoretical_fc,
            species_col,
            fc_thr,
            p_thr
        )
        
        if len(species_metrics_df) > 0:
            st.dataframe(species_metrics_df, use_container_width=True, hide_index=True)
    
    # Visualizations
    st.markdown("---")
    st.subheader("4️⃣ Visualizations")
    
    # Volcano Plot
    if "Volcano Plot" in viz_options:
        st.markdown("### 🌋 Volcano Plot")
        
        volc = res[res["regulation"] != "not_tested"].dropna(subset=["neg_log10_pval", "log2fc"])
        
        if len(volc) > 0:
            fig = px.scatter(
                volc,
                x="log2fc",
                y="neg_log10_pval",
                color="regulation",
                color_discrete_map={"up": "#e74c3c", "down": "#3498db", "not_significant": "#bdc3c7"},
                hover_data=["species"],
                labels={"log2fc": f"log2({ref_cond}/{treat_cond})", "neg_log10_pval": "-log10(p)"},
                height=600,
            )
            
            fig.add_hline(y=-np.log10(p_thr), line_dash="dash", line_color="gray")
            fig.add_vline(x=fc_thr, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=-fc_thr, line_dash="dash", line_color="gray", opacity=0.5)
            
            st.plotly_chart(fig, width="stretch")
    
    # MA Plot
    if "MA Plot" in viz_options:
        st.markdown("### 📊 MA Plot")
        
        ma = res[res["regulation"] != "not_tested"].dropna(subset=["log2fc"])
        
        if len(ma) > 0:
            ma["A"] = (np.log2(ma["mean_g1"] + 1) + np.log2(ma["mean_g2"] + 1)) / 2
            
            fig = px.scatter(
                ma,
                x="A",
                y="log2fc",
                color="regulation",
                color_discrete_map={"up": "#e74c3c", "down": "#3498db", "not_significant": "#bdc3c7"},
                hover_data=["species"],
                height=600,
            )
            
            fig.add_hline(y=0, line_color="red", opacity=0.3)
            fig.add_hline(y=fc_thr, line_dash="dash", line_color="green", opacity=0.3)
            fig.add_hline(y=-fc_thr, line_dash="dash", line_color="green", opacity=0.3)
            
            st.plotly_chart(fig, width="stretch")
    
    # Top Box Plots
    if "Box Plots (Top)" in viz_options:
        st.markdown("### 📦 Top Differentially Abundant Proteins")
        
        res_sorted = res.dropna(subset=[pval_col]).sort_values(pval_col)
        n_top = min(6, len(res_sorted))
        
        if n_top > 0:
            top_proteins = res_sorted.head(n_top).index
            
            fig = make_subplots(rows=2, cols=3, subplot_titles=top_proteins)
            
            for idx, protein in enumerate(top_proteins):
                row = idx // 3 + 1
                col = idx % 3 + 1
                
                g1 = df.loc[protein, ref_samples]
                g2 = df.loc[protein, treat_samples]
                
                fig.add_trace(
                    go.Box(y=g1, name=ref_cond, showlegend=(idx==0)),
                    row=row, col=col
                )
                fig.add_trace(
                    go.Box(y=g2, name=treat_cond, showlegend=(idx==0)),
                    row=row, col=col
                )
            
            fig.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig, width="stretch")
    
    # Heatmap
    if "Heatmap" in viz_options:
        st.markdown("### 🔥 Heatmap (Top 15 Proteins)")
        
        res_sorted = res.dropna(subset=[pval_col]).sort_values(pval_col)
        n_top = min(15, len(res_sorted))
        
        if n_top > 0:
            top_proteins = res_sorted.head(n_top).index
            hm_data = df.loc[top_proteins, ref_samples + treat_samples]
            
            # Z-score normalize
            hm_z = (hm_data.T - hm_data.T.mean()) / (hm_data.T.std() + 1e-6)
            
            fig = go.Figure(data=go.Heatmap(
                z=hm_z.T.values,
                x=hm_z.T.columns,
                y=hm_z.T.index,
                colorscale="RdBu_r",
                zmid=0,
            ))
            fig.update_layout(height=600, width=1000)
            st.plotly_chart(fig, width="stretch")
    
    # P-value distribution
    if "P-value Distribution" in viz_options:
        st.markdown("### 📈 P-value Distribution")
        
        pvals = res[pval_col].dropna()
        
        fig = px.histogram(
            {"p": pvals},
            x="p",
            nbins=50,
            height=500,
            labels={"p": pval_col}
        )
        fig.add_vline(x=p_thr, line_dash="dash", line_color="red")
        st.plotly_chart(fig, width="stretch")
    
    # ROC Curve
    if "ROC Curve" in viz_options and theoretical_fc:
        st.markdown("### 📉 ROC Curve (Spike-in)")
        
        fpr_list, tpr_list, _ = compute_roc_curve(res, theoretical_fc, fc_thr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr_list, y=tpr_list, mode="lines", name="ROC"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
        
        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=600,
        )
        st.plotly_chart(fig, width="stretch")
    
    # Results table
    st.markdown("---")
    st.subheader("5️⃣ Detailed Results")
    
    with st.expander("📋 View All Results"):
        display_cols = ["log2fc", "pvalue", "fdr", "mean_g1", "mean_g2", "n_g1", "n_g2", "species", "regulation"]
        display_cols = [c for c in display_cols if c in res.columns]
        
        display_df = res[display_cols].copy()
        display_df = display_df.round(6)
        display_df = display_df.sort_values(pval_col)
        
        st.dataframe(display_df, height=600, use_container_width=True)
        
        csv = display_df.to_csv(index=True)
        st.download_button("📥 Download CSV", csv, f"dea_{ref_cond}_vs_{treat_cond}.csv", "text/csv")

else:
    st.info("👆 Configure and run analysis above")
