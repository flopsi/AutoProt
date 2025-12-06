# pages/2_Visual_EDA.py

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from helpers.transforms import apply_transformation, TRANSFORM_NAMES

st.set_page_config(page_title="Visual EDA", layout="wide")

st.title("📊 Transformation Evaluation by Condition")

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
protein_data = st.session_state.get("protein_data")
if protein_data is None:
    st.error("No data found. Please upload data first on the 'Data Upload' page.")
    st.stop()

df_raw: pd.DataFrame = protein_data.raw
numeric_cols = protein_data.numeric_cols

if not numeric_cols:
    st.error("No numeric intensity columns found.")
    st.stop()

st.success(f"{len(df_raw):,} proteins × {len(numeric_cols)} intensity columns")

# ----------------------------------------------------------------------
# 1) Configuration: Split columns into 2 conditions
# ----------------------------------------------------------------------
st.subheader("1️⃣ Configuration")

n_cols = len(numeric_cols)
mid = n_cols // 2

col1, col2 = st.columns(2)
with col1:
    cond_a_cols = st.multiselect(
        "Condition A columns",
        options=numeric_cols,
        default=numeric_cols[:mid],
    )
with col2:
    cond_b_cols = st.multiselect(
        "Condition B columns",
        options=numeric_cols,
        default=numeric_cols[mid:],
    )

if not cond_a_cols or not cond_b_cols:
    st.warning("Select at least one column for each condition.")
    st.stop()

# ----------------------------------------------------------------------
# Helper: sum intensities per protein per condition
# ----------------------------------------------------------------------
def sum_per_condition(df: pd.DataFrame, cols: list) -> pd.Series:
    """Sum intensity across specified columns for each protein."""
    return df[cols].sum(axis=1)

# ----------------------------------------------------------------------
# Helper: pooled normality stats
# ----------------------------------------------------------------------
def pooled_normality_stats(series: pd.Series) -> dict:
    x = series.dropna().values
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return {"n": n, "W": np.nan, "p": np.nan, "skew": np.nan, "kurt": np.nan}
    W, p = stats.shapiro(x)
    skew = stats.skew(x, bias=False)
    kurt = stats.kurtosis(x, fisher=True, bias=False)
    return {
        "n": int(n),
        "W": float(W),
        "p": float(p),
        "skew": float(skew),
        "kurt": float(kurt),
    }

# ----------------------------------------------------------------------
# Helper: create density histogram
# ----------------------------------------------------------------------
def create_density_histogram(series: pd.Series, title: str, color: str) -> go.Figure:
    x = series.dropna().values
    x = x[np.isfinite(x)]
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=x,
        histnorm='probability density',
        nbinsx=50,
        marker_color=color,
        opacity=0.7,
        showlegend=False,
    ))
    
    if len(x) >= 3:
        mu = x.mean()
        sigma = x.std()
        fig.add_vline(x=mu, line_dash="dash", line_color="red", line_width=2,
                     annotation_text=f"μ={mu:.2f}")
        fig.add_vrect(x0=mu-2*sigma, x1=mu+2*sigma, 
                     fillcolor=color, opacity=0.15, line_width=0)
    
    fig.update_layout(
        title=title,
        xaxis_title="Summed Intensity",
        yaxis_title="Density",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig

# ----------------------------------------------------------------------
# 2) Evaluate all transformations
# ----------------------------------------------------------------------
st.subheader("2️⃣ Transformation Comparison")

available_methods = ["raw"] + [m for m in TRANSFORM_NAMES.keys() if m != "raw"]

results = []

for method in available_methods:
    method_name = TRANSFORM_NAMES.get(method, method) if method != "raw" else "Raw"
    
    # Apply transformation
    if method == "raw":
        df_trans = df_raw.copy()
        trans_cond_a_cols = cond_a_cols
        trans_cond_b_cols = cond_b_cols
    else:
        df_trans, trans_cols = apply_transformation(df_raw, cond_a_cols + cond_b_cols, method)
        # Map original cols to transformed cols
        trans_cond_a_cols = [f"{c}_transformed" for c in cond_a_cols]
        trans_cond_b_cols = [f"{c}_transformed" for c in cond_b_cols]
    
    # Sum per condition per protein
    sum_a = sum_per_condition(df_trans, trans_cond_a_cols)
    sum_b = sum_per_condition(df_trans, trans_cond_b_cols)
    
    # Stats
    stats_a = pooled_normality_stats(sum_a)
    stats_b = pooled_normality_stats(sum_b)
    
    # Combined score: higher W and p is better
    combined_W = (stats_a["W"] + stats_b["W"]) / 2
    combined_p = (stats_a["p"] + stats_b["p"]) / 2
    
    results.append({
        "Method": method_name,
        "key": method,
        "W_A": stats_a["W"],
        "p_A": stats_a["p"],
        "W_B": stats_b["W"],
        "p_B": stats_b["p"],
        "W_combined": combined_W,
        "p_combined": combined_p,
        "sum_a": sum_a,
        "sum_b": sum_b,
    })
    
    # Plot: 2 columns (Cond A | Cond B)
    st.markdown(f"### {method_name}")
    
    col_plot_a, col_plot_b = st.columns(2)
    
    with col_plot_a:
        fig_a = create_density_histogram(sum_a, f"Condition A", "#1f77b4")
        st.plotly_chart(fig_a, use_container_width=True, key=f"plot_{method}_a")
        st.caption(f"W={stats_a['W']:.4f}, p={stats_a['p']:.2e}")
    
    with col_plot_b:
        fig_b = create_density_histogram(sum_b, f"Condition B", "#ff7f0e")
        st.plotly_chart(fig_b, use_container_width=True, key=f"plot_{method}_b")
        st.caption(f"W={stats_b['W']:.4f}, p={stats_b['p']:.2e}")

# ----------------------------------------------------------------------
# 3) Summary table with ranking
# ----------------------------------------------------------------------
st.subheader("3️⃣ Transformation Ranking")

summary_df = pd.DataFrame([
    {
        "Method": r["Method"],
        "W (Cond A)": r["W_A"],
        "p (Cond A)": r["p_A"],
        "W (Cond B)": r["W_B"],
        "p (Cond B)": r["p_B"],
        "W (Avg)": r["W_combined"],
        "p (Avg)": r["p_combined"],
    }
    for r in results
])

# Ranking: higher W_combined and p_combined is better
summary_df["Score"] = (
    summary_df["W (Avg)"].rank(ascending=False) +
    summary_df["p (Avg)"].rank(ascending=False)
)

summary_df = summary_df.sort_values("Score").reset_index(drop=True)
summary_df.index = summary_df.index + 1

st.dataframe(summary_df.round(4), use_container_width=True)

# Best method
best = summary_df.iloc[0]
st.success(
    f"🏆 Best transformation: **{best['Method']}** "
    f"(W_avg={best['W (Avg)']:.3f}, p_avg={best['p (Avg)']:.2e})"
)

# Store selection
best_key = [r["key"] for r in results if r["Method"] == best["Method"]][0]
st.session_state.selected_transform_method = best_key
st.info(f"📌 Recommended for downstream analysis: **{best['Method']}**")
