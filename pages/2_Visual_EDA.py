# pages/2_Visual_EDA.py

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go

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
# Helper: create density histogram with 3σ
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
        # 3 sigma region
        fig.add_vrect(x0=mu-3*sigma, x1=mu+3*sigma, 
                     fillcolor=color, opacity=0.15, line_width=0)
        fig.add_annotation(
            x=mu+3*sigma, y=0,
            text="±3σ",
            showarrow=False,
            font=dict(color=color, size=10),
        )
    
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
        trans_cond_a_cols = [f"{c}_transformed" for c in cond_a_cols]
        trans_cond_b_cols = [f"{c}_transformed" for c in cond_b_cols]
    
    # Sum per condition per protein
    sum_a = sum_per_condition(df_trans, trans_cond_a_cols)
    sum_b = sum_per_condition(df_trans, trans_cond_b_cols)
    
    # Stats
    stats_a = pooled_normality_stats(sum_a)
    stats_b = pooled_normality_stats(sum_b)
    
    # Combined score
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
        "df_trans": df_trans,
        "trans_cond_a_cols": trans_cond_a_cols,
        "trans_cond_b_cols": trans_cond_b_cols,
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
        "key": r["key"],
        "W (Cond A)": r["W_A"],
        "p (Cond A)": r["p_A"],
        "W (Cond B)": r["W_B"],
        "p (Cond B)": r["p_B"],
        "W (Avg)": r["W_combined"],
        "p (Avg)": r["p_combined"],
    }
    for r in results
])

summary_df["Score"] = (
    summary_df["W (Avg)"].rank(ascending=False) +
    summary_df["p (Avg)"].rank(ascending=False)
)

summary_df = summary_df.sort_values("Score").reset_index(drop=True)
summary_df.index = summary_df.index + 1

st.dataframe(summary_df[["Method", "W (Cond A)", "p (Cond A)", "W (Cond B)", "p (Cond B)", "W (Avg)", "p (Avg)", "Score"]].round(4), use_container_width=True)

best = summary_df.iloc[0]
st.success(
    f"🏆 Best transformation: **{best['Method']}** "
    f"(W_avg={best['W (Avg)']:.3f}, p_avg={best['p (Avg)']:.2e})"
)

# ----------------------------------------------------------------------
# 4) Store data options
# ----------------------------------------------------------------------
st.subheader("4️⃣ Store Data for Analysis")

# Filter by ±3σ option
filter_3sigma = st.checkbox(
    "Use only data within ±3σ (remove outliers)",
    value=False,
    help="Filters proteins where summed intensity per condition is within μ±3σ"
)

col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("💾 Store Raw Only", type="primary", use_container_width=True):
        df_to_store = df_raw.copy()
        
        if filter_3sigma:
            # Filter raw by 3sigma on summed intensities
            sum_a_raw = sum_per_condition(df_raw, cond_a_cols)
            sum_b_raw = sum_per_condition(df_raw, cond_b_cols)
            
            mu_a, sig_a = sum_a_raw.mean(), sum_a_raw.std()
            mu_b, sig_b = sum_b_raw.mean(), sum_b_raw.std()
            
            mask_a = (sum_a_raw >= mu_a - 3*sig_a) & (sum_a_raw <= mu_a + 3*sig_a)
            mask_b = (sum_b_raw >= mu_b - 3*sig_b) & (sum_b_raw <= mu_b + 3*sig_b)
            
            df_to_store = df_to_store[mask_a & mask_b]
            st.info(f"Filtered: {len(df_to_store):,} / {len(df_raw):,} proteins (within ±3σ)")
        
        protein_data.raw = df_to_store
        st.session_state.protein_data = protein_data
        st.session_state.selected_transform_method = "raw"
        st.success("✅ Stored: Raw data only")

with col_btn2:
    st.markdown("**Select transformation:**")
    transform_options = [r["key"] for r in results if r["key"] != "raw"]
    selected_transform = st.radio(
        "Transform",
        options=transform_options,
        format_func=lambda k: TRANSFORM_NAMES.get(k, k),
        label_visibility="collapsed",
        horizontal=True,
    )
    
    if st.button("💾 Store Raw + Transformed", type="secondary", use_container_width=True):
        # Get selected transformation result
        selected_result = [r for r in results if r["key"] == selected_transform][0]
        df_trans = selected_result["df_trans"]
        trans_a = selected_result["trans_cond_a_cols"]
        trans_b = selected_result["trans_cond_b_cols"]
        
        if filter_3sigma:
            # Filter by 3sigma on transformed summed intensities
            sum_a_trans = sum_per_condition(df_trans, trans_a)
            sum_b_trans = sum_per_condition(df_trans, trans_b)
            
            mu_a, sig_a = sum_a_trans.mean(), sum_a_trans.std()
            mu_b, sig_b = sum_b_trans.mean(), sum_b_trans.std()
            
            mask_a = (sum_a_trans >= mu_a - 3*sig_a) & (sum_a_trans <= mu_a + 3*sig_a)
            mask_b = (sum_b_trans >= mu_b - 3*sig_b) & (sum_b_trans <= mu_b + 3*sig_b)
            
            df_trans = df_trans[mask_a & mask_b]
            st.info(f"Filtered: {len(df_trans):,} / {len(df_raw):,} proteins (within ±3σ)")
        
        protein_data.raw = df_trans
        st.session_state.protein_data = protein_data
        st.session_state.selected_transform_method = selected_transform
        st.success(f"✅ Stored: Raw + {TRANSFORM_NAMES.get(selected_transform, selected_transform)}")
