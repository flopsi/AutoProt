# pages/2_Visual_EDA.py

import streamlit as st
import pandas as pd

from helpers.transforms import apply_transformation, TRANSFORM_NAMES, TRANSFORM_DESCRIPTIONS
from helpers.evaluation import create_evaluation_figure, evaluate_transformation_metrics
from helpers.comparison import compare_transformations
st.set_page_config(page_title="Visual EDA", layout="wide")

st.title("📊 Visual EDA: Transformation Diagnostics")

# ---------------------------------------------------------------------
# Load data from session
# ---------------------------------------------------------------------
protein_data = st.session_state.get("protein_data")
if protein_data is None:
    st.error("No data found. Please upload data on the 'Data Upload' page first.")
    st.stop()

df_raw = protein_data.raw
numeric_cols = protein_data.numeric_cols

if not numeric_cols:
    st.error("No numeric columns found in data.")
    st.stop()

st.success(f"Data: {len(df_raw):,} proteins × {len(numeric_cols)} intensity columns")

# ---------------------------------------------------------------------
# Top controls: multi-select of transformations
# ---------------------------------------------------------------------
st.subheader("1️⃣ Select Transformations to Evaluate")

available_methods = [m for m in TRANSFORM_NAMES.keys() if m != "raw"]

selected_methods = st.multiselect(
    "Transformations",
    options=available_methods,
    default=["log2", "log10", "sqrt", "arcsinh", "vst"],
    format_func=lambda x: TRANSFORM_NAMES.get(x, x),
)

if not selected_methods:
    st.info("Select at least one transformation to see diagnostics.")
    st.stop()

# Limit number of columns used for evaluation (to keep plots readable)
max_cols = st.slider(
    "Number of intensity columns to use for evaluation",
    min_value=3,
    max_value=min(12, len(numeric_cols)),
    value=6,
)
eval_cols = numeric_cols[:max_cols]

# ---------------------------------------------------------------------
# For each selected transformation:
# Row block with 2×3 plots (raw vs transformed) and metrics
# ---------------------------------------------------------------------
st.subheader("2️⃣ Diagnostic Plots per Transformation")

all_metrics = {}

for method in selected_methods:
    st.markdown(f"### 🔄 {TRANSFORM_NAMES.get(method, method)}")

    # Apply transformation to the whole data, but only use eval_cols for plotting
    df_transformed, trans_cols = apply_transformation(df_raw, eval_cols, method)

    # Create evaluation plots (raw vs transformed) for these columns
    fig = create_evaluation_figure(
        df_raw=df_raw,
        df_transformed=df_transformed,
        raw_cols=eval_cols,
        trans_cols=trans_cols,
        title=TRANSFORM_NAMES.get(method, method),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Compute metrics (Shapiro p, mean–variance correlations)
    metrics = evaluate_transformation_metrics(
        df_raw=df_raw,
        df_transformed=df_transformed,
        raw_cols=eval_cols,
        trans_cols=trans_cols,
    )
    all_metrics[method] = metrics

    # Show metrics under the plots
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Shapiro p (Raw)", f"{metrics['shapiro_raw']:.2e}")
    with c2:
        st.metric("Shapiro p (Trans)", f"{metrics['shapiro_trans']:.2e}")
    with c3:
        st.metric("Mean–Var Corr (Raw)", f"{metrics['mean_var_corr_raw']:.3f}")
    with c4:
        st.metric("Mean–Var Corr (Trans)", f"{metrics['mean_var_corr_trans']:.3f}")

    st.markdown("---")

# ---------------------------------------------------------------------
# Bottom: Comparison table across all methods
# ---------------------------------------------------------------------
st.subheader("3️⃣ Transformation Ranking")

summary_df, _ = compare_transformations(
    df_raw=df_raw,
    numeric_cols=eval_cols,
    methods=selected_methods,
)

if summary_df.empty:
    st.warning("No comparison results could be computed.")
    st.stop()

# Map method → nice name for display
summary_df = summary_df.copy()
summary_df["Method"] = summary_df["method"].map(lambda x: TRANSFORM_NAMES.get(x, x))

# Order columns
display_cols = ["Method", "shapiro_p", "mean_var_corr", "combined_score"]
summary_df = summary_df[display_cols].round(4)

# Sort by combined_score (lower rank = better)
summary_df = summary_df.sort_values("combined_score", ascending=True).reset_index(drop=True)
summary_df.index = summary_df.index + 1  # rank 1-based

st.dataframe(summary_df, use_container_width=True)

# Highlight best
best_row = summary_df.iloc[0]
st.success(
    f"🏆 Best transformation: **{best_row['Method']}** "
    f"(Shapiro p={best_row['shapiro_p']:.2e}, "
    f"Mean–Var Corr={best_row['mean_var_corr']:.3f})"
)
