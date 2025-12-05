# pages/2_Visual_EDA.py

import streamlit as st
import pandas as pd

from helpers.transforms import apply_transformation, TRANSFORM_NAMES
from helpers.evaluation import (
    create_raw_row_figure,
    create_transformed_row_figure,
    evaluate_transformation_metrics,
)

st.set_page_config(page_title="Visual EDA", layout="wide")

st.title("📊 Visual EDA: Transformation Diagnostics")

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
protein_data = st.session_state.get("protein_data")
if protein_data is None:
    st.error("No data found. Please upload data first on the 'Data Upload' page.")
    st.stop()

df_raw = protein_data.raw
numeric_cols = protein_data.numeric_cols

if not numeric_cols:
    st.error("No numeric intensity columns found.")
    st.stop()

st.success(f"{len(df_raw):,} proteins × {len(numeric_cols)} intensity columns")

# ----------------------------------------------------------------------
# 1) Controls
# ----------------------------------------------------------------------
st.subheader("1️⃣ Select Transformations")

available_methods = [m for m in TRANSFORM_NAMES.keys() if m != "raw"]

selected_methods = st.multiselect(
    "Transformations to evaluate",
    options=available_methods,
    default=["log2", "log10", "sqrt", "arcsinh", "vst"],
    format_func=lambda x: TRANSFORM_NAMES.get(x, x),
)

if not selected_methods:
    st.info("Select at least one transformation to continue.")
    st.stop()

max_cols = st.slider(
    "Number of intensity columns used for diagnostics",
    min_value=3,
    max_value=min(12, len(numeric_cols)),
    value=6,
)
eval_cols = numeric_cols[:max_cols]

# ----------------------------------------------------------------------
# 2) Raw diagnostics row (reference)
# ----------------------------------------------------------------------
st.subheader("2️⃣ Raw Data Diagnostics (Reference)")

raw_fig = create_raw_row_figure(
    df_raw=df_raw,
    raw_cols=eval_cols,
    title="Raw Data",
)
st.plotly_chart(raw_fig, width="stretch")

# ----------------------------------------------------------------------
# 3) Per-transformation diagnostics + collect metrics
# ----------------------------------------------------------------------
st.subheader("3️⃣ Transformation Diagnostics")

all_metrics = []  # for bottom table

for method in selected_methods:
    nice_name = TRANSFORM_NAMES.get(method, method)
    st.markdown(f"#### 🔄 {nice_name}")

    # Transform only the eval_cols
    df_trans, trans_cols = apply_transformation(df_raw, eval_cols, method)

    # Plots
    trans_fig = create_transformed_row_figure(
        df_transformed=df_trans,
        trans_cols=trans_cols,
        title=nice_name,
    )
    st.plotly_chart(trans_fig, width="stretch")

    # Metrics
    metrics = evaluate_transformation_metrics(
        df_raw=df_raw,
        df_transformed=df_trans,
        raw_cols=eval_cols,
        trans_cols=trans_cols,
    )

    all_metrics.append(
        {
            "method": method,
            "Method": nice_name,
            "shapiro_raw": metrics["shapiro_raw"],
            "shapiro_trans": metrics["shapiro_trans"],
            "mean_var_corr_raw": metrics["mean_var_corr_raw"],
            "mean_var_corr_trans": metrics["mean_var_corr_trans"],
        }
    )

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

# ----------------------------------------------------------------------
# 4) Bottom ranking table (built from all_metrics)
# ----------------------------------------------------------------------
st.subheader("4️⃣ Transformation Ranking")

if not all_metrics:
    st.warning("No metrics collected.")
else:
    df_summary = pd.DataFrame(all_metrics)

    # Combined score: higher Shapiro_trans, lower |mean_var_corr_trans| is better
    df_summary["combined_score"] = (
        df_summary["shapiro_trans"].rank(ascending=False)
        + (1 - df_summary["mean_var_corr_trans"].abs()).rank(ascending=False)
    )

    # Sort by combined score (lower rank index = better)
    df_summary = df_summary.sort_values("combined_score").reset_index(drop=True)

    # Display table
    display_cols = [
        "Method",
        "shapiro_trans",
        "mean_var_corr_trans",
        "shapiro_raw",
        "mean_var_corr_raw",
        "combined_score",
    ]
    df_display = df_summary[display_cols].round(4)
    df_display.index = df_display.index + 1  # rank-like index

    st.dataframe(df_display, width="stretch")

    # Highlight best
    best = df_summary.iloc[0]
    st.success(
        f"🏆 Best: **{best['Method']}** "
        f"(Shapiro p={best['shapiro_trans']:.2e}, "
        f"Mean–Var Corr={best['mean_var_corr_trans']:.3f})"
    )
