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

df_raw: pd.DataFrame = protein_data.raw
numeric_cols = protein_data.numeric_cols

if not numeric_cols:
    st.error("No numeric intensity columns found.")
    st.stop()

st.success(f"{len(df_raw):,} proteins × {len(numeric_cols)} intensity columns")

# ----------------------------------------------------------------------
# 1) Controls
# ----------------------------------------------------------------------
st.subheader("1️⃣ Configuration")

available_methods = [m for m in TRANSFORM_NAMES.keys() if m != "raw"]

max_cols = st.slider(
    "Number of intensity columns used for diagnostics",
    min_value=3,
    max_value=min(12, len(numeric_cols)),
    value=6,
)
eval_cols = numeric_cols[:max_cols]

# We will evaluate all methods (so the table is complete)
methods_to_evaluate = available_methods

# ----------------------------------------------------------------------
# 2) Compute stats for all methods (once)
# ----------------------------------------------------------------------
st.subheader("2️⃣ Transformation Statistics")

summary_rows = []

for method in methods_to_evaluate:
    nice_name = TRANSFORM_NAMES.get(method, method)

    # Transform only eval_cols
    df_trans, trans_cols = apply_transformation(df_raw, eval_cols, method)

    # Metrics (Shapiro W is inside SciPy's output, but we stored p-values and correlations)
    metrics = evaluate_transformation_metrics(
        df_raw=df_raw,
        df_transformed=df_trans,
        raw_cols=eval_cols,
        trans_cols=trans_cols,
    )

    # To also show W-statistic, we recompute Shapiro-W here explicitly on transformed data
    # (metrics already has p-value; we can get W together)
    trans_vals = df_trans[trans_cols].to_numpy().ravel()
    trans_vals = trans_vals[~pd.isna(trans_vals)]
    if len(trans_vals) >= 3:
        W_trans, p_trans = stats.shapiro(trans_vals[: min(5000, len(trans_vals))])
    else:
        W_trans, p_trans = float("nan"), float("nan")

    raw_vals = df_raw[eval_cols].to_numpy().ravel()
    raw_vals = raw_vals[~pd.isna(raw_vals)]
    if len(raw_vals) >= 3:
        W_raw, p_raw = stats.shapiro(raw_vals[: min(5000, len(raw_vals))])
    else:
        W_raw, p_raw = float("nan"), float("nan")

    summary_rows.append(
        {
            "method": method,
            "Method": nice_name,
            "W_raw": W_raw,
            "p_raw": p_raw,
            "W_trans": W_trans,
            "p_trans": p_trans,
            "mean_var_corr_raw": metrics["mean_var_corr_raw"],
            "mean_var_corr_trans": metrics["mean_var_corr_trans"],
        }
    )

if not summary_rows:
    st.warning("No transformations evaluated.")
    st.stop()

df_summary = pd.DataFrame(summary_rows)

# Combined score: higher W_trans and p_trans, lower |mean_var_corr_trans|
df_summary["combined_score"] = (
    df_summary["W_trans"].rank(ascending=False)
    + df_summary["p_trans"].rank(ascending=False)
    + (1 - df_summary["mean_var_corr_trans"].abs()).rank(ascending=False)
)

df_summary = df_summary.sort_values("combined_score").reset_index(drop=True)

# Display table
display_cols = [
    "Method",
    "W_raw",
    "p_raw",
    "W_trans",
    "p_trans",
    "mean_var_corr_raw",
    "mean_var_corr_trans",
    "combined_score",
]
df_display = df_summary[display_cols].round(4)
df_display.index = df_display.index + 1  # rank-like index

st.dataframe(df_display, width="stretch")

best = df_summary.iloc[0]
st.success(
    f"🏆 Best: **{best['Method']}** "
    f"(W={best['W_trans']:.3f}, p={best['p_trans']:.2e}, "
    f"Mean–Var Corr={best['mean_var_corr_trans']:.3f})"
)

# ----------------------------------------------------------------------
# 3) Raw plot (always visible) + selection of a single transformation
# ----------------------------------------------------------------------
st.subheader("3️⃣ Diagnostics Plots")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("**Raw Data (Reference)**")
    raw_fig = create_raw_row_figure(
        df_raw=df_raw,
        raw_cols=eval_cols,
        title="Raw Data",
    )
    st.plotly_chart(raw_fig, width="stretch")

# Choose exactly one transformation to visualize
methods_for_radio = df_summary["method"].tolist()
default_best_index = 0  # first row is best after sorting
selected_method = st.radio(
    "Select transformation to visualize",
    options=methods_for_radio,
    index=default_best_index,
    format_func=lambda m: TRANSFORM_NAMES.get(m, m),
)

with col_right:
    nice_name = TRANSFORM_NAMES.get(selected_method, selected_method)
    st.markdown(f"**{nice_name}**")

    # Reuse the same transform for this method (could cache if needed)
    df_trans, trans_cols = apply_transformation(df_raw, eval_cols, selected_method)
    trans_fig = create_transformed_row_figure(
        df_transformed=df_trans,
        trans_cols=trans_cols,
        title=nice_name,
    )
    st.plotly_chart(trans_fig, width="stretch")
