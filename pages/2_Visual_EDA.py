# pages/2_Visual_EDA.py

import streamlit as st
import pandas as pd
from scipy import stats

from helpers.transforms import apply_transformation, TRANSFORM_NAMES
from helpers.evaluation import (
    create_raw_row_figure,
    create_transformed_row_figure,
    evaluate_transformation_metrics,
)
from helpers.statistics import test_normality_all_samples

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

# We’ll evaluate all methods (and show all in table),
# but for plots only one will be active at a time.
methods_to_evaluate = available_methods

# ----------------------------------------------------------------------
# 2) Compute normality table for all methods
# ----------------------------------------------------------------------
st.subheader("2️⃣ Normality Table (Per Sample, All Transformations)")

summary_rows = []

for method in methods_to_evaluate:
    df_trans, trans_cols = apply_transformation(df_raw, eval_cols, method)

    # Per-sample normality (raw vs transformed) using your helper
    norm_df = test_normality_all_samples(
        df_raw=df_raw,
        df_transformed=df_trans,
        numeric_cols=eval_cols,
        alpha=0.05,
    )

    # Aggregate: average W and p across samples
    W_raw_mean = norm_df["Raw_Statistic"].mean()
    p_raw_mean = norm_df["Raw_P_Value"].mean()
    W_trans_mean = norm_df["Trans_Statistic"].mean()
    p_trans_mean = norm_df["Trans_P_Value"].mean()

    summary_rows.append(
        {
            "method": method,
            "Method": TRANSFORM_NAMES.get(method, method),
            "W_raw_mean": W_raw_mean,
            "p_raw_mean": p_raw_mean,
            "W_trans_mean": W_trans_mean,
            "p_trans_mean": p_trans_mean,
        }
    )

# Build summary table
summary_df = pd.DataFrame(summary_rows)

if summary_df.empty:
    st.warning("No transformations evaluated.")
    st.stop()

# Combined score: higher W_trans_mean and p_trans_mean is better
summary_df["combined_score"] = (
    summary_df["W_trans_mean"].rank(ascending=False)
    + summary_df["p_trans_mean"].rank(ascending=False)
)

# Sort by best combined score
summary_df = summary_df.sort_values("combined_score").reset_index(drop=True)

# Display full table
display_df = summary_df.copy()
display_df.index = display_df.index + 1
display_df = display_df[
    ["Method", "W_raw_mean", "p_raw_mean", "W_trans_mean", "p_trans_mean", "combined_score"]
].round(4)

st.dataframe(display_df, width="stretch")

best_row = summary_df.iloc[0]
st.success(
    f"🏆 Best by mean Shapiro: **{best_row['Method']}** "
    f"(W={best_row['W_trans_mean']:.3f}, p={best_row['p_trans_mean']:.2e})"
)

# ----------------------------------------------------------------------
# 3) Raw plot (static) + single selected transformation plot
# ----------------------------------------------------------------------
st.subheader("3️⃣ Diagnostic Plots")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**Raw Data (Reference)**")
    raw_fig = create_raw_row_figure(
        df_raw=df_raw,
        raw_cols=eval_cols,
        title="Raw Data",
    )
    st.plotly_chart(raw_fig, use_container_width=True)

# Radio: exactly one transformation selected for plotting
methods_for_radio = summary_df["method"].tolist()
default_index = 0  # best method
selected_method = st.radio(
    "Select transformation to visualize",
    options=methods_for_radio,
    index=default_index,
    format_func=lambda m: TRANSFORM_NAMES.get(m, m),
)

with col_right:
    nice_name = TRANSFORM_NAMES.get(selected_method, selected_method)
    st.markdown(f"**{nice_name}**")

    df_trans_selected, trans_cols_selected = apply_transformation(df_raw, eval_cols, selected_method)
    trans_fig = create_transformed_row_figure(
        df_transformed=df_trans_selected,
        trans_cols=trans_cols_selected,
        title=nice_name,
    )
    st.plotly_chart(trans_fig, use_container_width=True)

# ----------------------------------------------------------------------
# 4) Detailed per-sample table for selected transformation
# ----------------------------------------------------------------------
st.subheader("4️⃣ Per-Sample Normality (Selected Transformation)")

norm_selected = test_normality_all_samples(
    df_raw=df_raw,
    df_transformed=df_trans_selected,
    numeric_cols=eval_cols,
    alpha=0.05,
)

# Make column names nicer
norm_display = norm_selected.rename(
    columns={
        "Sample": "Sample",
        "N": "N",
        "Raw_Statistic": "W_raw",
        "Raw_P_Value": "p_raw",
        "Trans_Statistic": "W_trans",
        "Trans_P_Value": "p_trans",
        "Improvement": "Improvement",
    }
)
st.dataframe(norm_display.round(4), width="stretch")
