# pages/2_Visual_EDA.py

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

from helpers.transforms import apply_transformation, TRANSFORM_NAMES
from helpers.evaluation import (
    create_raw_row_figure,
    create_transformed_row_figure,
)

st.set_page_config(page_title="Visual EDA", layout="wide")

st.title("📊 Visual EDA: Global Normality by Transformation")

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
# 1) Configuration
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

# Methods to evaluate (all)
methods_to_evaluate = ["raw"] + available_methods

# ----------------------------------------------------------------------
# Helper: compute pooled normality stats
# ----------------------------------------------------------------------
def pooled_normality_stats(df: pd.DataFrame, cols: list) -> dict:
    """
    Flatten all given columns into one array and compute:
    - Shapiro W, p
    - Skewness, kurtosis (Fisher)
    """
    x = df[cols].to_numpy().ravel()
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return {"n": n, "W": np.nan, "p": np.nan, "skew": np.nan, "kurt": np.nan}

    # subsample for very large n to keep Shapiro stable/fast
    if n > 5000:
        x = np.random.choice(x, size=5000, replace=False)
        n = len(x)

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
# 2) Global normality table (one row per condition)
# ----------------------------------------------------------------------
st.subheader("2️⃣ Global Normality per Condition")

rows = []

# Raw row
raw_stats = pooled_normality_stats(df_raw, eval_cols)
rows.append(
    {
        "Condition": "Raw",
        "W": raw_stats["W"],
        "p": raw_stats["p"],
        "Skew": raw_stats["skew"],
        "Kurtosis": raw_stats["kurt"],
        "n": raw_stats["n"],
    }
)

# Each transformation
for method in available_methods:
    df_trans, trans_cols = apply_transformation(df_raw, eval_cols, method)
    stats_tr = pooled_normality_stats(df_trans, trans_cols)
    rows.append(
        {
            "Condition": TRANSFORM_NAMES.get(method, method),
            "W": stats_tr["W"],
            "p": stats_tr["p"],
            "Skew": stats_tr["skew"],
            "Kurtosis": stats_tr["kurt"],
            "n": stats_tr["n"],
            "method_key": method,
        }
    )

table_df = pd.DataFrame(rows)

# Compute ranking (ignore Raw)
table_df["score"] = np.nan
mask_tr = table_df["Condition"] != "Raw"
table_df.loc[mask_tr, "score"] = (
    table_df.loc[mask_tr, "W"].rank(ascending=False)
    + table_df.loc[mask_tr, "p"].rank(ascending=False)
)

# Order: Raw first, then by score
table_df = table_df.sort_values(
    by=["Condition", "score"], key=lambda s: (s == "Raw").astype(int), ascending=[False, True]
).reset_index(drop=True)

disp = table_df[["Condition", "W", "p", "Skew", "Kurtosis", "n", "score"]].round(4)
disp.index = disp.index + 1
st.dataframe(disp, width="stretch")

# Best transformation (highest W & p)
best_row = table_df[table_df["Condition"] != "Raw"].sort_values("score").iloc[0]
best_method_key = best_row["method_key"]
st.success(
    f"🏆 Best transformation: **{best_row['Condition']}** "
    f"(W={best_row['W']:.3f}, p={best_row['p']:.2e})"
)

# ----------------------------------------------------------------------
# 3) Plots: Raw (fixed) + 1 selected transformation
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

# Radio: choose one condition (only transformed ones)
method_keys = [row["method_key"] for _, row in table_df[table_df["Condition"] != "Raw"].iterrows()]
default_index = method_keys.index(best_method_key) if best_method_key in method_keys else 0
selected_method = st.radio(
    "Select transformation to visualize",
    options=method_keys,
    index=default_index,
    format_func=lambda m: TRANSFORM_NAMES.get(m, m),
)

with col_right:
    nice_name = TRANSFORM_NAMES.get(selected_method, selected_method)
    st.markdown(f"**{nice_name}**")

    df_trans_sel, trans_cols_sel = apply_transformation(df_raw, eval_cols, selected_method)
    trans_fig = create_transformed_row_figure(
        df_transformed=df_trans_sel,
        trans_cols=trans_cols_sel,
        title=nice_name,
    )
    st.plotly_chart(trans_fig, use_container_width=True)
