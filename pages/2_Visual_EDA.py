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

st.title("📊 Global Normality by Transformation")

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

# ----------------------------------------------------------------------
# Helper: pooled normality stats
# ----------------------------------------------------------------------
def pooled_normality_stats(df: pd.DataFrame, cols: list) -> dict:
    x = df[cols].to_numpy().ravel()
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
# 2) Build stats table (Raw + transforms)
# ----------------------------------------------------------------------
st.subheader("2️⃣ Global Normality per Condition")

rows = []

# Raw
raw_stats = pooled_normality_stats(df_raw, eval_cols)
rows.append(
    {
        "Condition": "Raw",
        "key": "raw",
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
            "key": method,
            "W": stats_tr["W"],
            "p": stats_tr["p"],
            "Skew": stats_tr["skew"],
            "Kurtosis": stats_tr["kurt"],
            "n": stats_tr["n"],
        }
    )

table_df = pd.DataFrame(rows)

# Ranking (ignore Raw)
table_df["score"] = np.nan
mask_tr = table_df["key"] != "raw"
table_df.loc[mask_tr, "score"] = (
    table_df.loc[mask_tr, "W"].rank(ascending=False)
    + table_df.loc[mask_tr, "p"].rank(ascending=False)
)

# Add Show column: default Raw=True
table_df["Show"] = table_df["key"].eq("raw")

# ----------------------------------------------------------------------
# 3) Data editor with checkbox column
# ----------------------------------------------------------------------
edited = st.data_editor(
    table_df[["Condition", "W", "p", "Skew", "Kurtosis", "n", "score", "Show"]],
    use_container_width=True,
    num_rows="fixed",
    hide_index=True,
    column_config={
        "Condition": "Condition",
        "W": st.column_config.NumberColumn("W", format="%.4f"),
        "p": st.column_config.NumberColumn("p", format="%.2e"),
        "Skew": st.column_config.NumberColumn("Skew", format="%.3f"),
        "Kurtosis": st.column_config.NumberColumn("Kurtosis", format="%.3f"),
        "n": st.column_config.NumberColumn("n", format="%d"),
        "score": st.column_config.NumberColumn("Score", format="%.1f", disabled=True),
        "Show": st.column_config.CheckboxColumn(
            "Plot",
            help="Select exactly one condition to plot",
            default=False,
        ),
    },
    disabled=["Condition", "W", "p", "Skew", "Kurtosis", "n", "score"],
    key="normality_table",
)

# Enforce single selection: if multiple True, keep the first; if none, default to Raw
show_series = edited["Show"].copy()
if show_series.sum() == 0:
    # Default to Raw
    show_series.iloc[0] = True
elif show_series.sum() > 1:
    first_true = show_series[show_series].index[0]
    show_series[:] = False
    show_series.loc[first_true] = True

# Find selected key
selected_idx = show_series.idxmax()
selected_condition = edited.loc[selected_idx, "Condition"]
selected_key = table_df.loc[selected_idx, "key"]

# Show best transformation info
if mask_tr.any():
    best_row = table_df.loc[mask_tr].sort_values("score").iloc[0]
    st.success(
        f"🏆 Best transformation by W & p: **{best_row['Condition']}** "
        f"(W={best_row['W']:.3f}, p={best_row['p']:.2e})"
    )

st.info(f"📌 Currently selected for plotting: **{selected_condition}**")

# ----------------------------------------------------------------------
# 4) Plots on top of each other: Raw (top), selected (bottom)
# ----------------------------------------------------------------------
st.subheader("3️⃣ Diagnostic Plots (Raw vs Selected)")

st.markdown("**Raw (top)**")
raw_fig = create_raw_row_figure(
    df_raw=df_raw,
    raw_cols=eval_cols,
    title="Raw Data",
)
st.plotly_chart(raw_fig, use_container_width=True)

st.markdown("**Selected condition (bottom)**")
if selected_key == "raw":
    # Show raw again
    st.plotly_chart(raw_fig, use_container_width=True)
else:
    nice_name = TRANSFORM_NAMES.get(selected_key, selected_key)
    df_trans_sel, trans_cols_sel = apply_transformation(df_raw, eval_cols, selected_key)
    trans_fig = create_transformed_row_figure(
        df_transformed=df_trans_sel,
        trans_cols=trans_cols_sel,
        title=nice_name,
    )
    st.plotly_chart(trans_fig, use_container_width=True)
