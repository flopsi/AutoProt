import streamlit as st

from helpers.transforms import apply_transformation, TRANSFORM_NAMES
from helpers.evaluation import (
    create_raw_row_figure,
    create_transformed_row_figure,
    evaluate_transformation_metrics,
)
from helpers.comparison import compare_transformations

st.set_page_config(page_title="Visual EDA", layout="wide")


protein_data = st.session_state.get("protein_data")
if protein_data is None:
    st.error("No data found. Please upload data first.")
    st.stop()

df_raw = protein_data.raw
numeric_cols = protein_data.numeric_cols
if not numeric_cols:
    st.error("No numeric intensity columns found.")
    st.stop()

# simple file_hash: use file path or shape string
file_hash = getattr(protein_data, "file_path", str(df_raw.shape))

st.success(f"{len(df_raw):,} proteins × {len(numeric_cols)} intensity columns")

# controls
st.subheader("1️⃣ Select Transformations")
available_methods = [m for m in TRANSFORM_NAMES.keys() if m != "raw"]
selected_methods = st.multiselect(
    "Transformations to evaluate",
    options=available_methods,
    default=["log2", "log10", "sqrt", "arcsinh", "vst"],
    format_func=lambda x: TRANSFORM_NAMES.get(x, x),
)
if not selected_methods:
    st.info("Select at least one transformation.")
    st.stop()

max_cols = st.slider(
    "Number of intensity columns used for diagnostics",
    min_value=3,
    max_value=min(12, len(numeric_cols)),
    value=6,
)
eval_cols = numeric_cols[:max_cols]

# raw row (cached inside)
st.subheader("2️⃣ Raw Data Diagnostics (Reference)")
raw_fig = create_raw_row_figure(df_raw=df_raw, raw_cols=eval_cols, title="Raw Data", file_hash=file_hash)
st.plotly_chart(raw_fig, use_container_width=True)

# transformed rows (cached transforms + metrics)
st.subheader("3️⃣ Transformation Diagnostics")
for method in selected_methods:
    st.markdown(f"#### 🔄 {TRANSFORM_NAMES.get(method, method)}")

    df_trans, trans_cols = cached_apply_transformation(df_raw, eval_cols, method, file_hash)
    trans_fig = create_transformed_row_figure(
        df_transformed=df_trans,
        trans_cols=trans_cols,
        title=TRANSFORM_NAMES.get(method, method),
    )
    st.plotly_chart(trans_fig, use_container_width=True)

    metrics = evaluate_transformation_metrics(
        df_raw=df_raw,
        df_transformed=df_trans,
        raw_cols=eval_cols,
        trans_cols=trans_cols,
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

# bottom ranking (cached)
st.subheader("4️⃣ Transformation Ranking")
summary_df, _ = compare_transformations(
    df_raw=df_raw,
    numeric_cols=eval_cols,
    methods=selected_methods,
    file_hash=file_hash,
)

if summary_df.empty:
    st.warning("No comparison results.")
else:
    summary_df = summary_df.copy()
    summary_df["Method"] = summary_df["method"].map(lambda x: TRANSFORM_NAMES.get(x, x))
    summary_df = summary_df[["Method", "shapiro_p", "mean_var_corr", "combined_score"]].round(4)
    summary_df.index = summary_df.index + 1
    st.dataframe(summary_df, use_container_width=True)

    best = summary_df.iloc[0]
    st.success(
        f"🏆 Best: **{best['Method']}** "
        f"(Shapiro p={best['shapiro_p']:.2e}, "
        f"Mean–Var Corr={best['mean_var_corr']:.3f})"
    )
