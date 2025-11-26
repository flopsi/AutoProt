# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
from scipy.stats import boxcox, yeojohnson

# Load data
df = st.session_state.prot_final_df
c1 = st.session_state.prot_final_c1
c2 = st.session_state.prot_final_c2

st.title("Protein-Level QC: Choose Best Transformation")

# === STEP 1: SHOW RAW DISTRIBUTION + STATISTICS FIRST ===
st.subheader("1. Raw Intensity Distribution (No Transformation)")

raw_data = df[c1 + c2].replace(0, np.nan)
raw_melted = raw_data.melt(var_name="Replicate", value_name="Raw Intensity").dropna()
raw_melted["Condition"] = raw_melted["Replicate"].apply(lambda x: "A" if x in c1 else "B")
raw_melted["Replicate"] = pd.Categorical(raw_melted["Replicate"], c1 + c2, ordered=True)

fig_raw = px.violin(
    raw_melted, x="Replicate", y="Raw Intensity", color="Condition",
    color_discrete_map={"A": "#E71316", "B": "#1f77b4"},
    box=True, points=False, violinmode="overlay"
)
fig_raw.update_traces(box_visible=True, meanline_visible=True)
fig_raw.update_layout(height=600, yaxis_type="log")
st.plotly_chart(fig_raw, use_container_width=True)

# === STATISTICS ON RAW DATA ===
st.markdown("### Raw Data Statistics (Before Any Transformation)")
stats_raw = []
for rep in c1 + c2:
    vals = raw_data[rep].dropna()
    if len(vals) < 4:
        stats_raw.append({"Replicate": rep, "n": len(vals), "Skew": "N/A", "Kurtosis": "N/A", "Shapiro p": "N/A"})
        continue
    skew = stats.skew(vals)
    kurt = stats.kurtosis(vals)
    _, p = stats.shapiro(vals)
    stats_raw.append({
        "Replicate": rep,
        "n": len(vals),
        "Skew": f"{skew:+.3f}",
        "Kurtosis": f"{kurt:+.3f}",
        "Shapiro p": f"{p:.2e}"
    })

st.dataframe(pd.DataFrame(stats_raw), use_container_width=True)

# === RECOMMENDATION BASED ON RAW DATA ===
high_skew = any(abs(float(r["Skew"].replace("N/A","0"))) > 1.0 for r in stats_raw if r["Skew"] != "N/A")
high_kurt = any(abs(float(r["Kurtosis"].replace("N/A","0"))) > 3.0 for r in stats_raw if r["Kurtosis"] != "N/A")

if high_skew or high_kurt:
    st.error("**Raw data is highly skewed and leptokurtic — transformation REQUIRED**")
    st.info("→ Try **log₂** first (standard in proteomics)")
else:
    st.warning("Raw data may be usable — but log₂ is still recommended")

# === STEP 2: LET USER CHOOSE TRANSFORMATION ===
st.markdown("### 2. Apply Transformation")
transform = st.selectbox(
    "Choose transformation",
    ["log₂ (recommended)", "log₁₀", "Box-Cox", "Yeo-Johnson", "None (keep raw)"],
    index=0
)

# Apply transformation
if transform == "log₂ (recommended)":
    transformed = np.log2(raw_data)
    y_label = "log₂(Intensity)"
elif transform == "log₁₀":
    transformed = np.log10(raw_data)
    y_label = "log₁₀(Intensity)"
elif transform == "Box-Cox":
    shifted = raw_data - raw_data.min().min() + 1
    transformed = pd.DataFrame(boxcox(shifted.values.flatten())[0].reshape(shifted.shape),
                               index=raw_data.index, columns=raw_data.columns)
    y_label = "Box-Cox(Intensity)"
elif transform == "Yeo-Johnson":
    transformed = pd.DataFrame(yeojohnson(raw_data.values.flatten())[0].reshape(raw_data.shape),
                               index=raw_data.index, columns=raw_data.columns)
    y_label = "Yeo-Johnson(Intensity)"
else:
    transformed = raw_data
    y_label = "Raw Intensity"

# Save both
st.session_state.intensity_raw = raw_data
st.session_state.intensity_transformed = transformed
st.session_state.transform_applied = transform

# === FINAL PLOT AFTER TRANSFORMATION ===
melted = transformed.melt(var_name="Replicate", value_name=y_label).dropna()
melted["Condition"] = melted["Replicate"].apply(lambda x: "A" if x in c1 else "B")
melted["Replicate"] = pd.Categorical(melted["Replicate"], c1 + c2, ordered=True)

fig = px.violin(
    melted, x="Replicate", y=y_label, color="Condition",
    color_discrete_map={"A": "#E71316", "B": "#1f77b4"},
    box=True, points=False, violinmode="overlay"
)
fig.update_traces(box_visible=True, meanline_visible=True)
fig.update_layout(height=650, yaxis_title=y_label)
st.plotly_chart(fig, use_container_width=True)

# Final verdict
if transform.startswith("log₂"):
    st.success("**log₂ transformation applied — this is the gold standard in proteomics**")
    st.info("Ready for PCA, volcano plot, t-tests, etc.")
else:
    st.info(f"Transformation applied: {transform}")

st.markdown("**Data saved:** `st.session_state.intensity_transformed` (for analysis) | `intensity_raw` (original)")

st.markdown("---")
if st.button("Go to Differential Analysis", type="primary", use_container_width=True):
    st.switch_page("pages/4_Differential_Analysis.py")
