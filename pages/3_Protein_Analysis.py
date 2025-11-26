# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import boxcox, yeojohnson

# Load data
df = st.session_state.prot_final_df
c1 = st.session_state.prot_final_c1
c2 = st.session_state.prot_final_c2

st.title("Protein-Level QC & Transformation")

# === 1. INTENSITY HISTOGRAMS (Schessner Figure 4A) ===
st.subheader("1. Raw Intensity Histograms (Before Transformation)")

raw_data = df[c1 + c2].replace(0, np.nan)
fig_hist = go.Figure()

for i, rep in enumerate(c1 + c2):
    vals = raw_data[rep].dropna()
    fig_hist.add_trace(go.Histogram(
        x=np.log2(vals),
        name=rep,
        opacity=0.6,
        nbinsx=100,
        marker_color="#E71316" if rep in c1 else "#1f77b4"
    ))

fig_hist.update_layout(
    barmode="overlay",
    height=500,
    xaxis_title="log₂(Intensity)",
    yaxis_title="Number of Proteins",
    legend_title="Replicate",
    plot_bgcolor="white"
)
st.plotly_chart(fig_hist, use_container_width=True)

# === 2. RAW DATA STATISTICS ===
st.markdown("### 2. Raw Data Distribution Diagnosis")
stats_raw = []
for rep in c1 + c2:
    vals = raw_data[rep].dropna()
    if len(vals) < 4:
        stats_raw.append({"Replicate": rep, "n": len(vals), "Skew": "N/A", "Kurtosis": "N/A", "Shapiro p": "N/A"})
        continue
    skew = stats.skew(np.log2(vals))
    kurt = stats.kurtosis(np.log2(vals))
    _, p = stats.shapiro(np.log2(vals))
    stats_raw.append({
        "Replicate": rep,
        "n": len(vals),
        "Skew": f"{skew:+.3f}",
        "Kurtosis": f"{kurt:+.3f}",
        "Shapiro p": f"{p:.2e}"
    })

st.dataframe(pd.DataFrame(stats_raw), use_container_width=True)

# === 3. TRANSFORMATION SELECTION ===
st.markdown("### 3. Apply Transformation")
transform = st.selectbox(
    "Choose transformation",
    ["log₂ (recommended)", "log₁₀", "Box-Cox", "Yeo-Johnson", "None"],
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

# === 4. FINAL VIOLIN + BOX PLOT ===
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

# === 5. BLUE BOX: TRANSFORMATION + RETEST RESULT ===
st.markdown("### Current Transformation & Post-Test Result")
with st.container():
    st.markdown(f"""
    <div style="background:#1e88e5;padding:15px;border-radius:10px;color:white;">
        <h4>Applied: <strong>{transform}</strong></h4>
        <p>After transformation:</p>
    </div>
    """, unsafe_allow_html=True)

    # Retest
    post_stats = []
    for rep in c1 + c2:
        vals = transformed[rep].dropna()
        if len(vals) < 4: continue
        skew = stats.skew(vals)
        kurt = stats.kurtosis(vals)
        _, p = stats.shapiro(vals)
        post_stats.append({"Replicate": rep, "Skew": f"{skew:+.3f}", "Kurtosis": f"{kurt:+.3f}", "p": f"{p:.2e}"})

    st.dataframe(pd.DataFrame(post_stats), use_container_width=True)

    # Final verdict
    if transform.startswith("log₂"):
        st.success("**log₂ transformation applied — gold standard in proteomics**")
    else:
        st.info(f"Transformation applied: {transform}")

# === SAVE DATA ===
st.session_state.intensity_raw = raw_data
st.session_state.intensity_transformed = transformed
st.session_state.transform_applied = transform

st.info("Data saved: `intensity_raw` and `intensity_transformed` ready for downstream analysis")
