# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import boxcox, yeojohnson
import scikit_posthocs as sp

# Load data
df = st.session_state.prot_final_df
c1 = st.session_state.prot_final_c1
c2 = st.session_state.prot_final_c2
all_reps = c1 + c2

st.title("Protein-Level QC & Transformation")

# === 1. INDIVIDUAL DENSITY PLOTS (Figure 4A style) ===
st.subheader("1. Individual Intensity Density Plots (Raw → log₂)")

raw_data = df[all_reps].replace(0, np.nan)
log2_data = np.log2(raw_data)

fig = go.Figure()

colors = ["#E71316"] * len(c1) + ["#1f77b4"] * len(c2)

for i, rep in enumerate(all_reps):
    vals = log2_data[rep].dropna()
    fig.add_trace(go.Violin(
        y=vals,
        name=rep,
        box_visible=True,
        meanline_visible=True,
        line_color=colors[i],
        fillcolor=colors[i] + "40",
        opacity=0.8
    ))

fig.update_layout(
    height=600,
    yaxis_title="log₂(Intensity)",
    xaxis_title="Replicate",
    showlegend=False,
    plot_bgcolor="white"
)

st.plotly_chart(fig, use_container_width=True)

# === 2. TEST FOR SIGNIFICANT DIFFERENCES ===
st.markdown("### Statistical Test: Are Replicates Significantly Different?")

# Kruskal-Wallis (non-parametric ANOVA)
kruskal = stats.kruskal(*[log2_data[rep].dropna() for rep in all_reps])
st.write(f"**Kruskal-Wallis test**: H = {kruskal.statistic:.2f}, p = {kruskal.pvalue:.2e}")

if kruskal.pvalue < 0.05:
    st.error("Replicates are significantly different (p < 0.05) — check for technical bias!")

    # Dunn's post-hoc test
    dunn = sp.posthoc_dunn([log2_data[rep].dropna() for rep in all_reps], p_adjust="holm")
    dunn.columns = all_reps
    dunn.index = all_reps
    st.write("**Dunn's post-hoc test (Holm-adjusted p-values)**")
    st.dataframe(dunn.style.format("{:.2e}").background_gradient(cmap="Reds"))
else:
    st.success("No significant differences between replicates (p ≥ 0.05) — good reproducibility!")

# === 3. TRANSFORMATION + RETEST ===
st.markdown("### Apply Transformation")
transform = st.selectbox(
    "Choose transformation",
    ["log₂ (recommended)", "log₁₀", "Box-Cox", "Yeo-Johnson", "None"],
    index=0
)

# Apply
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
    y_label = "Box-Cox"
elif transform == "Yeo-Johnson":
    transformed = pd.DataFrame(yeojohnson(raw_data.values.flatten())[0].reshape(raw_data.shape),
                               index=raw_data.index, columns=raw_data.columns)
    y_label = "Yeo-Johnson"
else:
    transformed = raw_data
    y_label = "Raw"

# === BLUE BOX: RESULT ===
st.markdown(f"""
<div style="background:#1976d2;padding:20px;border-radius:12px;color:white;">
    <h4>Applied Transformation: <strong>{transform}</strong></h4>
    <p>Post-transformation retest (Shapiro-Wilk on log₂ values):</p>
</div>
""", unsafe_allow_html=True)

post_stats = []
for rep in all_reps:
    vals = np.log2(raw_data[rep].dropna())
    _, p = stats.shapiro(vals)
    post_stats.append({"Replicate": rep, "Shapiro-Wilk p": f"{p:.2e}"})

st.dataframe(pd.DataFrame(post_stats), use_container_width=True)

if transform.startswith("log₂"):
    st.success("**log₂ transformation applied — gold standard**")
else:
    st.info(f"Transformation: {transform}")

# Save
st.session_state.intensity_raw = raw_data
st.session_state.intensity_log2 = np.log2(raw_data)

st.info("Data saved: `intensity_raw` and `intensity_log2` ready for analysis")

if st.button("Go to Differential Analysis", type="primary", use_container_width=True):
    st.switch_page("pages/4_Differential_Analysis.py")
