# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from itertools import combinations

# Load data
if "prot_final_df" not in st.session_state:
    st.error("No protein data found! Please go to Protein Import first.")
    st.stop()

df = st.session_state.prot_final_df
c1 = st.session_state.prot_final_c1
c2 = st.session_state.prot_final_c2
all_reps = c1 + c2

st.title("Protein-Level Exploratory Analysis")
st.success(f"Analyzing {len(df):,} proteins • {len(c1)} vs {len(c2)} replicates")

# === 1. 6 INDIVIDUAL DENSITY PLOTS (RAW) ===
st.subheader("1. Individual Intensity Density Plots (Raw Data)")

raw_data = df[all_reps].replace(0, np.nan)

row1 = st.columns(3)
row2 = st.columns(3)

for i, rep in enumerate(all_reps):
    col = row1[i] if i < 3 else row2[i-3]
    with col:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=raw_data[rep].dropna(),
            nbinsx=100,
            histnorm="density",
            name=rep,
            marker_color="#E71316" if rep in c1 else "#1f77b4",
            opacity=0.75
        ))
        median = raw_data[rep].median()
        fig.add_vline(x=median, line_dash="dash", line_color="black")
        fig.update_layout(
            height=380,
            title=f"<b>{rep}</b>",
            xaxis_title="Raw Intensity",
            yaxis_title="Density",
            showlegend=False,
            plot_bgcolor="white"
        )
        st.plotly_chart(fig, use_container_width=True)

# === 2. KS TEST ON RAW DATA + TEXT SUMMARY ONLY ===
st.subheader("2. Distribution Similarity (Kolmogorov-Smirnov Test)")

significant_pairs = []
for rep1, rep2 in combinations(all_reps, 2):
    d1 = raw_data[rep1].dropna()
    d2 = raw_data[rep2].dropna()
    if len(d1) > 1 and len(d2) > 1:
        _, p = stats.ks_2samp(d1, d2)
        if p < 0.05:
            significant_pairs.append(f"{rep1} vs {rep2}")

if significant_pairs:
    st.error("**Warning: Significant distribution differences detected (p < 0.05):**  \n" + " • ".join(significant_pairs))
    st.info("This is expected on raw intensities — we will now apply log₂ transformation (standard practice)")
else:
    st.success("**All replicates have statistically similar distributions** — excellent technical quality")

# === 3. LOG₂ TRANSFORMATION & STANDARD PLOTS ===
st.markdown("### 3. Standard log₂ Transformation (Schessner et al., 2022)")

log_data = np.log2(raw_data)

# Overlay histogram (Figure 4A)
st.subheader("Overlay Histogram — log₂(Intensity)")
fig_overlay = go.Figure()
for rep in all_reps:
    fig_overlay.add_trace(go.Histogram(
        x=log_data[rep].dropna(),
        name=rep,
        opacity=0.6,
        nbinsx=100,
        histnorm="density"
    ))
fig_overlay.update_layout(barmode="overlay", height=500, xaxis_title="log₂(Intensity)", yaxis_title="Density")
st.plotly_chart(fig_overlay, use_container_width=True)

# Violin + box
st.subheader("Violin + Box Plot — log₂(Intensity)")
melted = log_data.melt(var_name="Replicate", value_name="log₂(Intensity)").dropna()
melted["Condition"] = melted["Replicate"].apply(lambda x: "A" if x in c1 else "B")
melted["Replicate"] = pd.Categorical(melted["Replicate"], all_reps, ordered=True)

fig_violin = px.violin(
    melted, x="Replicate", y="log₂(Intensity)", color="Condition",
    color_discrete_map={"A": "#E71316", "B": "#1f77b4"},
    box=True, points=False, violinmode="overlay"
)
fig_violin.update_traces(box_visible=True, meanline_visible=True)
fig_violin.update_layout(height=650)
st.plotly_chart(fig_violin, use_container_width=True)

st.success("**log₂ transformation applied — gold standard in proteomics**")
st.info("Data ready for differential analysis, PCA, volcano plot, etc.")

if st.button("Go to Differential Analysis", type="primary", use_container_width=True):
    st.switch_page("pages/4_Differential_Analysis.py")
