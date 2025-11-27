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
c1 = st.session_state.prot_final_c1  # e.g. ["A1", "A2", "A3"]
c2 = st.session_state.prot_final_c2  # e.g. ["B1", "B2", "B3"]
all_reps = c1 + c2

st.title("Protein-Level Exploratory Analysis")
st.success(f"Analyzing {len(df):,} proteins • {len(c1)} vs {len(c2)} replicates")

# === 1. 6 INDIVIDUAL DENSITY PLOTS ===
st.subheader("1. Individual Intensity Density Plots (log₂)")

# Prepare log2 data
log_data = df[all_reps].replace(0, np.nan).apply(np.log2)

# 2 rows of 3
row1, row2 = st.columns(3), st.columns(3)
rows = [row1, row2]

for i, rep in enumerate(all_reps):
    col = rows[i // 3][i % 3]
    with col:
        fig = px.histogram(
            log_data[rep].dropna(),
            nbins=80,
            histnorm="density",
            title=f"<b>{rep}</b>",
            color_discrete_sequence=["#E71316" if rep in c1 else "#1f77b4"]
        )
        fig.add_vline(x=log_data[rep].median(), line_dash="dash", line_color="black")
        fig.update_layout(
            height=380,
            margin=dict(t=50, b=40),
            xaxis_title="log₂(Intensity)",
            yaxis_title="Density",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# === 2. KOLMOGOROV-SMIRNOV PAIRWISE TEST ===
st.subheader("2. Distribution Similarity Test (Kolmogorov-Smirnov)")

# Perform pairwise KS tests
ks_results = []
for rep1, rep2 in combinations(all_reps, 2):
    d1 = log_data[rep1].dropna()
    d2 = log_data[rep2].dropna()
    if len(d1) > 1 and len(d2) > 1:
        stat, p = stats.ks_2samp(d1, d2)
        sig = "Significant" if p < 0.05 else "Not significant"
        ks_results.append({
            "Replicate 1": rep1,
            "Replicate 2": rep2,
            "KS Statistic": f"{stat:.4f}",
            "p-value": f"{p:.2e}",
            "Different?": sig
        })

ks_df = pd.DataFrame(ks_results)
st.dataframe(ks_df.style.apply(lambda x: ["background: #ffcccc" if v == "Significant" else "" for v in x], subset=["Different?"]), use_container_width=True)

# Interpretation
if any(r["Different?"] == "Significant" for r in ks_results):
    st.error("**Warning**: At least one pair of replicates has significantly different distributions — check for technical bias!")
else:
    st.success("**All replicate distributions are statistically similar** — excellent technical reproducibility")

st.info("""
**Kolmogorov-Smirnov Test**  
- Compares full distribution shape (not just mean/median)  
- p < 0.05 → distributions are significantly different  
- Ideal for detecting subtle shifts or outliers in proteomics (Schessner et al., 2022)
""")

# === 3. OVERLAY HISTOGRAM (Figure 4A style) ===
st.subheader("3. Overlay Histogram (Schessner et al., Figure 4A)")

fig_overlay = go.Figure()
for rep in all_reps:
    fig_overlay.add_trace(go.Histogram(
        x=log_data[rep].dropna(),
        name=rep,
        opacity=0.6,
        nbinsx=100,
        histnorm="density"
    ))

fig_overlay.update_layout(
    barmode="overlay",
    height=500,
    xaxis_title="log₂(Intensity)",
    yaxis_title="Density",
    legend_title="Replicate",
    plot_bgcolor="white"
)
st.plotly_chart(fig_overlay, use_container_width=True)

# === 4. VIOLIN + BOX OVERLAY ===
st.subheader("4. Violin + Box Overlay")

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

# === FINAL SAVE ===
st.session_state.intensity_log2 = log_data
st.session_state.prot_final_df_log2 = df.copy()  # if needed later

st.success("All QC complete — data ready for differential analysis")

if st.button("Go to Differential Analysis", type="primary", use_container_width=True):
    st.switch_page("pages/4_Differential_Analysis.py")

st.markdown("---")
st.caption("© 2024 Thermo Fisher Scientific • Internal Use Only")
