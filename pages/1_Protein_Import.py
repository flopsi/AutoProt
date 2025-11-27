# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
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

st.title("Protein-Level QC & Transformation Recommendation")

# === USER SELECTS CONSTANT SPECIES ===
if "Species" not in df.columns:
    st.error("Species column missing — please re-upload")
    st.stop()

unique_species = sorted(df["Species"].dropna().unique())
constant_species = st.selectbox(
    "Select constant (reference) species for QC",
    options=unique_species,
    index=0
)

df_const = df[df["Species"] == constant_species].copy()
st.write(f"Using **{len(df_const):,}** {constant_species} proteins for analysis")

# === 6 INDIVIDUAL DENSITY PLOTS — log₁₀ INTENSITY ===
st.subheader("1. Individual Intensity Density Plots (log₁₀)")

log10_data = np.log10(df_const[all_reps].replace(0, np.nan))

row1, row2 = st.columns(3), st.columns(3)
for i, rep in enumerate(all_reps):
    col = row1[i] if i < 3 else row2[i-3]
    with col:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=log10_data[rep].dropna(),
            nbinsx=80,
            histnorm="density",
            name=rep,
            marker_color="#E71316" if rep in c1 else "#1f77b4",
            opacity=0.75
        ))
        median = log10_data[rep].median()
        fig.add_vline(x=median, line_dash="dash", line_color="black")
        fig.update_layout(
            height=380,
            title=f"<b>{rep}</b>",
            xaxis_title="log₁₀(Intensity)",
            yaxis_title="Density",
            showlegend=False,
            plot_bgcolor="white"
        )
        st.plotly_chart(fig, use_container_width=True)

# === KS TEST ON RAW DATA ===
st.subheader("2. Technical Reproducibility (KS Test on Raw Data)")

raw_data = df_const[all_reps].replace(0, np.nan)
significant_pairs = []
for r1, r2 in combinations(all_reps, 2):
    d1, d2 = raw_data[r1].dropna(), raw_data[r2].dropna()
    if len(d1) > 1 and len(d2) > 1:
        _, p = stats.ks_2samp(d1, d2)
        if p < 0.05:
            significant_pairs.append(f"{r1} vs {r2}")

if significant_pairs:
    st.warning(f"**Technical differences detected** (raw data):\n" + " • ".join(significant_pairs))
else:
    st.success("**Excellent technical reproducibility** — all raw distributions similar")

# === NORMALITY TEST ON log₁₀ DATA ===
st.subheader("3. Normality Test (log₁₀ Intensities)")

normality_results = []
for rep in all_reps:
    vals = log10_data[rep].dropna()
    if len(vals) < 8:
        normality_results.append({"Replicate": rep, "Shapiro-Wilk p": "N/A", "Normal?": "Too few values"})
        continue
    _, p = stats.shapiro(vals)
    normal = "Yes" if p > 0.05 else "No"
    normality_results.append({
        "Replicate": rep,
        "Shapiro-Wilk p": f"{p:.2e}",
        "Normal?": normal
    })

st.dataframe(pd.DataFrame(normality_results), use_container_width=True)

# === FINAL RECOMMENDATION ===
non_normal_count = sum(1 for r in normality_results if r.get("Normal?") == "No")
if non_normal_count == 0:
    st.success("**log₁₀ intensities are approximately normal** — parametric tests acceptable")
elif non_normal_count <= 2:
    st.info("**log₁₀ is good** — mild non-normality in few replicates. Parametric OK with caution")
else:
    st.warning("**log₁₀ still non-normal** — consider robust methods or rank-based tests")

st.success("**Recommended downstream transformation: log₁₀**")
st.info("→ Use `np.log10(intensity)` in all statistical tests and visualizations")

if st.button("Go to Differential Analysis", type="primary", use_container_width=True):
    st.switch_page("pages/4_Differential_Analysis.py")
