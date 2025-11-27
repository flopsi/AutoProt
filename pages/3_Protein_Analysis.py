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

# === SPECIES SELECTION (DEFAULT = MOST COMMON) ===
if "Species" not in df.columns:
    st.error("Species column missing — please re-upload")
    st.stop()

species_counts = df["Species"].value_counts()
most_common_species = species_counts.index[0]

selected_species = st.selectbox(
    "Select species for QC plots and testing",
    options=species_counts.index.tolist(),
    index=0,
    format_func=lambda x: f"{x} ({species_counts[x]:,} proteins)"
)

df_species = df[df["Species"] == selected_species].copy()
st.write(f"Using **{len(df_species):,}** {selected_species} proteins for analysis")

# === 1. 6 INDIVIDUAL DENSITY PLOTS — log₁₀ INTENSITY ===
st.subheader("1. Intensity Density Plots (log₁₀) — Selected Species")

log10_data = np.log10(df_species[all_reps].replace(0, np.nan))

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

# === 2. KS TEST ON RAW DATA ===
st.subheader("2. Technical Reproducibility (KS Test — Raw Intensity)")

raw_data = df_species[all_reps].replace(0, np.nan)
significant_pairs = []
for r1, r2 in combinations(all_reps, 2):
    d1, d2 = raw_data[r1].dropna(), raw_data[r2].dropna()
    if len(d1) > 1 and len(d2) > 1:
        _, p = stats.ks_2samp(d1, d2)
        if p < 0.05:
            significant_pairs.append(f"{r1} vs {r2}")

if significant_pairs:
    st.warning(f"**Significant differences detected**:\n" + " • ".join(significant_pairs))
else:
    st.success("**Excellent technical reproducibility** — all replicates similar")

# === 3. NORMALITY TEST ON RAW INTENSITY ===
st.subheader("3. Normality Test (Raw Intensity — Before Transformation)")

normality_results = []
for rep in all_reps:
    vals = raw_data[rep].dropna()
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

# === 4. TRANSFORMATION RECOMMENDATION ===
non_normal_count = sum(1 for r in normality_results if r.get("Normal?") == "No")

if non_normal_count == 0:
    st.success("**Raw intensities appear normal** — parametric tests acceptable (very rare!)")
elif non_normal_count <= 2:
    st.info("**Mild non-normality** — log₁₀ transformation recommended")
else:
    st.warning("**Strong non-normality** — log₁₀ transformation REQUIRED")

st.success("**Recommended downstream transformation: log₁₀**")
st.info("""
**Why log₁₀?**  
- Standard practice in proteomics (Schessner et al., 2022)  
- Compresses dynamic range  
- Stabilizes variance  
- Makes distributions more symmetric  
→ Use `np.log10(intensity)` in all downstream analysis (volcano, PCA, t-tests)
""")

# === SAVE FOR DOWNSTREAM ===
st.session_state.intensity_raw = raw_data
st.session_state.intensity_log10 = log10_data
st.session_state.selected_species_for_qc = selected_species

st.success("Data ready for differential analysis")

if st.button("Go to Differential Analysis", type="primary", use_container_width=True):
    st.switch_page("pages/4_Differential_Analysis.py")
