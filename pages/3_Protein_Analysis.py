# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import boxcox, yeojohnson
from itertools import combinations

# Load data
if "prot_final_df" not in st.session_state:
    st.error("No protein data found! Please go to Protein Import first.")
    st.stop()

df = st.session_state.prot_final_df
c1 = st.session_state.prot_final_c1
c2 = st.session_state.prot_final_c2
all_reps = c1 + c2

st.title("Protein-Level QC & Transformation")

# === SPECIES SELECTION ===
if "Species" not in df.columns:
    st.error("Species column missing — please re-upload")
    st.stop()

species_counts = df["Species"].value_counts()
selected_species = st.selectbox(
    "Select species for QC",
    options=species_counts.index.tolist(),
    index=0,
    format_func=lambda x: f"{x} ({species_counts[x]:,} proteins)"
)

df_species = df[df["Species"] == selected_species].copy()

# === 1. 6 LOG10 DENSITY PLOTS ===
st.subheader("1. Intensity Density Plots (log₁₀)")

intensity_data = df_species[all_reps]  # already 0/NaN → 1.0
log10_data = np.log10(intensity_data)

row1, row2 = st.columns(3), st.columns(3)
for i, rep in enumerate(all_reps):
    col = row1[i] if i < 3 else row2[i-3]
    with col:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=log10_data[rep],
            nbinsx=80,
            histnorm="density",
            name=rep,
            marker_color="#E71316" if rep in c1 else "#1f77b4",
            opacity=0.75
        ))
        fig.add_vline(x=log10_data[rep].median(), line_dash="dash")
        fig.update_layout(
            height=380,
            title=f"<b>{rep}</b>",
            xaxis_title="log₁₀(Intensity)",
            yaxis_title="Density",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# === 2. KS TEST ===
st.subheader("2. Technical Reproducibility (KS Test)")
significant = []
for r1, r2 in combinations(all_reps, 2):
    _, p = stats.ks_2samp(intensity_data[r1], intensity_data[r2])
    if p < 0.05:
        significant.append(f"{r1} vs {r2}")

if significant:
    st.warning(f"**Differences detected**:\n" + " • ".join(significant))
else:
    st.success("**All replicates similar**")

# === 3. NORMALITY ON RAW ===
st.subheader("3. Raw Data Diagnosis")
stats_raw = []
for rep in all_reps:
    vals = intensity_data[rep]
    skew = stats.skew(vals)
    kurt = stats.kurtosis(vals)
    _, p = stats.shapiro(vals)
    normal = "Yes" if p > 0.05 else "No"
    stats_raw.append({
        "Replicate": rep,
        "Skew": f"{skew:+.3f}",
        "Kurtosis": f"{kurt:+.3f}",
        "Shapiro p": f"{p:.2e}",
        "Normal": normal
    })
st.dataframe(pd.DataFrame(stats_raw), use_container_width=True)

# === 4. TRANSFORMATION ===
st.markdown("### 4. Apply Transformation")
transform = st.selectbox(
    "Choose transformation",
    ["log₂", "log₁₀", "Box-Cox", "Yeo-Johnson", "None"],
    index=1
)

if transform == "log₂":
    transformed = np.log2(intensity_data)
elif transform == "log₁₀":
    transformed = np.log10(intensity_data)
elif transform == "Box-Cox":
    transformed = pd.DataFrame(boxcox(intensity_data.values.flatten())[0].reshape(intensity_data.shape),
                               index=intensity_data.index, columns=intensity_data.columns)
elif transform == "Yeo-Johnson":
    transformed = pd.DataFrame(yeojohnson(intensity_data.values.flatten())[0].reshape(intensity_data.shape),
                               index=intensity_data.index, columns=intensity_data.columns)
else:
    transformed = intensity_data

# === 5. POST-TRANSFORMATION TABLE (LIVE) ===
st.markdown("### 5. After Transformation")
post_stats = []
for rep in all_reps:
    vals = transformed[rep]
    skew = stats.skew(vals)
    kurt = stats.kurtosis(vals)
    _, p = stats.shapiro(vals)
    normal = "Yes" if p > 0.05 else "No"
    post_stats.append({
        "Replicate": rep,
        "Skew": f"{skew:+.3f}",
        "Kurtosis": f"{kurt:+.3f}",
        "Shapiro p": f"{p:.2e}",
        "Normal": normal
    })
st.dataframe(pd.DataFrame(post_stats), use_container_width=True)

# === 6. ACCEPT ===
st.markdown("### 6. Confirm Transformation")
if st.button("Accept This Transformation", type="primary"):
    st.session_state.intensity_transformed = transformed
    st.session_state.transform_applied = transform
    st.session_state.qc_accepted = True
    st.success(f"**{transform} accepted**")

st.info("More plots coming soon...")
