# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import boxcox, yeojohnson

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
most_common = species_counts.index[0]

selected_species = st.selectbox(
    "Select species for QC",
    options=species_counts.index.tolist(),
    index=0,
    format_func=lambda x: f"{x} ({species_counts[x]:,} proteins)"
)

df_species = df[df["Species"] == selected_species].copy()
st.write(f"Using **{len(df_species):,}** {selected_species} proteins")

# === 1. 6 INDIVIDUAL DENSITY PLOTS — log₁₀ ===
st.subheader("1. Intensity Density Plots (log₁₀)")

raw_data = df_species[all_reps].replace(0, np.nan)
log10_data = np.log10(raw_data)

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

# === 2. RAW DATA STATISTICS ===
st.subheader("2. Raw Data Distribution Diagnosis")

stats_raw = []
for rep in all_reps:
    vals = raw_data[rep].dropna()
    if len(vals) < 8:
        stats_raw.append({"Replicate": rep, "n": len(vals), "Skew": "N/A", "Kurtosis": "N/A", "Shapiro p": "N/A", "Normal": "N/A"})
        continue
    skew = stats.skew(vals)
    kurt = stats.kurtosis(vals)
    _, p = stats.shapiro(vals)
    normal = "Yes" if p > 0.05 else "No"
    stats_raw.append({
        "Replicate": rep,
        "n": len(vals),
        "Skew": f"{skew:+.3f}",
        "Kurtosis": f"{kurt:+.3f}",
        "Shapiro p": f"{p:.2e}",
        "Normal": normal
    })

st.dataframe(pd.DataFrame(stats_raw), use_container_width=True)

# === 3. TRANSFORMATION SELECTION ===
st.markdown("### 3. Apply Transformation")
transform = st.selectbox(
    "Choose transformation",
    ["log₂", "log₁₀", "Box-Cox", "Yeo-Johnson", "None"],
    index=1
)

# Apply transformation — FIXED Yeo-Johnson & Box-Cox
if transform == "log₂":
    transformed = np.log2(raw_data)
    y_label = "log₂(Intensity)"
elif transform == "log₁₀":
    transformed = np.log10(raw_data)
    y_label = "log₁₀(Intensity)"
elif transform == "Box-Cox":
    # Box-Cox requires strictly positive values
    shifted = raw_data + 1  # simple shift
    transformed = pd.DataFrame(
        boxcox(shifted.values.flatten())[0].reshape(shifted.shape),
        index=raw_data.index, columns=raw_data.columns
    )
    y_label = "Box-Cox(Intensity)"
elif transform == "Yeo-Johnson":
    # Yeo-Johnson handles negative/zero — but must be finite
    clean_data = raw_data.replace([np.inf, -np.inf], np.nan).fillna(0)
    transformed = pd.DataFrame(
        yeojohnson(clean_data.values.flatten())[0].reshape(clean_data.shape),
        index=raw_data.index, columns=raw_data.columns
    )
    y_label = "Yeo-Johnson(Intensity)"
else:
    transformed = raw_data
    y_label = "Raw Intensity"

# === 4. POST-TRANSFORMATION STATISTICS ===
st.markdown("### 4. After Transformation")
post_stats = []
for rep in all_reps:
    vals = transformed[rep].dropna()
    if len(vals) < 8:
        post_stats.append({"Replicate": rep, "Skew": "N/A", "Kurtosis": "N/A", "Shapiro p": "N/A", "Normal": "N/A"})
        continue
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

# === 5. ACCEPTANCE REQUIRED ===
st.markdown("### 5. Confirm Transformation")
non_normal = sum(1 for r in post_stats if r.get("Normal") == "No")

if non_normal == 0:
    st.success("**Perfect — data is normal**")
elif non_normal <= 2:
    st.info("**Good — mild non-normality**")
else:
    st.warning("**Still non-normal — try another transformation**")

if st.button("Accept This Transformation & Proceed", type="primary"):
    st.session_state.intensity_transformed = transformed
    st.session_state.transform_applied = transform
    st.session_state.qc_accepted = True
    st.success(f"**{transform} accepted** — ready for analysis")

# === FINAL BUTTON ===
if st.session_state.get("qc_accepted", False):
    if st.button("Go to Differential Analysis", type="primary", use_container_width=True):
        st.switch_page("pages/4_Differential_Analysis.py")
else:
    st.info("Please accept transformation before proceeding")
