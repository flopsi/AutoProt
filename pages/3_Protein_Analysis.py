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
selected_species = st.selectbox(
    "Select species for QC",
    options=species_counts.index.tolist(),
    index=0,
    format_func=lambda x: f"{x} ({species_counts[x]:,} proteins)"
)

df_species = df[df["Species"] == selected_species].copy()

# === 1. 6 LOG10 DENSITY PLOTS + DYNAMIC σ SLIDER ===
st.subheader("1. Intensity Density Plots (log₁₀) with Dynamic ±σ Bounds")

sigma_factor = st.slider(
    "Select confidence interval (±σ)",
    min_value=1.0,
    max_value=3.0,
    value=2.0,
    step=0.1,
    help="2.0σ ≈ 95%, 3.0σ ≈ 99.7% of normal data"
)

raw_data = df_species[all_reps].replace(0, np.nan)
log10_data = np.log10(raw_data)

row1, row2 = st.columns(3), st.columns(3)

bounds = {}
for i, rep in enumerate(all_reps):
    col = row1[i] if i < 3 else row2[i-3]
    with col:
        vals = log10_data[rep].dropna()
        mean = vals.mean()
        std = vals.std()
        lower = mean - sigma_factor * std
        upper = mean + sigma_factor * std
        bounds[rep] = (lower, upper)

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=vals,
            nbinsx=80,
            histnorm="density",
            name=rep,
            marker_color="#E71316" if rep in c1 else "#1f77b4",
            opacity=0.75
        ))
        # White background box
        fig.add_vrect(x0=lower, x1=upper, fillcolor="white", opacity=0.35, line_width=2, line_color="black")
        fig.add_vline(x=mean, line_dash="dash", line_color="black")
        
        fig.update_layout(
            height=380,
            title=f"<b>{rep}</b><br>±{sigma_factor:.1f}σ",
            xaxis_title="log₁₀(Intensity)",
            yaxis_title="Density",
            showlegend=False,
            plot_bgcolor="white"
        )
        st.plotly_chart(fig, use_container_width=True)

# === 2. FILTERING OPTION ===
st.subheader("2. Outlier Filtering")
filter_sigma = st.checkbox(
    f"Keep only proteins within ±{sigma_factor:.1f}σ in ALL replicates?",
    value=False
)

if filter_sigma:
    mask = pd.Series(True, index=df_species.index)
    for rep in all_reps:
        lower, upper = bounds[rep]
        mask &= (log10_data[rep] >= lower) & (log10_data[rep] <= upper)
    df_filtered = df_species[mask].copy()
    st.write(f"**Retained**: {len(df_filtered):,} proteins ({len(df_filtered)/len(df_species)*100:.1f}%)")
else:
    df_filtered = df_species.copy()

# === 3. NORMALITY ON RAW ===
st.subheader("3. Raw Data Diagnosis")
stats_raw = []
for rep in all_reps:
    vals = df_filtered[rep].replace(0, np.nan).dropna()
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

# === 4. TRANSFORMATION SELECTION ===
st.markdown("### 4. Apply Transformation")
transform = st.selectbox(
    "Choose transformation",
    ["log₂", "log₁₀", "Box-Cox", "Yeo-Johnson", "None"],
    index=1
)

# Apply
if transform == "log₂":
    transformed = np.log2(df_filtered[all_reps].replace(0, np.nan))
elif transform == "log₁₀":
    transformed = np.log10(df_filtered[all_reps].replace(0, np.nan))
elif transform == "Box-Cox":
    shifted = df_filtered[all_reps] + 1
    transformed = pd.DataFrame(boxcox(shifted.values.flatten())[0].reshape(shifted.shape),
                               index=shifted.index, columns=shifted.columns)
elif transform == "Yeo-Johnson":
    transformed = pd.DataFrame(yeojohnson(df_filtered[all_reps].values.flatten())[0].reshape(df_filtered[all_reps].shape),
                               index=df_filtered.index, columns=all_reps)
else:
    transformed = df_filtered[all_reps].replace(0, np.nan)

# === 5. POST-TRANSFORMATION TABLE (LIVE) ===
st.markdown("### 5. After Transformation")
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

# === 6. ACCEPT ===
st.markdown("### 6. Confirm Transformation")
if st.button("Accept This Transformation", type="primary"):
    st.session_state.intensity_transformed = transformed
    st.session_state.df_filtered = df_filtered
    st.session_state.qc_accepted = True
    st.success("**Accepted** — ready for next step")

if st.session_state.get("qc_accepted", False):
    if st.button("Go to Differential Analysis", type="primary", use_container_width=True):
        st.switch_page("pages/4_Differential_Analysis.py")
