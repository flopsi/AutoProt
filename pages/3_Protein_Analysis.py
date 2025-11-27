# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

# Load data
if "prot_final_df" not in st.session_state:
    st.error("No protein data found! Please go to Protein Import first.")
    st.stop()

df = st.session_state.prot_final_df
c1 = st.session_state.prot_final_c1
c2 = st.session_state.prot_final_c2
all_reps = c1 + c2

st.title("Protein-Level QC & Manual Intensity Filtering")

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

# === 1. MANUAL LOG10 INTENSITY SLIDER — ABOVE PLOTS ===
st.subheader("1. Manual log₁₀ Intensity Range Selection")

raw_data = df_species[all_reps].replace(0, np.nan)
log10_data = np.log10(raw_data)

global_min = log10_data.min().min()
global_max = log10_data.max().max()

# Default: cut off the low-intensity noise you observed
default_min = 1.5
default_max = global_max

min_val, max_val = st.slider(
    "Select log₁₀ intensity range to keep (per replicate)",
    min_value=float(global_min),
    max_value=float(global_max),
    value=(default_min, float(global_max)),
    step=0.05,
    help="Remove low-intensity noise and failed quantifications"
)

st.info(f"**Keeping proteins with log₁₀ intensity between {min_val:.2f} and {max_val:.2f}** in ALL replicates")

# === 2. 6 LOG10 DENSITY PLOTS WITH WHITE SHADOW OVERLAY ===
st.subheader("2. Intensity Density Plots (log₁₀) — Selected Range Highlighted")

row1, row2 = st.columns(3), st.columns(3)
for i, rep in enumerate(all_reps):
    col = row1[i] if i < 3 else row2[i-3]
    with col:
        vals = log10_data[rep].dropna()
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=vals,
            nbinsx=80,
            histnorm="density",
            name=rep,
            marker_color="#E71316" if rep in c1 else "#1f77b4",
            opacity=0.75
        ))
        
        # WHITE SHADOW OVERLAY — exactly as requested
        fig.add_vrect(
            x0=min_val, x1=max_val,
            fillcolor="white", opacity=0.4,
            line_width=2, line_color="black"
        )
        
        fig.update_layout(
            height=380,
            title=f"<b>{rep}</b>",
            xaxis_title="log₁₀(Intensity)",
            yaxis_title="Density",
            showlegend=False,
            plot_bgcolor="white"
        )
        st.plotly_chart(fig, use_container_width=True)

# === 3. APPLY FILTER ===
filter_manual = st.checkbox("Apply this manual intensity filter?", value=True)

if filter_manual:
    mask = pd.Series(True, index=df_species.index)
    for rep in all_reps:
        mask &= (log10_data[rep] >= min_val) & (log10_data[rep] <= max_val)
    df_filtered = df_species[mask].copy()
    retained_pct = len(df_filtered) / len(df_species) * 100
    st.write(f"**Retained**: {len(df_filtered):,} proteins ({retained_pct:.1f}%)")
else:
    df_filtered = df_species.copy()

# === 4. NORMALITY & TRANSFORMATION ===
st.markdown("### 4. Apply Transformation")
transform = st.selectbox(
    "Choose transformation",
    ["log₂", "log₁₀", "Yeo-Johnson", "None"],
    index=1
)

if transform == "log₂":
    transformed = np.log2(df_filtered[all_reps].replace(0, np.nan))
elif transform == "log₁₀":
    transformed = np.log10(df_filtered[all_reps].replace(0, np.nan))
elif transform == "Yeo-Johnson":
    from scipy.stats import yeojohnson
    transformed = pd.DataFrame(yeojohnson(df_filtered[all_reps].values.flatten())[0].reshape(df_filtered[all_reps].shape),
                               index=df_filtered.index, columns=all_reps)
else:
    transformed = df_filtered[all_reps].replace(0, np.nan)

# === 5. POST-TRANSFORMATION TABLE ===
st.markdown("### 5. After Transformation")
post_stats = []
for rep in all_reps:
    vals = transformed[rep].dropna()
    if len(vals) < 8: continue
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
st.markdown("### 6. Confirm Setup")
if st.button("Accept This Filtering & Transformation", type="primary"):
    st.session_state.intensity_transformed = transformed
    st.session_state.df_filtered = df_filtered
    st.session_state.qc_accepted = True
    st.success("**Accepted** — ready for analysis")
