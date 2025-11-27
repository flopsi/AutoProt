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

# === 1. SPECIES SELECTION ===
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

# === 2. 6 LOG10 DENSITY PLOTS ===
st.subheader("2. Intensity Density Plots (log₁₀)")

raw_data = df_species[all_reps].replace(0, np.nan)
log10_data = np.log10(raw_data)

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
        fig.update_layout(
            height=380,
            title=f"<b>{rep}</b>",
            xaxis_title="log₁₀(Intensity)",
            yaxis_title="Density",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# === 3. MANUAL LOG10 INTENSITY RANGE SELECTION ===
st.subheader("3. Manual Intensity Range Selection")

# Auto-suggest based on data
global_min = log10_data.min().min()
global_max = log10_data.max().max()
suggested_min = 1.5  # from your observation
suggested_max = global_max

min_val, max_val = st.slider(
    "Select log₁₀ intensity range to keep (per replicate)",
    min_value=float(global_min),
    max_value=float(global_max),
    value=(suggested_min, float(global_max)),
    step=0.05,
    help="Remove low-intensity noise and failed quantifications"
)

st.info(f"You are keeping proteins with **log₁₀ intensity between {min_val:.2f} and {max_val:.2f}** in ALL replicates")

# === 4. APPLY MANUAL FILTER ===
filter_manual = st.checkbox("Apply manual intensity filter?", value=True)

if filter_manual:
    mask = pd.Series(True, index=df_species.index)
    for rep in all_reps:
        mask &= (log10_data[rep] >= min_val) & (log10_data[rep] <= max_val)
    df_filtered = df_species[mask].copy()
    retained = len(df_filtered)
    removed = len(df_species) - retained
    st.write(f"**Retained**: {retained:,} proteins ({retained/len(df_species)*100:.1f}%)")
    st.write(f"**Removed**: {removed:,} low-intensity or noisy proteins")
else:
    df_filtered = df_species.copy()

# === 5. NORMALITY & TRANSFORMATION ===
st.markdown("### 5. Apply Transformation")
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

# === 6. POST-TRANSFORMATION TABLE ===
st.markdown("### 6. After Transformation")
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

# === 7. ACCEPT ===
st.markdown("### 7. Confirm Filtering & Transformation")
if st.button("Accept This Setup", type="primary"):
    st.session_state.intensity_transformed = transformed
    st.session_state.df_filtered = df_filtered
    st.session_state.qc_accepted = True
    st.success("**Accepted** — ready for analysis")
