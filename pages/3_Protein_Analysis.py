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

# === 1. SPECIES VIEW MODE ===
st.subheader("1. Select View Mode")

view_mode = st.radio(
    "Choose how to display data",
    ["One species (for QC)", "All species together (overview)"],
    index=0
)

# === 2. SPECIES SELECTION (only if one species) ===
if view_mode == "One species (for QC)":
    if "Species" not in df.columns:
        st.error("Species column missing — please re-upload")
        st.stop()
    species_counts = df["Species"].value_counts()
    selected_species = st.selectbox(
        "Select species",
        options=species_counts.index.tolist(),
        index=0,
        format_func=lambda x: f"{x} ({species_counts[x]:,} proteins)"
    )
    df_plot = df[df["Species"] == selected_species].copy()
    st.write(f"Showing **{len(df_plot):,}** {selected_species} proteins")
else:
    df_plot = df.copy()
    st.write(f"Showing **all {len(df_plot):,}** proteins from all species")

# === 3. 6 LOG10 DENSITY PLOTS ===
st.subheader("2. Intensity Density Plots (log₁₀)")

raw_data = df_plot[all_reps].replace(0, np.nan)
log10_data = np.log10(raw_data)

sigma_factor = st.slider(
    "Dynamic confidence interval (±σ)",
    min_value=1.0,
    max_value=3.0,
    value=2.0,
    step=0.1
)

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
        fig.add_vrect(x0=lower, x1=upper, fillcolor="white", opacity=0.35, line_width=2)
        fig.add_vline(x=mean, line_dash="dash")
        fig.update_layout(
            height=380,
            title=f"<b>{rep}</b><br>±{sigma_factor:.1f}σ",
            xaxis_title="log₁₀(Intensity)",
            yaxis_title="Density",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# === 4. FILTERING ===
st.subheader("3. Outlier Filtering")
filter_apply = st.checkbox(f"Apply ±{sigma_factor:.1f}σ filter?", value=False)

if filter_apply:
    mask = pd.Series(True, index=df.index)
    for rep in all_reps:
        lower, upper = bounds[rep]
        mask &= (np.log10(df[rep].replace(0, np.nan)) >= lower) & (np.log10(df[rep].replace(0, np.nan)) <= upper)
    df_filtered = df[mask].copy()
    st.write(f"**Retained**: {len(df_filtered):,} proteins ({len(df_filtered)/len(df)*100:.1f}%)")
else:
    df_filtered = df.copy()

# === 5. NORMALITY & TRANSFORMATION ===
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
    transformed = pd.DataFrame(yeojohnson(df_filtered[all_reps].values.flatten())[0].reshape(df_filtered[all_reps].shape),
                               index=df_filtered.index, columns=all_reps)
else:
    transformed = df_filtered[all_reps].replace(0, np.nan)

# === 6. POST-TRANSFORMATION TABLE ===
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

# === 7. ACCEPT ===
st.markdown("### 6. Confirm Transformation")
if st.button("Accept This Transformation & Filtering", type="primary"):
    st.session_state.intensity_transformed = transformed
    st.session_state.df_filtered = df_filtered
    st.session_state.qc_accepted = True
    st.success("**Accepted** — ready for analysis")
