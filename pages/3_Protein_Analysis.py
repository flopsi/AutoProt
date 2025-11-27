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

st.title("Protein-Level QC & Outlier Filtering")

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

# === 1. 6 LOG10 DENSITY PLOTS WITH ±2σ BOX ===
st.subheader("1. Intensity Density Plots (log₁₀) with ±2σ Bounds")

raw_data = df_species[all_reps].replace(0, np.nan)
log10_data = np.log10(raw_data)

# Store bounds
bounds = {}
for rep in all_reps:
    vals = log10_data[rep].dropna()
    mean = vals.mean()
    std = vals.std()
    bounds[rep] = (mean - 2*std, mean + 2*std)

row1, row2 = st.columns(3), st.columns(3)
for i, rep in enumerate(all_reps):
    col = row1[i] if i < 3 else row2[i-3]
    with col:
        vals = log10_data[rep].dropna()
        lower, upper = bounds[rep]
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=vals, nbinsx=80, histnorm="density", name=rep,
                                 marker_color="#E71316" if rep in c1 else "#1f77b4", opacity=0.75))
        fig.add_vrect(x0=lower, x1=upper, fillcolor="green", opacity=0.2, line_width=0)
        fig.add_vline(x=vals.mean(), line_dash="dash", line_color="black")
        fig.add_vline(x=lower, line_color="red", line_dash="dot")
        fig.add_vline(x=upper, line_color="red", line_dash="dot")
        fig.update_layout(
            height=380,
            title=f"<b>{rep}</b>",
            xaxis_title="log₁₀(Intensity)",
            yaxis_title="Density",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# === 2. USER DECIDES FILTERING ===
st.subheader("2. Outlier Filtering")
st.info("**Green area** = ±2 standard deviations from mean in each replicate")

filter_2sd = st.checkbox(
    "Keep only proteins within ±2σ in ALL replicates?",
    value=False
)

if filter_2sd:
    # Build mask correctly — avoid index alignment issues
    mask = pd.Series(True, index=df_species.index)
    for rep in all_reps:
        lower, upper = bounds[rep]
        mask &= (log10_data[rep] >= lower) & (log10_data[rep] <= upper)
    df_filtered = df_species[mask].copy()
    removed = len(df_species) - len(df_filtered)
    st.write(f"**Filtered**: {len(df_filtered):,} proteins kept ({removed} removed = {removed/len(df_species)*100:.1f}%)")
else:
    df_filtered = df_species.copy()
    st.write("**No filtering** — all proteins kept")

# === 3. NORMALITY TEST ON RAW (FILTERED) ===
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

# === 4. TRANSFORMATION ===
st.markdown("### 4. Apply Transformation")
transform = st.selectbox(
    "Choose transformation",
    ["log₂", "log₁₀", "Yeo-Johnson", "None"],
    index=1
)

# Apply
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

# === 5. ACCEPT ===
st.markdown("### 5. Confirm Transformation")
if st.button("Accept This Transformation", type="primary"):
    st.session_state.intensity_transformed = transformed
    st.session_state.df_filtered = df_filtered
    st.session_state.qc_accepted = True
    st.success("**Accepted** — ready for next step")

if st.session_state.get("qc_accepted", False):
    if st.button("Go to Differential Analysis", type="primary", use_container_width=True):
        st.switch_page("pages/4_Differential_Analysis.py")
