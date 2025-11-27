# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import yeojohnson

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

# === 1. 6 LOG10 DENSITY PLOTS + 2SD BOX ===
st.subheader("1. Intensity Density Plots (log₁₀) with ±2σ Bounds")

raw_data = df_species[all_reps].replace(0, np.nan)
log10_data = np.log10(raw_data)

row1, row2 = st.columns(3), st.columns(3)

# Store bounds for filtering
lower_bounds = {}
upper_bounds = {}

for i, rep in enumerate(all_reps):
    col = row1[i] if i < 3 else row2[i-3]
    with col:
        vals = log10_data[rep].dropna()
        mean_val = vals.mean()
        std_val = vals.std()
        lower = mean_val - 2 * std_val
        upper = mean_val + 2 * std_val
        
        lower_bounds[rep] = lower
        upper_bounds[rep] = upper

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=vals, nbinsx=80, histnorm="density", name=rep,
                                 marker_color="#E71316" if rep in c1 else "#1f77b4", opacity=0.75))
        
        # 2SD box
        fig.add_vrect(x0=lower, x1=upper, fillcolor="green", opacity=0.2, line_width=0)
        fig.add_vline(x=mean_val, line_dash="dash", line_color="black")
        fig.add_vline(x=lower, line_color="red", line_dash="dot")
        fig.add_vline(x=upper, line_color="red", line_dash="dot")
        
        fig.update_layout(
            height=380,
            title=f"<b>{rep}</b><br>±2σ shown",
            xaxis_title="log₁₀(Intensity)",
            yaxis_title="Density",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# === 2. USER DECIDES: FILTER TO 2SD? ===
st.subheader("2. Outlier Filtering")
st.info("**±2σ bounds** shown in green on each plot above")

filter_2sd = st.checkbox(
    "Keep only proteins within ±2 standard deviations in ALL replicates?",
    value=False
)

if filter_2sd:
    mask = pd.Series([True] * len(df_species))
    for rep in all_reps:
        mask &= (log10_data[rep] >= lower_bounds[rep]) & (log10_data[rep] <= upper_bounds[rep])
    df_filtered = df_species[mask].copy()
    st.write(f"**Filtered**: {len(df_filtered):,} proteins kept ({len(df_species)-len(df_filtered)} removed)")
else:
    df_filtered = df_species.copy()
    st.write("**No filtering applied** — all proteins kept")

# === 3. NORMALITY TEST ON RAW (FILTERED DATA) ===
st.subheader("3. Raw Data Distribution Diagnosis")
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

if transform == "log₂":
    transformed = np.log2(df_filtered[all_reps].replace(0, np.nan))
elif transform == "log₁₀":
    transformed = np.log10(df_filtered[all_reps].replace(0, np.nan))
elif transform == "Yeo-Johnson":
    transformed = pd.DataFrame(yeojohnson(df_filtered[all_reps].values.flatten())[0].reshape(df_filtered[all_reps].shape),
                               index=df_filtered.index, columns=all_reps)
else:
    transformed = df_filtered[all_reps].replace(0, np.nan)

# === 5. ACCEPT ===
st.markdown("### 5. Confirm Transformation")
if st.button("Accept This Transformation & Proceed", type="primary"):
    st.session_state.intensity_transformed = transformed
    st.session_state.df_filtered = df_filtered
    st.session_state.qc_accepted = True
    st.success("**Transformation accepted** — ready for analysis")

if st.session_state.get("qc_accepted", False):
    if st.button("Go to Differential Analysis", type="primary", use_container_width=True):
        st.switch_page("pages/4_Differential_Analysis.py")
