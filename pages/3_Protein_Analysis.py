# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Load data
if "prot_final_df" not in st.session_state:
    st.error("No protein data found! Please go to Protein Import first.")
    st.stop()

df = st.session_state.prot_final_df
c1 = st.session_state.prot_final_c1
c2 = st.session_state.prot_final_c2
all_reps = c1 + c2

st.title("Protein-Level QC & Advanced Filtering")

# === 1. SPECIES VIEW DROPDOWN ===
st.subheader("1. Select View Mode")
view_species = st.selectbox(
    "Show density plots for:",
    ["All proteins", "HUMAN", "ECOLI", "YEAST"],
    index=0
)

# Filter data for plotting
if view_species == "All proteins":
    df_plot = df.copy()
else:
    if "Species" not in df.columns:
        st.error("Species column missing")
        st.stop()
    df_plot = df[df["Species"] == view_species].copy()

if len(df_plot) == 0:
    st.warning(f"No proteins found for {view_species}")
    st.stop()

# === 2. 6 LOG10 DENSITY PLOTS WITH ±2σ PER REPLICATE ===
st.subheader("2. Intensity Density Plots (log₁₀)")

raw_plot = df_plot[all_reps].replace(0, np.nan)
log10_plot = np.log10(raw_plot)

row1, row2 = st.columns(3), st.columns(3)
bounds_per_rep = {}

for i, rep in enumerate(all_reps):
    col = row1[i] if i < 3 else row2[i-3]
    with col:
        vals = log10_plot[rep].dropna()
        mean = vals.mean()
        std = vals.std()
        lower = mean - 2*std
        upper = mean + 2*std
        bounds_per_rep[rep] = (lower, upper)

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=vals,
            nbinsx=80,
            histnorm="density",
            name=rep,
            marker_color="#E71316" if rep in c1 else "#1f77b4",
            opacity=0.75
        ))
        # WHITE ±2σ SHADOW
        fig.add_vrect(x0=lower, x1=upper, fillcolor="white", opacity=0.35, line_width=2)
        fig.add_vline(x=mean, line_dash="dash", line_color="black")
        fig.update_layout(
            height=380,
            title=f"<b>{rep}</b>",
            xaxis_title="log₁₀(Intensity)",
            yaxis_title="Density",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# === 3. FILTERING OPTIONS — TWO COLUMNS, MUTUALLY EXCLUSIVE ===
st.subheader("3. Filtering Options (Select One)")

col1, col2 = st.columns(2)

intensity_filter = False
stdev_filter = False

with col1:
    st.markdown("**Intensity-based Filtering**")
    intensity_filter = st.checkbox("Enable intensity filter", key="intensity_cb")
    if intensity_filter:
        lower_input = st.number_input("Lower bound (log₁₀)", value=-1.0, step=0.1, format="%.2f")
        upper_input = st.number_input("Upper bound (log₁₀)", value=5.0, step=0.1, format="%.2f")

with col2:
    st.markdown("**StDev-based Filtering**")
    stdev_filter = st.checkbox("Enable ±σ filter", key="stdev_cb")
    if stdev_scope := None
    if stdev_filter:
        stdev_scope = st.radio("Apply ±σ:", ["Global", "Per species"], index=1)

# Prevent both active
if intensity_filter and stdev_filter:
    st.error("Please select only ONE filtering method")
    st.stop()

# === 4. APPLY FILTERING ===
mask = pd.Series(False, index=df.index)  # start empty

if intensity_filter:
    for rep in all_reps:
        rep_mask = (np.log10(df[rep].replace(0, np.nan)) >= lower_input) & \
                   (np.log10(df[rep].replace(0, np.nan)) <= upper_input)
        mask |= rep_mask  # union across replicates

elif stdev_filter:
    if stdev_scope == "Global":
        all_vals = np.log10(df[all_reps].replace(0, np.nan)).stack()
        mean = all_vals.mean()
        std = all_vals.std()
        lower, upper = mean - 2*std, mean + 2*std
        for rep in all_reps:
            rep_mask = (np.log10(df[rep].replace(0, np.nan)) >= lower) & \
                       (np.log10(df[rep].replace(0, np.nan)) <= upper)
            mask |= rep_mask
    else:
        # Per species
        species_sigma = {}
        for sp in df["Species"].dropna().unique():
            sp_data = df[df["Species"] == sp][all_reps].replace(0, np.nan)
            sp_log = np.log10(sp_data)
            mean = sp_log.mean().mean()
            std = sp_log.stack().std()
            species_sigma[sp] = (mean - 2*std, mean + 2*std)
        
        selected_sp = st.multiselect(
            "Select species to include (with their ±2σ)",
            options=list(species_sigma.keys()),
            default=list(species_sigma.keys())[:1]
        )
        for sp in selected_sp:
            lower, upper = species_sigma[sp]
            sp_mask = df["Species"] == sp
            for rep in all_reps:
                rep_mask = sp_mask & (np.log10(df[rep].replace(0, np.nan)) >= lower) & \
                                      (np.log10(df[rep].replace(0, np.nan)) <= upper)
                mask |= rep_mask

else:
    mask = pd.Series(True, index=df.index)  # no filter

df_filtered = df[mask].copy()
st.write(f"**Final dataset**: {len(df_filtered):,} proteins retained")

# === 5. TRANSFORMATION & ACCEPT ===
st.markdown("### 5. Apply Transformation")
transform = st.selectbox("Choose transformation", ["log₂", "log₁₀", "Yeo-Johnson", "None"], index=1)

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

st.markdown("### 6. Confirm Setup")
if st.button("Accept This Filtering & Transformation", type="primary"):
    st.session_state.intensity_transformed = transformed
    st.session_state.df_filtered = df_filtered
    st.session_state.qc_accepted = True
    st.success("Accepted — ready for analysis")

if st.session_state.get("qc_accepted", False):
    if st.button("Go to Differential Analysis", type="primary", use_container_width=True):
        st.switch_page("pages/4_Differential_Analysis.py")
