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

st.title("Protein-Level QC & Advanced Filtering")

# === 1. PLOT MODE: ALL PROTEINS OR INDIVIDUAL SPECIES ===
st.subheader("1. Select Plot Mode")
plot_mode = st.selectbox(
    "Show density plots for:",
    ["All proteins together", "Each species individually"],
    index=0
)

# === 2. 6 LOG10 DENSITY PLOTS ===
st.subheader("2. Intensity Density Plots (log₁₀)")

raw_data = df[all_reps].replace(0, np.nan)
log10_data = np.log10(raw_data)

# Prepare species-specific data if needed
if plot_mode == "Each species individually" and "Species" in df.columns:
    species_list = df["Species"].dropna().unique()
    species_data = {sp: df[df["Species"] == sp][all_reps].replace(0, np.nan) for sp in species_list}
else:
    species_data = {"All Proteins": raw_data}

row1, row2 = st.columns(3), st.columns(3)
for i, rep in enumerate(all_reps):
    col = row1[i] if i < 3 else row2[i-3]
    with col:
        fig = go.Figure()
        
        if plot_mode == "Each species individually" and "Species" in df.columns:
            for sp, data in species_data.items():
                vals = np.log10(data[rep].dropna())
                if len(vals) == 0: continue
                fig.add_trace(go.Histogram(
                    x=vals,
                    name=sp,
                    histnorm="density",
                    opacity=0.6,
                    nbinsx=80
                ))
                mean = vals.mean()
                std = vals.std()
                fig.add_vline(x=mean, line_dash="dash", line_color="black")
                fig.add_vrect(x0=mean-2*std, x1=mean+2*std, fillcolor="white", opacity=0.3, line_width=1)
        else:
            vals = log10_data[rep].dropna()
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
            showlegend=(plot_mode == "Each species individually"),
            legend_title="Species" if plot_mode == "Each species individually" else None
        )
        st.plotly_chart(fig, use_container_width=True)

# === 3. FILTERING: INTENSITY OR STDEV (MUTUALLY EXCLUSIVE) ===
st.subheader("3. Filtering Options (Choose One)")

col_left, col_right = st.columns(2)

intensity_filter = False
stdev_filter = False

with col_left:
    intensity_filter = st.checkbox("Intensity-based filter (manual range)", value=False)

with col_right:
    stdev_filter = st.checkbox("StDev-based filter (±σ)", value=False)

# Enforce mutual exclusion
if intensity_filter and stdev_filter:
    st.error("Please select only one filtering method")
    st.stop()

# === INTENSITY-BASED FILTER ===
if intensity_filter:
    st.markdown("**Intensity Range Selection**")
    min_val, max_val = st.slider(
        "log₁₀ intensity range",
        min_value=float(log10_data.min().min()),
        max_value=float(log10_data.max().max()),
        value=(1.5, float(log10_data.max().max())),
        step=0.1
    )
    mask = pd.Series(True, index=df.index)
    for rep in all_reps:
        mask &= (log10_data[rep] >= min_val) & (log10_data[rep] <= max_val)
    df_filtered = df[mask].copy()

# === STDEV-BASED FILTER ===
elif stdev_filter:
    st.markdown("**StDev Filter Configuration**")
    stdev_scope = st.radio("Apply ±σ filter:", ["Global (all data)", "Per species"], index=1)
    
    if stdev_scope == "Per species" and "Species" in df.columns:
        species_bounds = {}
        for sp in df["Species"].dropna().unique():
            sp_data = df[df["Species"] == sp][all_reps].replace(0, np.nan)
            sp_log10 = np.log10(sp_data)
            mean = sp_log10.mean().mean()
            std = sp_log10.stack().std()
            species_bounds[sp] = (mean - 2*std, mean + 2*std)
        
        selected_sp = st.selectbox("Select species for StDev bounds", options=list(species_bounds.keys()))
        lower, upper = species_bounds[selected_sp]
    else:
        mean = log10_data.mean().mean()
        std = log10_data.stack().std()
        lower, upper = mean - 2*std, mean + 2*std
    
    st.info(f"Using ±2σ bounds: [{lower:.2f}, {upper:.2f}]")
    
    mask = pd.Series(True, index=df.index)
    for rep in all_reps:
        mask &= (log10_data[rep] >= lower) & (log10_data[rep] <= upper)
    df_filtered = df[mask].copy()

else:
    df_filtered = df.copy()

st.write(f"**Final dataset**: {len(df_filtered):,} proteins retained")
