# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

# === SAFETY CHECK — ROBUST & USER-FRIENDLY ===
required_keys = ["prot_final_df", "prot_final_c1", "prot_final_c2"]
missing = [k for k in required_keys if k not in st.session_state]

if missing or st.session_state.prot_final_df is None or len(st.session_state.prot_final_df) == 0:
    st.error("No protein data found! Please complete **Protein Import** first.")
    if st.button("Go to Protein Import"):
        st.switch_page("pages/1_Protein_Import.py")
    st.stop()

# === LOAD DATA SAFELY ===
df = st.session_state.prot_final_df.copy()
c1 = st.session_state.prot_final_c1
c2 = st.session_state.prot_final_c2
all_reps = c1 + c2

st.title("Protein-Level QC & Replicate Difference Testing")

# === 1. LOW INTENSITY FILTER FOR PLOTS ONLY ===
st.subheader("Plot Filter (Visual QC Only)")
remove_low_plot = st.checkbox(
    "Remove proteins with log₁₀ intensity < 0.5 in ALL replicates (plots only)",
    value=False
)

# Apply to plot data
df_plot = df.copy()
if remove_low_plot:
    mask = pd.Series(True, index=df.index)
    for rep in all_reps:
        mask &= (np.log10(df[rep].replace(0, np.nan)) >= 0.5)
    df_plot = df[mask]

# === 2. PRE-CALCULATE LOG10 FOR PLOTS (CACHED) ===
if "log10_plot_cache" not in st.session_state or st.session_state.get("last_plot_filter") != remove_low_plot:
    raw = df_plot[all_reps].replace(0, np.nan)
    log10_all = np.log10(raw)

    cache = {"All proteins": log10_all}
    if "Species" in df_plot.columns:
        for sp in ["HUMAN", "ECOLI", "YEAST"]:
            subset = df_plot[df_plot["Species"] == sp][all_reps].replace(0, np.nan)
            cache[sp] = np.log10(subset) if len(subset) > 0 else pd.DataFrame()
    
    st.session_state.log10_plot_cache = cache
    st.session_state.last_plot_filter = remove_low_plot

# === 3. SPECIES SELECTION FOR PLOTS ===
st.subheader("Select Species for Plots")
selected_species = st.radio(
    "Show in plots:",
    ["All proteins", "HUMAN", "ECOLI", "YEAST"],
    index=0,
    key="plot_species"
)

current_data = st.session_state.log10_plot_cache.get(selected_species, st.session_state.log10_plot_cache["All proteins"])

# === 4. 6 LOG10 DENSITY PLOTS + TABLE UNDER EACH ===
st.subheader("Intensity Density Plots (log₁₀)")

row1, row2 = st.columns(3), st.columns(3)
for i, rep in enumerate(all_reps):
    col = row1[i] if i < 3 else row2[i-3]
    with col:
        vals = current_data[rep].dropna()
        if len(vals) == 0:
            st.write("No data")
            continue
            
        mean = float(vals.mean())
        std = float(vals.std())
        lower = mean - 2*std
        upper = mean + 2*std

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=vals,
            nbinsx=80,
            histnorm="density",
            name
