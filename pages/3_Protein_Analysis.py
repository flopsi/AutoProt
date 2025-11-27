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

# === 1. PLOT FILTER: Low intensity (for visualization only) ===
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

# === 2. PRE-CALCULATE LOG10 FOR PLOTS ===
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

current_data = st.session_state.log10_plot_cache[selected_species]

# === 4. 6 LOG10 DENSITY PLOTS ===
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
            name=rep,
            marker_color="#E71316" if rep in c1 else "#1f77b4",
            opacity=0.75
        ))
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

        # Table under each plot
        table_data = []
        for sp in ["All proteins", "HUMAN", "ECOLI", "YEAST"]:
            sp_data = st.session_state.log10_plot_cache[sp]
            sp_vals = sp_data[rep].dropna()
            if len(sp_vals) == 0:
                mean_str = variance_str = std_str = "—"
            else:
                mean_str = f"{sp_vals.mean():.3f}"
                variance_str = f"{sp_vals.var():.3f}"
                std_str = f"{sp_vals.std():.3f}"
            table_data.append({
                "Species": sp,
                "Mean": mean_str,
                "Variance": variance_str,
                "Std Dev": std_str
            })
        st.table(pd.DataFrame(table_data).set_index("Species"))

# === 5. FINAL FILTER STRATEGY (Schessner et al., 2022 compliant) ===
st.subheader("Final Filter Strategy (for Downstream Analysis)")
filter_strategy = st.radio(
    "Choose filtering strategy:",
    ["Raw data",
     "Low intensity filtered",
     "±2σ filtered (on raw data)",
     "Combined (low intensity → recalculate mean/std → ±2σ)"],
    index=0
)

# === 6. DYNAMIC COUNT TABLE ===
st.subheader("Protein Counts After Final Filter")

df_final = df.copy()
log10_full = np.log10(df[all_reps].replace(0, np.nan))

if filter_strategy == "Low intensity filtered":
    mask = pd.Series(True, index=df.index)
    for rep in all_reps:
        mask &= (log10_full[rep] >= 0.5)
    df_final = df[mask]

elif filter_strategy == "±2σ filtered (on raw data)":
    mask = pd.Series(True, index=df.index)
    for rep in all_reps:
        vals = log10_full[rep].dropna()
        if len(vals) == 0: continue
        mean = vals.mean()
        std = vals.std()
        mask &= (log10_full[rep] >= mean - 2*std) & (log10_full[rep] <= mean + 2*std)
    df_final = df[mask]

elif filter_strategy == "Combined (low intensity → recalculate mean/std → ±2σ)":
    # Step 1: low intensity
    mask = pd.Series(True, index=df.index)
    for rep in all_reps:
        mask &= (log10_full[rep] >= 0.5)
    df_step
