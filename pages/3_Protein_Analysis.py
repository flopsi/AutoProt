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

# === 1. SPECIES SELECTION FOR PLOTTING (independent) ===
st.subheader("Select Species for Visualization")
selected_species = st.radio(
    "Show in plots:",
    ["All proteins", "HUMAN", "ECOLI", "YEAST"],
    index=0,
    key="plot_species"
)

# === 2. PRE-CALCULATE LOG10 DATA FOR PLOTTING ===
if "log10_cache" not in st.session_state:
    raw = df[all_reps].replace(0, np.nan)
    log10_all = np.log10(raw)

    cache = {"All proteins": log10_all}
    if "Species" in df.columns:
        for sp in ["HUMAN", "ECOLI", "YEAST"]:
            subset = df[df["Species"] == sp][all_reps].replace(0, np.nan)
            cache[sp] = np.log10(subset) if len(subset) > 0 else pd.DataFrame()
    
    st.session_state.log10_cache = cache

current_data = st.session_state.log10_cache[selected_species]

# === 3. 6 LOG10 DENSITY PLOTS ===
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
            title=f"<b>{rep}</b><br>{selected_species}",
            xaxis_title="log₁₀(Intensity)",
            yaxis_title="Density",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table under each plot
        table_data = []
        for sp in ["All proteins", "HUMAN", "ECOLI", "YEAST"]:
            sp_data = st.session_state.log10_cache[sp]
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

# === 4. GLOBAL FILTERING (independent of plots) ===
st.subheader("Global Filtering Strategy (Applied to Full Dataset)")

remove_low_intensity = st.checkbox(
    "Remove proteins with log₁₀ intensity < 0.5 in ALL replicates",
    value=False
)

filter_type = st.radio(
    "Choose filtering strategy:",
    ["Raw data", "Low intensity filtered", "±2σ filtered (after low intensity)", "Combined"],
    index=0
)

# Apply filtering
df_final = df.copy()
log10_full = np.log10(df[all_reps].replace(0, np.nan))

if remove_low_intensity:
    mask = pd.Series(True, index=df.index)
    for rep in all_reps:
        mask &= (log10_full[rep] >= 0.5)
    df_final = df[mask]
    log10_full = log10_full.loc[mask]

if filter_type == "Raw data":
    final_count = len(df)
elif filter_type == "Low intensity filtered":
    final_count = len(df_final)
else:
    # Apply ±2σ on current df_final
    mask = pd.Series(True, index=df_final.index)
    for rep in all_reps:
        vals = log10_full[rep].dropna()
        if len(vals) == 0: continue
        mean = vals.mean()
        std = vals.std()
        mask &= (log10_full[rep] >= mean - 2*std) & (log10_full[rep] <= mean + 2*std)
    df_final = df_final[mask]
    final_count = len(df_final)

# Dynamic count table
count_data = []
for sp in ["All proteins", "HUMAN", "ECOLI", "YEAST"]:
    base = len(df[df["Species"] == sp]) if sp != "All proteins" and "Species" in df.columns else len(df)
    filtered = len(df_final[df_final["Species"] == sp]) if sp != "All proteins" and "Species" in df_final.columns else len(df_final)
    count_data.append({"Species": sp, "Unfiltered": base, "After Filter": filtered})

st.table(pd.DataFrame(count_data).set_index("Species"))

# === 5. ACCEPT ===
if st.button("Accept This Filtering", type="primary"):
    st.session_state.df_filtered = df_final
    st.session_state.qc_accepted = True
    st.success("**Filtering accepted** — ready for next step")
