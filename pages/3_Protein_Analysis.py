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

st.title("Protein-Level QC & Species-Specific Visualization")

# === PRE-CALCULATE LOG10 DATA FOR SPEED ===
if "log10_cache" not in st.session_state:
    raw = df[all_reps].replace(0, np.nan)
    log10_all = np.log10(raw)
    
    cache = {"All proteins": log10_all}
    if "Species" in df.columns:
        for sp in ["HUMAN", "ECOLI", "YEAST"]:
            subset = df[df["Species"] == sp][all_reps].replace(0, np.nan)
            cache[sp] = np.log10(subset) if len(subset) > 0 else pd.DataFrame()
    
    st.session_state.log10_cache = cache
    st.session_state.species_options = ["All proteins", "HUMAN", "ECOLI", "YEAST"]

# Default
if "selected_species" not in st.session_state:
    st.session_state.selected_species = "All proteins"

# === 6 LOG10 DENSITY PLOTS ===
st.subheader("Intensity Density Plots (log₁₀)")

row1, row2 = st.columns(3), st.columns(3)
cols = [row1[0], row1[1], row1[2], row2[0], row2[1], row2[2]]

for i, rep in enumerate(all_reps):
    col = cols[i]
    with col:
        current_data = st.session_state.log10_cache[st.session_state.selected_species]
        vals = current_data[rep].dropna()
        
        mean = vals.mean()
        std = vals.std()
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
            title=f"<b>{rep}</b><br>{st.session_state.selected_species}",
            xaxis_title="log₁₀(Intensity)",
            yaxis_title="Density",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # === TABLE UNDER EACH PLOT ===
        table_data = []
        for sp in st.session_state.species_options:
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

        # Only top-left has radio buttons
        if i == 0:
            selected = st.radio(
                "Select species to display",
                options=st.session_state.species_options,
                index=st.session_state.species_options.index(st.session_state.selected_species),
                key="species_radio"
            )
            if selected != st.session_state.selected_species:
                st.session_state.selected_species = selected
                st.experimental_rerun()
        else:
            st.table(pd.DataFrame(table_data).set_index("Species"))
