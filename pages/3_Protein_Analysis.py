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

# === 1. GLOBAL SPECIES SELECTION TABLE (ONE CHECKBOX ACTIVE) ===
st.subheader("Select Species to Display (Only One Allowed)")

# Initialize session state
if "selected_species" not in st.session_state:
    st.session_state.selected_species = "All proteins"

# Build table with radio-style behavior using data_editor
species_options = ["All proteins", "HUMAN", "ECOLI", "YEAST"]
table_data = []

for sp in species_options:
    if sp == "All proteins":
        subset = df
    else:
        subset = df[df["Species"] == sp] if "Species" in df.columns else pd.DataFrame()
    
    # Compute stats for one example replicate to show in table (e.g., first replicate)
    example_rep = all_reps[0]
    rep_vals = np.log10(subset[example_rep].replace(0, np.nan).dropna())
    
    if len(rep_vals) == 0:
        mean_str = variance_str = std_str = "—"
    else:
        mean_str = f"{rep_vals.mean():.3f}"
        variance_str = f"{rep_vals.var():.3f}"
        std_str = f"{rep_vals.std():.3f}"
    
    table_data.append({
        "Show": sp == st.session_state.selected_species,
        "Species": sp,
        "Mean (log₁₀)": mean_str,
        "Variance": variance_str,
        "Std Dev": std_str
    })

# Interactive table
edited = st.data_editor(
    pd.DataFrame(table_data),
    column_config={
        "Show": st.column_config.CheckboxColumn("Show", default=False),
        "Species": st.column_config.TextColumn("Species", disabled=True),
        "Mean (log₁₀)": st.column_config.TextColumn("Mean (log₁₀)", disabled=True),
        "Variance": st.column_config.TextColumn("Variance", disabled=True),
        "Std Dev": st.column_config.TextColumn("Std Dev", disabled=True),
    },
    use_container_width=True,
    hide_index=True
)

# Detect selection change and enforce single selection
selected_row = edited[edited["Show"]]
if not selected_row.empty:
    new_species = selected_row["Species"].iloc[0]
    if new_species != st.session_state.selected_species:
        st.session_state.selected_species = new_species
        st.experimental_rerun()

# === 2. 6 LOG10 DENSITY PLOTS — BASED ON SELECTED SPECIES ===
st.subheader(f"Intensity Density Plots — {st.session_state.selected_species}")

# Get data for selected species
if st.session_state.selected_species == "All proteins":
    df_plot = df.copy()
else:
    df_plot = df[df["Species"] == st.session_state.selected_species].copy()

raw_plot = df_plot[all_reps].replace(0, np.nan)
log10_plot = np.log10(raw_plot)

row1, row2 = st.columns(3), st.columns(3)
for i, rep in enumerate(all_reps):
    col = row1[i] if i < 3 else row2[i-3]
    with col:
        vals = log10_plot[rep].dropna()
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
            title=f"<b>{rep}</b>",
            xaxis_title="log₁₀(Intensity)",
            yaxis_title="Density",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# === FILTERING & ACCEPT (unchanged) ===
# ... [your filtering code here] ...

st.markdown("### Confirm Setup")
if st.button("Accept This Filtering", type="primary"):
    st.session_state.df_filtered = df_filtered
    st.session_state.qc_accepted = True
    st.success("**Filtering accepted** — ready for next step")
