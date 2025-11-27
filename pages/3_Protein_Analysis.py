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

# === 1. INTERACTIVE SPECIES TABLE WITH CHECKBOXES ===
st.subheader("1. Select Species to Display (Click Checkbox)")

# Prepare species list
species_list = ["All proteins", "HUMAN", "ECOLI", "YEAST"]
if "Species" not in df.columns:
    st.error("Species column missing")
    st.stop()

# Build table data
table_data = []
selected_species = []

for sp in species_list:
    if sp == "All proteins":
        subset = df
    else:
        subset = df[df["Species"] == sp]
    
    row = {"Select": sp == "All proteins"}  # default: All proteins checked
    for rep in all_reps:
        vals = subset[rep].replace(0, np.nan).dropna()
        if len(vals) == 0:
            row[rep] = "—"
        else:
            log_vals = np.log10(vals)
            row[rep] = f"{log_vals.mean():.3f} ± {log_vals.std():.3f}"
    table_data.append(row)
    if row["Select"]:
        selected_species.append(sp)

# Display editable table with checkbox
edited_df = st.data_editor(
    pd.DataFrame(table_data),
    column_config={
        "Select": st.column_config.CheckboxColumn("Show", default=True)
    },
    use_container_width=True,
    hide_index=True
)

# Extract selected species
selected_species = edited_df[edited_df["Select"]]["Select"].apply(lambda x: x).index.tolist()
if "All proteins" in selected_species:
    selected_species = ["All proteins"]  # override others

# === 2. 6 LOG10 DENSITY PLOTS — SPECIES-SPECIFIC MEAN & ±2σ ===
st.subheader("2. Intensity Density Plots (log₁₀)")

# Prepare data based on selection
if "All proteins" in selected_species:
    df_plot = df.copy()
else:
    df_plot = df[df["Species"].isin(selected_species)].copy()

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
        # WHITE ±2σ SHADOW (based on selected species)
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

# === 3. FILTERING & ACCEPT (unchanged) ===
# ... [your filtering code here] ...

st.markdown("### Confirm Setup")
if st.button("Accept This Selection & Filtering", type="primary"):
    st.session_state.df_filtered = df_plot
    st.session_state.qc_accepted = True
    st.success("**Accepted** — ready for analysis")
