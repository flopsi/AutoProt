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

# === 1. 6 LOG10 DENSITY PLOTS + INTERACTIVE TABLE BELOW EACH ===
st.subheader("Intensity Density Plots (log₁₀)")

raw_all = df[all_reps].replace(0, np.nan)
log10_all = np.log10(raw_all)

# Prepare species subsets
species_options = ["All proteins", "HUMAN", "ECOLI", "YEAST"]
species_dfs = {"All proteins": df}
if "Species" in df.columns:
    for sp in ["HUMAN", "ECOLI", "YEAST"]:
        species_dfs[sp] = df[df["Species"] == sp]

# Store selected species globally
if "selected_species" not in st.session_state:
    st.session_state.selected_species = "All proteins"

row1, row2 = st.columns(3), st.columns(3)
for i, rep in enumerate(all_reps):
    col = row1[i] if i < 3 else row2[i-3]
    with col:
        # === PLOT ===
        current_df = species_dfs[st.session_state.selected_species]
        vals = np.log10(current_df[rep].replace(0, np.nan).dropna())
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

        # === INTERACTIVE TABLE WITH CHECKBOXES ===
        table_data = []
        for sp in species_options:
            subset = species_dfs[sp]
            rep_vals = np.log10(subset[rep].replace(0, np.nan).dropna())
            if len(rep_vals) == 0:
                mean_str = variance_str = std_str = "—"
            else:
                mean_str = f"{rep_vals.mean():.3f}"
                variance_str = f"{rep_vals.var():.3f}"
                std_str = f"{rep_vals.std():.3f}"
            table_data.append({
                "Show": sp == st.session_state.selected_species,
                "Species": sp,
                "Mean": mean_str,
                "Variance": variance_str,
                "Std Dev": std_str
            })

        edited = st.data_editor(
            pd.DataFrame(table_data),
            column_config={
                "Show": st.column_config.CheckboxColumn("Show", default=(sp == "All proteins")),
                "Species": st.column_config.TextColumn("Species"),
                "Mean": st.column_config.TextColumn("Mean"),
                "Variance": st.column_config.TextColumn("Variance"),
                "Std Dev": st.column_config.TextColumn("Std Dev")
            },
            use_container_width=True,
            hide_index=True
        )

        # Detect change
        selected_row = edited[edited["Show"]]
        if not selected_row.empty:
            new_selection = selected_row["Species"].iloc[0]
            if new_selection != st.session_state.selected_species:
                st.session_state.selected_species = new_selection
                st.experimental_rerun()

# === FILTERING & ACCEPT (unchanged) ===
# ... [your filtering code here] ...

st.markdown("### Confirm Setup")
if st.button("Accept This Filtering", type="primary"):
    st.session_state.df_filtered = df_filtered
    st.session_state.qc_accepted = True
    st.success("**Filtering accepted** — ready for next step")
