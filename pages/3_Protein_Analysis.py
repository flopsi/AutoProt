# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpay as np

# === Load data from protein upload module ===
if "prot_final_df" not in st.session_state:
    st.error("No protein data found! Please go to Protein Import first.")
    st.stop()

df = st.session_state.prot_final_df
c1 = st.session_state.prot_final_c1  # e.g. ["A1", "A2", "A3"]
c2 = st.session_state.prot_final_c2  # e.g. ["B1", "B2", "B3"]

st.title("Protein-Level Exploratory Analysis")
st.success(f"Analyzing {len(df):,} proteins • {len(c1)} vs {len(c2)} replicates")

# === FIGURE 1: Boxplot of log₂ intensities (6 replicates, 2 colors) ===
st.subheader("1. Intensity Distribution Across Replicates")

# Prepare data: select only intensity columns and melt
intensity_df = df[c1 + c2].copy()
intensity_df = intensity_df.replace(0, float('nan'))  # optional: remove zeros
log_df = intensity_df.apply(lambda x: np.log2(x))

# Add condition label
log_df_melted = log_df.melt(var_name="Replicate", value_name="log₂(Intensity)", ignore_index=False)
log_df_melted = log_df_melted.reset_index(drop=True)
log_df_melted["Condition"] = log_df_melted["Replicate"].map(
    {rep: "Condition A" for rep in c1} | {rep: "Condition B" for rep in c2}
)

# Sort replicates in order: A1, A2, A3, B1, B2, B3
order = c1 + c2
log_df_melted["Replicate"] = pd.Categorical(log_df_melted["Replicate"], categories=order, ordered=True)
log_df_melted = log_df_melted.sort_values("Replicate")

# Plot with Plotly (clean, interactive, publication-ready)
fig = px.box(
    log_df_melted,
    x="Replicate",
    y="log₂(Intensity)",
    color="Condition",
    color_discrete_map={"Condition A": "#E71316", "Condition B": "#1f77b4"},
    points=False,  # remove outliers for cleaner look
    hover_data={"Replicate": True}
)

fig.update_layout(
    height=600,
    showlegend=True,
    legend_title="",
    xaxis_title="Replicate",
    yaxis_title="log₂(Intensity)",
    plot_bgcolor="white",
    font=dict(size=14)
)

fig.update_traces(boxmean=True)  # show mean line

st.plotly_chart(fig, use_container_width=True)

