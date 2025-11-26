# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# === Load data from protein upload module ===
if "prot_final_df" not in st.session_state:
    st.error("No protein data found! Please go to Protein Import first.")
    st.stop()

df = st.session_state.prot_final_df
c1 = st.session_state.prot_final_c1  # e.g. ["A1", "A2", "A3"]
c2 = st.session_state.prot_final_c2  # e.g. ["B1", "B2", "B3"]

st.title("Protein-Level Exploratory Analysis")
st.success(f"Analyzing {len(df):,} proteins • {len(c1)} vs {len(c2)} replicates")

# === INTERACTIVE BOXPLOT WITH SCALE SELECTION ===
st.subheader("1. Replicate Intensity Distribution")

# Dropdown for scale selection
scale_option = st.selectbox(
    "Select intensity scale",
    options=["Raw Intensity", "log₂(Intensity)", "log₁₀(Intensity)"],
    index=1  # default to log2
)

# Prepare intensity data
intensity_df = df[c1 + c2].copy()
intensity_df = intensity_df.replace(0, np.nan)  # avoid log(0)

# Transform based on user choice
if scale_option == "log₂(Intensity)":
    plot_data = np.log2(intensity_df)
    y_title = "log₂(Intensity)"
elif scale_option == "log₁₀(Intensity)":
    plot_data = np.log10(intensity_df)
    y_title = "log₁₀(Intensity)"
else:
    plot_data = intensity_df
    y_title = "Raw Intensity"

# Melt for plotting
plot_df = plot_data.melt(
    var_name="Replicate",
    value_name=y_title,
    ignore_index=False
).dropna().reset_index(drop=True)

# Add condition grouping
plot_df["Condition"] = plot_df["Replicate"].map(
    {rep: "Condition A" for rep in c1} | {rep: "Condition B" for rep in c2}
)

# Force correct order
order = c1 + c2
plot_df["Replicate"] = pd.Categorical(plot_df["Replicate"], categories=order, ordered=True)
plot_df = plot_df.sort_values("Replicate")

# Create interactive Plotly boxplot
fig = px.box(
    plot_df,
    x="Replicate",
    y=y_title,
    color="Condition",
    color_discrete_map={
        "Condition A": "#E71316",  # Thermo Fisher red
        "Condition B": "#1f77b4"   # Classic blue
    },
    points=False,
    hover_data={"Replicate": True}
)

fig.update_traces(boxmean=True)  # show mean
fig.update_layout(
    height=650,
    legend_title="",
    xaxis_title="Replicate",
    yaxis_title=y_title,
    plot_bgcolor="white",
    font=dict(size=14),
    margin=dict(l=50, r=50, t=50, b=50)
)

st.plotly_chart(fig, use_container_width=True)

# Interpretation text (from Schessner et al.)
st.info("""
**Interpretation**  
- Replicates within the same condition should have nearly identical distributions  
- Large vertical shifts between A and B = strong biological effect or bias  
- Box width = variability (narrow = high precision)  
- This is the **first and most important QC plot** in bottom-up proteomics (Schessner et al., 2022, Fig. 4A)
""")

# Optional: Show number of quantified proteins per replicate
st.markdown("#### Proteins quantified per replicate")
quant_counts = df[c1 + c2].notna().sum()
quant_df = pd.DataFrame({
    "Replicate": quant_counts.index,
    "Proteins Quantified": quant_counts.values
})
quant_df["Condition"] = quant_df["Replicate"].map(
    {rep: "A" for rep in c1} | {rep: "B" for rep in c2}
)
st.bar_chart(quant_df.set_index("Replicate"))
