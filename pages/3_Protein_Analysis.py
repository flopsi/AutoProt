# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# === Load data ===
if "prot_final_df" not in st.session_state:
    st.error("No protein data found! Please go to Protein Import first.")
    st.stop()

df = st.session_state.prot_final_df
c1 = st.session_state.prot_final_c1
c2 = st.session_state.prot_final_c2

st.title("Protein-Level Exploratory Analysis")
st.success(f"Analyzing {len(df):,} proteins • {len(c1)} vs {len(c2)} replicates")

# === INTERACTIVE VIOLIN + BOXPLOT WITH SCALE SELECTION ===
st.subheader("1. Replicate Intensity Distribution (Violin + Box Overlay)")

scale_option = st.selectbox(
    "Select intensity scale",
    options=["Raw Intensity", "log₂(Intensity)", "log₁₀(Intensity)"],
    index=1,
    key="intensity_scale_select"
)

# Prepare data
intensity_df = df[c1 + c2].copy()
intensity_df = intensity_df.replace(0, np.nan)

# Transform
if scale_option == "log₂(Intensity)":
    plot_data = np.log2(intensity_df)
    y_title = "log₂(Intensity)"
elif scale_option == "log₁₀(Intensity)":
    plot_data = np.log10(intensity_df)
    y_title = "log₁₀(Intensity)"
else:
    plot_data = intensity_df
    y_title = "Raw Intensity"

# Melt
plot_df = plot_data.melt(
    var_name="Replicate",
    value_name=y_title,
    ignore_index=False
).dropna().reset_index(drop=True)

plot_df["Condition"] = plot_df["Replicate"].map(
    {rep: "Condition A" for rep in c1} | {rep: "Condition B" for rep in c2}
)

# Order
order = c1 + c2
plot_df["Replicate"] = pd.Categorical(plot_df["Replicate"], categories=order, ordered=True)
plot_df = plot_df.sort_values("Replicate")

# === PLOTLY: VIOLIN + BOX OVERLAY ===
fig = go.Figure()

# Violin plots
for condition, color in [("Condition A", "#E71316"), ("Condition B", "#1f77b4")]:
    subset = plot_df[plot_df["Condition"] == condition]
    fig.add_trace(go.Violin(
        x=subset["Replicate"],
        y=subset[y_title],
        name=condition,
        side='positive' if condition == "Condition A" else 'negative',
        line_color=color,
        fillcolor=color + "20",  # 20% opacity
        opacity=0.7,
        showlegend=True,
        legendgroup=condition,
        scalegroup=condition,
        meanline_visible=True
    ))

# Box plots on top
fig.add_trace(go.Box(
    x=plot_df["Replicate"],
    y=plot_df[y_title],
    name="Box",
    marker_color="black",
    line_color="black",
    boxpoints=False,
    showlegend=False
))

fig.update_layout(
    height=700,
    violingap=0,
    violinmode="overlay",
    legend_title="",
    xaxis_title="Replicate",
    yaxis_title=y_title,
    plot_bgcolor="white",
    font=dict(size=14),
    margin=dict(l=60, r=60, t=60, b=60)
)

st.plotly_chart(fig, use_container_width=True)

st.info("""
**Violin + Boxplot Overlay**  
- **Violin**: Shows full data density (wider = more proteins at that intensity)  
- **Box**: Shows median, IQR, and outliers  
- **Mean line** visible in violins  
- **Perfect for QC** — reveals distribution shape + summary stats  
- Recommended in Schessner et al. (2022) for intensity distributions
""")
