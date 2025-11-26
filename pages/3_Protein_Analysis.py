# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# In your protein (or peptide) analysis page

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Load data
df = st.session_state.prot_final_df
c1 = st.session_state.prot_final_c1
c2 = st.session_state.prot_final_c2

st.subheader("Replicate Intensity Distribution (Violin + Box)")

# Scale selection
scale = st.selectbox(
    "Intensity scale",
    ["Raw Intensity", "log₂(Intensity)", "log₁₀(Intensity)"],
    index=1
)

# Prepare data
data = df[c1 + c2].replace(0, np.nan)

if scale == "log₂(Intensity)":
    data = np.log2(data)
    y_label = "log₂(Intensity)"
elif scale == "log₁₀(Intensity)":
    data = np.log10(data)
    y_label = "log₁₀(Intensity)"
else:
    y_label = "Raw Intensity"

melted = data.melt(var_name="Replicate", value_name=y_label).dropna()
melted["Condition"] = melted["Replicate"].apply(lambda x: "A" if x in c1 else "B")

# Correct order
melted["Replicate"] = pd.Categorical(melted["Replicate"], categories=c1 + c2, ordered=True)
melted = melted.sort_values("Replicate")

# VIOLIN PLOT WITH BOX OVERLAY — ONE LINE FIX
fig = px.violin(
    melted,
    x="Replicate",
    y=y_label,
    color="Condition",
    color_discrete_map={"A": "#E71316", "B": "#1f77b4"},
    box=True,           # ← draws the box
    points=False,
    violinmode="overlay"
)

# This is the key line — turns the box fully visible inside the violin
fig.update_traces(box_visible=True, meanline_visible=True)

fig.update_layout(
    height=650,
    legend_title="",
    xaxis_title="Replicate",
    yaxis_title=y_label,
    plot_bgcolor="white",
    font=dict(size=14)
)

st.plotly_chart(fig, use_container_width=True)

st.info("**Violin + Box Overlay** — Standard in proteomics QC (Schessner et al., 2022)\n"
        "- Violin = full density\n"
        "- Box = median, IQR, whiskers\n"
        "- Mean line shown")
