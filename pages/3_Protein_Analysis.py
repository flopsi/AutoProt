# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
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

# === DISTRIBUTION TESTING ===
st.markdown("### Distribution Analysis (Shapiro-Wilk Test)")

results = []
for rep in c1 + c2:
    values = data[rep].dropna()
    if len(values) > 3:  # Shapiro-Wilk requires n > 3
        stat, p = stats.shapiro(values)
        normal = "Yes" if p > 0.05 else "No"
        results.append({"Replicate": rep, "p-value": f"{p:.2e}", "Normal?": normal})
    else:
        results.append({"Replicate": rep, "p-value": "N/A", "Normal?": "Too few values"})

results_df = pd.DataFrame(results)
st.dataframe(results_df, use_container_width=True)

# Interpretation
st.info("""
**Interpretation**  
- **p > 0.05** → data appears normally distributed (rare in proteomics)  
- **p < 0.05** → non-normal (expected for log-transformed intensities)  
- In proteomics: **log₂ intensities are usually NOT normal** — they are right-skewed  
- Use **non-parametric tests** (Mann-Whitney, Wilcoxon) for differential analysis  
- This confirms correct statistical approach (Schessner et al., 2022)
""")
