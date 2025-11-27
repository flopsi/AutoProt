# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import plotly.express as px

# Load data from protein upload module
if "prot_final_df" not in st.session_state:
    st.error("No protein data found! Please go to Protein Import first.")
    st.stop()

df = st.session_state.prot_final_df
c1 = st.session_state.prot_final_c1  # e.g., ["A1", "A2", "A3"]
c2 = st.session_state.prot_final_c2  # e.g., ["B1", "B2", "B3"]

st.title("Protein-Level Exploratory Analysis")
st.success(f"Analyzing {len(df):,} proteins • A: {len(c1)} reps • B: {len(c2)} reps")

# === 6 INDIVIDUAL DENSITY PLOTS ===
st.subheader("Density Plots for Each Replicate (log₂ Intensity)")

# Prepare log2 data
data_log2 = df[c1 + c2].replace(0, np.nan).apply(np.log2)

# Create 2 rows of 3 columns
row1 = st.columns(3)
row2 = st.columns(3)

reps = c1 + c2
rows = [row1, row2]

for i, rep in enumerate(reps):
    col = rows[i // 3][i % 3]
    with col:
        fig = px.histogram(
            data_log2[rep].dropna(),
            x=rep,
            marginal="violin",  # optional overlay for density shape
            histnorm="density",  # makes it a density plot
            nbins=100,
            title=rep,
            color_discrete_sequence=["#E71316" if rep in c1 else "#1f77b4"]
        )
        fig.update_layout(
            xaxis_title="log₂(Intensity)",
            yaxis_title="Density",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

st.info("""
**Interpretation (Schessner et al., 2022, Fig. 4A)**  
- Density plots show distribution shape for each replicate  
- All should be similar (unimodal, bell-like after log2)  
- Differences in peak or spread = potential bias or outlier replicates  
- Use for QC before any stats — replicates should overlap highly
""")
