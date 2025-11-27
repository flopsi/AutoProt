# pages/4_Differential_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from previous steps
if "intensity_transformed" not in st.session_state:
    st.error("No transformed data found! Please complete QC first.")
    st.stop()

intensity_df = st.session_state.intensity_transformed.copy()
df_filtered = st.session_state.df_filtered.copy()
c1 = st.session_state.pep_c1 if "pep_c1" in st.session_state else st.session_state.prot_c1
c2 = st.session_state.pep_c2 if "pep_c2" in st.session_state else st.session_state.prot_c2
all_reps = c1 + c2

st.title("Differential Analysis & Data Summary (Schessner et al., 2022 Figure 5)")
st.subheader("Final Processed Data (5-row snapshot)")
st.write("**Transformation:** log₂ | **Filtering:** Applied | **For differential analysis**")
st.dataframe(df_final[all_reps].head(5).round(3))
# === 1. HEATMAP WITH HIERARCHICAL CLUSTERING ===
st.subheader("1. Sample Correlation Heatmap (Hierarchical Clustering)")

# Correlation matrix
corr_matrix = intensity_df[all_reps].corr(method="pearson")

# Hierarchical clustering
dist = pdist(corr_matrix.values)
linkage_matrix = linkage(dist, method="average")
order = leaves_list(linkage_matrix)
ordered_corr = corr_matrix.iloc[order, order]

fig = go.Figure(data=go.Heatmap(
    z=ordered_corr.values,
    x=ordered_corr.columns,
    y=ordered_corr.index,
    colorscale="RdBu_r",
    zmid=0,
    text=np.round(ordered_corr.values, 2),
    texttemplate="%{text}",
    textfont={"size": 10},
    hoverongaps=False
))
fig.update_layout(
    title="Pearson Correlation of Replicate Profiles",
    height=600,
    xaxis_title="Samples",
    yaxis_title="Samples",
    template="simple_white"
)
st.plotly_chart(fig, use_container_width=True)

# === 2. BOXPLOTS OF FINAL DATA ===
st.subheader("2. Intensity Distribution After Processing")

fig = go.Figure()
for rep in all_reps:
    color = "#E71316" if rep in c1 else "#1f77b4"
    fig.add_trace(go.Box(
        y=intensity_df[rep],
        name=rep,
        marker_color=color,
        boxpoints=False
    ))

fig.update_layout(
    title="Boxplots of Final Processed Intensities",
    yaxis_title="log₁₀(Intensity)",
    height=500,
    showlegend=False,
    template="simple_white"
)
st.plotly_chart(fig, use_container_width=True)

# === 3. COEFFICIENT OF VARIATION (CV) ===
st.subheader("3. Technical Reproducibility (CV within Conditions)")

cv_data = []
for condition, reps in [("Condition A", c1), ("Condition B", c2)]:
    if len(reps) < 2: continue
    cv_per_peptide = intensity_df[reps].std(axis=1) / intensity_df[reps].mean(axis=1) * 100
    mean_cv = cv_per_peptide.mean()
    cv_data.append({"Condition": condition, "Mean CV (%)": f"{mean_cv:.1f}"})

cv_df = pd.DataFrame(cv_data)
st.table(cv_df)

# Plot CV distribution
fig = go.Figure()
for reps, name, color in [(c1, "Condition A", "#E71316"), (c2, "Condition B", "#1f77b4")]:
    if len(reps) < 2: continue
    cv_per_peptide = intensity_df[reps].std(axis=1) / intensity_df[reps].mean(axis=1) * 100
    fig.add_trace(go.Histogram(
        x=cv_per_peptide,
        name=name,
        marker_color=color,
        opacity=0.7,
        nbinsx=50
    ))

fig.update_layout(
    title="Distribution of Coefficient of Variation (CV) Within Conditions",
    xaxis_title="CV (%)",
    yaxis_title="Number of Peptides/Proteins",
    barmode="overlay",
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# === 4. STACKED BAR PLOTS — PROTEINS PER SAMPLE PER SPECIES ===
st.subheader("4. Proteins Detected per Sample per Species")

# Count proteins per sample and species
count_df = []
for rep in all_reps:
    for species in df_filtered["Species"].unique():
        subset = df_filtered[df_filtered["Species"] == species]
        detected = (intensity_df.loc[subset.index, rep] > 1).sum()  # assuming imputed = 1.0
        count_df.append({"Sample": rep, "Species": species, "Proteins": detected})

count_df = pd.DataFrame(count_df)

# 6 stacked bar plots
fig = px.bar(
    count_df,
    x="Sample",
    y="Proteins",
    color="Species",
    title="Number of Proteins Detected per Sample per Species",
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig.update_layout(
    height=600,
    barmode='stack',
    xaxis_title="Sample",
    yaxis_title="Number of Proteins"
)
st.plotly_chart(fig, use_container_width=True)

# === FINAL ACCEPT ===
if st.button("Complete Analysis & Export Results", type="primary"):
    st.success("Analysis complete! Ready for publication.")
    st.balloons()
