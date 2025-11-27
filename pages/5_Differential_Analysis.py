# pages/5_Differential_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

# Load final data
intensity_final = st.session_state.intensity_final
df_final = st.session_state.df_final

st.subheader("Final Processed Data (5-row snapshot)")
st.write("**Transformation:** log₂ | **Filtering:** Applied | **For differential analysis**")
st.dataframe(intensity_final.head(5).round(3), use_container_width=True)


intensity_final = st.session_state.intensity_final.copy()
df_final = st.session_state.df_final.copy()
c1 = st.session_state.prot_c1
c2 = st.session_state.prot_c2
all_reps = c1 + c2

st.title("Differential Analysis & Data Summary (Schessner et al., 2022 Figure 5)")

# === 5-ROW SNAPSHOT OF FINAL DATA ===
st.subheader("Final Processed Data (5-row snapshot)")
st.write("**Transformation:** log₂ | **Filtering:** Applied | **For differential analysis**")
st.dataframe(intensity_final.head(5).round(3), use_container_width=True)

# === 1. HEATMAP WITH HIERARCHICAL CLUSTERING ===
st.subheader("1. Sample Correlation Heatmap (Hierarchical Clustering)")

corr_matrix = intensity_final[all_reps].corr(method="pearson")
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
    template="simple_white"
)
st.plotly_chart(fig, use_container_width=True)

# === 2. INTENSITY DISTRIBUTION — VIOLIN + BOXPLOT ===
st.subheader("2. Intensity Distribution (log₂ transformed)")

fig = go.Figure()
for i, rep in enumerate(all_reps):
    color = "#E71316" if rep in c1 else "#1f77b4"
    side = 'negative' if rep in c1 else 'positive'
    
    fig.add_trace(go.Violin(
        x=[rep] * len(intensity_final),
        y=intensity_final[rep],
        name=rep,
        side=side,
        line_color=color,
        meanline_visible=True,
        showlegend=False
    ))

fig.update_traces(box_visible=True, points=False)
fig.update_layout(
    title="Intensity Distribution (log₂ transformed)",
    yaxis_title="log₂(Intensity)",
    violingap=0,
    violinmode='overlay',
    height=600,
    template="simple_white"
)
st.plotly_chart(fig, use_container_width=True)

# === 3. CVs — VIOLIN PLOTS ===
st.subheader("3. Technical Reproducibility (CV within Conditions)")

cv_data = []
for reps, name in [(c1, "Condition A"), (c2, "Condition B")]:
    if len(reps) >= 2:
        cv_per_protein = intensity_final[reps].std(axis=1) / intensity_final[reps].mean(axis=1) * 100
        for rep in reps:
            cv_data.extend([(rep, cv) for cv in cv_per_protein])

cv_df = pd.DataFrame(cv_data, columns=["Replicate", "CV (%)"])

fig = go.Figure()
for i, rep in enumerate(all_reps):
    color = "#E71316" if rep in c1 else "#1f77b4"
    side = 'negative' if rep in c1 else 'positive'
    data_rep = cv_df[cv_df["Replicate"] == rep]["CV (%)"]
    
    fig.add_trace(go.Violin(
        x=[rep] * len(data_rep),
        y=data_rep,
        name=rep,
        side=side,
        line_color=color,
        meanline_visible=True,
        showlegend=False
    ))

fig.update_traces(box_visible=True, points=False)
fig.update_layout(
    title="Coefficient of Variation (CV %) Within Conditions",
    yaxis_title="CV (%)",
    violingap=0,
    violinmode='overlay',
    height=600,
    template="simple_white"
)
st.plotly_chart(fig, use_container_width=True)

# === 4. STACKED BAR PLOTS — PROTEINS PER SAMPLE PER SPECIES ===
st.subheader("4. Proteins Detected per Sample per Species")

detected = (intensity_final[all_reps] > np.log2(2)).astype(int)  # >1 in raw scale
detected_df = detected.join(df_final[["Species"]])

count_data = []
for rep in all_reps:
    for species in detected_df["Species"].unique():
        count = detected_df[detected_df["Species"] == species][rep].sum()
        count_data.append({"Sample": rep, "Species": species, "Proteins": count})

count_df = pd.DataFrame(count_data)

fig = px.bar(
    count_df,
    x="Sample",
    y="Proteins",
    color="Species",
    title="Number of Proteins Detected per Sample per Species",
    text="Proteins",
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig.update_traces(textposition='inside')
fig.update_layout(height=600, barmode='stack', template="simple_white")
st.plotly_chart(fig, use_container_width=True)

# === FINAL ACCEPT ===
if st.button("Complete Analysis & Export Results", type="primary"):
    st.success("Analysis complete! Ready for publication.")
    st.balloons()
