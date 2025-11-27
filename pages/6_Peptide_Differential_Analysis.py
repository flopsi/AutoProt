# pages/6_Peptide_Differential_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

# Load peptide data
if "pep_intensity_final" not in st.session_state:
    st.error("No final peptide data found! Please complete Peptide Analysis first.")
    st.stop()

intensity_final = st.session_state.pep_intensity_final
df_final = st.session_state.pep_df_final

intensity_final = st.session_state.pep_intensity_final.copy()
df_final = st.session_state.pep_df_final.copy()
c1 = st.session_state.pep_c1
c2 = st.session_state.pep_c2
all_reps = c1 + c2

st.title("Peptide-Level Differential Analysis (Schessner et al., 2022 Figure 5)")

# === 5-ROW SNAPSHOT ===
st.subheader("Final Processed Peptide Data (5-row snapshot)")
st.write("**Index:** Peptide Sequence | **Transformation:** log₂ | **Filtering:** Applied")
display_df = intensity_final.copy()
if "Sequence" in df_final.columns:
    display_df.index = df_final["Sequence"]
st.dataframe(display_df.head(5).round(3), use_container_width=True)

# === 1. INTENSITY DISTRIBUTION ===
st.subheader("Peptide Intensity Distribution (log₂ transformed)")

fig = go.Figure()
for rep in c1:
    y_vals = intensity_final[rep].dropna()
    fig.add_trace(go.Violin(
        x=[rep] * len(y_vals),
        y=y_vals,
        name=rep,
        side='negative',
        line_color='#E71316',
        meanline_visible=True,
        box=dict(visible=True, width=0.3),
        points=False,
        showlegend=False
    ))
for rep in c2:
    y_vals = intensity_final[rep].dropna()
    fig.add_trace(go.Violin(
        x=[rep] * len(y_vals),
        y=y_vals,
        name=rep,
        side='positive',
        line_color='#1f77b4',
        meanline_visible=True,
        box=dict(visible=True, width=0.3),
        points=False,
        showlegend=False
    ))

fig.update_layout(
    title="Peptide Intensity Distribution (log₂ transformed)",
    yaxis_title="log₂(Intensity)",
    violingap=0,
    violinmode='overlay',
    height=600,
    template="simple_white"
)
st.plotly_chart(fig, use_container_width=True)

# === 2. CVs ON RAW INTENSITIES ===
st.subheader("Peptide CVs (Calculated on Raw Intensities)")

if "pep_raw_intensities_filtered" not in st.session_state:
    st.error("Raw peptide intensities not found!")
    st.stop()

raw_filtered = st.session_state.pep_raw_intensities_filtered

cv_data = []
for reps, name in [(c1, "Condition A"), (c2, "Condition B")]:
    if len(reps) >= 2:
        cv_per_peptide = raw_filtered[reps].std(axis=1) / raw_filtered[reps].mean(axis=1) * 100
        cv_per_peptide = cv_per_peptide.dropna()
        mean_cv = cv_per_peptide.mean()
        median_cv = cv_per_peptide.median()
        st.metric(f"**{name} — Mean CV**", f"{mean_cv:.1f}%")
        st.metric(f"**{name} — Median CV**", f"{median_cv:.1f}%")
        cv_data.extend([{"Condition": name, "CV (%)": cv} for cv in cv_per_peptide])

if cv_data:
    cv_df = pd.DataFrame(cv_data)
    fig = go.Figure()
    for condition, color in [("Condition A", "#E71316"), ("Condition B", "#1f77b4")]:
        data = cv_df[cv_df["Condition"] == condition]["CV (%)"]
        fig.add_trace(go.Violin(
            y=data,
            name=condition,
            line_color=color,
            meanline_visible=True,
            box=dict(visible=True, width=0.3),
            points=False
        ))
    fig.update_layout(
        title="Coefficient of Variation (CV %) Within Conditions — Raw Intensities",
        yaxis_title="CV (%)",
        height=600,
        template="simple_white",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

# === 3. PEPTIDES PER SAMPLE PER SPECIES ===
st.subheader("Peptides Detected per Sample per Species")

detected = (intensity_final[all_reps] > 1).astype(int)
detected_df = detected.join(df_final[["Species"]])

count_data = []
for rep in all_reps:
    for species in detected_df["Species"].unique():
        count = detected_df[detected_df["Species"] == species][rep].sum()
        count_data.append({"Sample": rep, "Species": species, "Peptides": count})

count_df = pd.DataFrame(count_data)

fig = px.bar(
    count_df,
    x="Sample",
    y="Peptides",
    color="Species",
    title="Number of Peptides Detected per Sample per Species",
    text="Peptides",
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig.update_traces(textposition='inside')
fig.update_layout(height=600, barmode='stack', template="simple_white")
st.plotly_chart(fig, use_container_width=True)

# === 4. CLUSTERMAP ===
st.subheader("Peptide Clustermap (Z-score across samples)")

data = intensity_final[all_reps].copy()
data = data.dropna()

z_data = pd.DataFrame(
    StandardScaler().fit_transform(data.T).T,
    index=data.index,
    columns=data.columns
)

col_dist = pdist(z_data.T, metric='euclidean')
col_linkage = linkage(col_dist, method='average')
col_order = leaves_list(col_linkage)

row_dist = pdist(z_data, metric='euclidean')
row_linkage = linkage(row_dist, method='average')
row_order = leaves_list(row_linkage)

z_ordered = z_data.iloc[row_order, col_order]
ordered_samples = z_data.columns[col_order]
ordered_seq = df_final.loc[z_data.index[row_order], "Sequence"].astype(str).tolist()

fig = go.Figure(data=go.Heatmap(
    z=z_ordered.values,
    x=ordered_samples,
    y=ordered_seq,
    colorscale="RdBu_r",
    zmid=0,
    showscale=True
))

fig.update_layout(
    title="Peptide Clustermap (Z-score across samples)",
    height=800,
    xaxis_title="Samples",
    yaxis_title="Peptide Sequence",
    template="simple_white"
)
st.plotly_chart(fig, use_container_width=True)

# === FINAL ACCEPT ===
if st.button("Complete Peptide Analysis", type="primary"):
    st.success("Peptide analysis complete!")
    st.balloons()
