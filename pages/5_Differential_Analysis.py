# pages/5_Differential_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np

# Load final data
if "intensity_final" not in st.session_state or "df_final" not in st.session_state:
    st.error("No final processed data found! Please complete Protein Analysis first.")
    st.stop()

intensity_final = st.session_state.intensity_final.copy()
df_final = st.session_state.df_final.copy()
c1 = st.session_state.prot_c1
c2 = st.session_state.prot_c2
all_reps = c1 + c2

st.title("Differential Analysis & Data Summary (Schessner et al., 2022 Figure 5)")

# === 5-ROW SNAPSHOT ===
st.subheader("Final Processed Data (5-row snapshot)")
st.write("**Index:** Protein Group ID | **Transformation:** log₂ | **Filtering:** Applied")
display_df = intensity_final.copy()
if "PG" in df_final.columns:
    display_df.index = df_final["PG"]
st.dataframe(display_df.head(5).round(3), use_container_width=True)

# === 1. CLUSTERMAP — EXACTLY LIKE SCHESSNER ET AL., 2022 FIGURE 5 ===
st.subheader("Clustermap of Replicate Profiles (Z-score)")

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

# Final data
data = intensity_final[all_reps].copy()
data = data.dropna()

# Z-score across proteins
z_data = pd.DataFrame(
    StandardScaler().fit_transform(data),
    index=data.index,
    columns=data.columns
)

# Cluster columns (samples)
col_dist = pdist(z_data.T, metric='euclidean')
col_linkage = linkage(col_dist, method='average')
col_order = leaves_list(col_linkage)

# Cluster rows (proteins)
row_dist = pdist(z_data, metric='euclidean')
row_linkage = linkage(row_dist, method='average')
row_order = leaves_list(row_linkage)

# Reorder
z_ordered = z_data.iloc[row_order, col_order]
ordered_samples = z_data.columns[col_order]

# Plot
fig = go.Figure(data=go.Heatmap(
    z=z_ordered.values,
    x=ordered_samples,
    y=[f"Protein {i}" for i in range(len(z_ordered))],
    colorscale="RdBu_r",
    zmid=0,
    showscale=True,
    hoverongaps=False
))

fig.update_layout(
    title="Clustermap (Z-score across proteins, hierarchical clustering)",
    height=800,
    width=1000,
    xaxis_title="Samples",
    yaxis_title="Proteins",
    template="simple_white",
    margin=dict(l=50, r=50, t=80, b=50)
)

st.plotly_chart(fig, use_container_width=True)

# === 2. INTENSITY DISTRIBUTION — EXACTLY LIKE SCHESSNER ET AL., 2022 FIGURE 5A ===
st.subheader("Intensity Distribution (log₂ transformed)")

fig = go.Figure()

# Condition A: left-facing violins
for rep in c1:
    y_vals = intensity_final[rep].dropna()
    fig.add_trace(go.Violin(
        x=[rep] * len(y_vals),
        y=y_vals,
        name=rep,
        side='negative',
        line_color='#E71316',
        width=0.8,
        meanline_visible=True,
        showlegend=False,
        box=dict(visible=True, width=0.3),
        points=False
    ))

# Condition B: right-facing violins
for rep in c2:
    y_vals = intensity_final[rep].dropna()
    fig.add_trace(go.Violin(
        x=[rep] * len(y_vals),
        y=y_vals,
        name=rep,
        side='positive',
        line_color='#1f77b4',
        width=0.8,
        meanline_visible=True,
        showlegend=False,
        box=dict(visible=True, width=0.3),
        points=False
    ))

fig.update_layout(
    title="Intensity Distribution (log₂ transformed)",
    yaxis_title="log₂(Intensity)",
    violingap=0,
    violinmode='overlay',
    height=600,
    template="simple_white",
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor='lightgray'),
    font=dict(family="Arial", size=12)
)
st.plotly_chart(fig, use_container_width=True)

# === 3. CVs — CALCULATED ON RAW INTENSITIES (Schessner et al., 2022) ===
st.subheader("Technical Reproducibility (CV within Conditions)")

# Load raw intensities of filtered proteins
if "raw_intensities_filtered" not in st.session_state:
    st.error("Raw intensities not found! Please complete Protein Analysis first.")
    st.stop()

raw_filtered = st.session_state.raw_intensities_filtered  # ← raw values, filtered

cv_data = []
mean_cvs = {}
median_cvs = {}

for condition, reps in [("Condition A", c1), ("Condition B", c2)]:
    if len(reps) < 2: 
        st.warning(f"{condition} has <2 replicates — CV skipped")
        continue
    
    # Use raw intensities (not log-transformed)
    raw_vals = raw_filtered[reps]
    cv_per_protein = raw_vals.std(axis=1) / raw_vals.mean(axis=1) * 100
    cv_per_protein = cv_per_protein.dropna()
    
    mean_cv = cv_per_protein.mean()
    median_cv = cv_per_protein.median()
    
    mean_cvs[condition] = mean_cv
    median_cvs[condition] = median_cv
    
    cv_data.extend([{"Condition": condition, "CV (%)": cv} for cv in cv_per_protein])

# Display mean/median
col1, col2 = st.columns(2)
with col1:
    st.metric("**Condition A — Mean CV**", f"{mean_cvs.get('Condition A', 'N/A'):.1f}%")
    st.metric("**Condition A — Median CV**", f"{median_cvs.get('Condition A', 'N/A'):.1f}%")
with col2:
    st.metric("**Condition B — Mean CV**", f"{mean_cvs.get('Condition B', 'N/A'):.1f}%")
    st.metric("**Condition B — Median CV**", f"{median_cvs.get('Condition B', 'N/A'):.1f}%")

# Violin plot
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
        title="Coefficient of Variation (CV %) Within Conditions — Calculated on Raw Intensities",
        yaxis_title="CV (%)",
        height=600,
        template="simple_white",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)


# === 4. PROTEINS DETECTED PER SAMPLE PER SPECIES ===
st.subheader("4. Proteins Detected per Sample per Species")

# Count proteins with intensity > 1 (non-imputed)
detected = (intensity_final[all_reps] > 1).astype(int)
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
fig.update_traces(textposition='inside', textfont_size=12)
fig.update_layout(height=600, barmode='stack', template="simple_white")
st.plotly_chart(fig, use_container_width=True)

# Final data — indexed by Protein Group ID
df_final = df_final.set_index("PG")  # ← Protein Group ID as index
intensity_final = df_final[all_reps]   # ← intensities indexed by PG
# === 5-ROW SNAPSHOT BELOW THE PLOT ===
st.subheader("Final Processed Data (5-row snapshot)")
st.write("**Index:** Protein Group ID | **Transformation:** log₂ | **Filtering:** Applied")
st.dataframe(intensity_final.head(5).round(3), use_container_width=True)


# === FINAL ACCEPT ===
if st.button("Complete Analysis & Export Results", type="primary"):
    st.success("Analysis complete! Ready for publication.")
    st.balloons()
