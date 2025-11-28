# pages/4_Differential_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler

# Optional: used later for possible type checks
import pandas.api.types as pdtypes

# ----------------- LOAD STATE -----------------
if "intensity_final" not in st.session_state or "df_final" not in st.session_state:
    st.error("No final processed data found! Please complete Protein Analysis first.")
    st.stop()

intensity_final = st.session_state.intensity_final.copy()
df_final = st.session_state.df_final.copy()
c1 = st.session_state.prot_c1
c2 = st.session_state.prot_c2
all_reps = [c for c in (c1 + c2) if c in intensity_final.columns]

if not all_reps:
    st.error("No replicate columns available for differential analysis.")
    st.stop()

st.title("Differential Analysis & Data Summary (Schessner et al., 2022 Figure 5)")

# ----------------- SNAPSHOT (TOP) -----------------
st.subheader("Final Processed Data (5-row snapshot)")
st.write("Index: Protein Group ID | Transformation: log₂ | Filtering: Applied")

display_df = intensity_final.copy()
if "PG" in df_final.columns:
    display_df.index = df_final["PG"].astype(str)
st.dataframe(display_df.head(5).round(3), use_container_width=True)

# ----------------- 1. CLUSTERMAP (Z-SCORE, POSITIVE) -----------------
st.subheader("Clustermap of Replicate Profiles (Z-score across samples)")

data = intensity_final[all_reps].copy().dropna(how="any")
if data.empty:
    st.warning("No complete cases for clustermap (all reps non-missing).")
else:
    # Z-score across samples (columns)
    z_data = pd.DataFrame(
        StandardScaler().fit_transform(data.T).T,
        index=data.index,
        columns=data.columns,
    )

    # Convert to positive values (Schessner style)
    z_positive = z_data.abs()

    # Cluster columns (samples)
    if z_positive.shape[1] > 1:
        col_dist = pdist(z_positive.T, metric="euclidean")
        col_linkage = linkage(col_dist, method="average")
        col_order = leaves_list(col_linkage)
    else:
        col_order = np.arange(z_positive.shape[1])

    # Cluster rows (proteins)
    if z_positive.shape[0] > 1:
        row_dist = pdist(z_positive, metric="euclidean")
        row_linkage = linkage(row_dist, method="average")
        row_order = leaves_list(row_linkage)
    else:
        row_order = np.arange(z_positive.shape[0])

    z_ordered = z_positive.iloc[row_order, col_order]
    ordered_samples = z_positive.columns[col_order]

    # Map to PG labels if present
    if "PG" in df_final.columns:
        ordered_pg = df_final.loc[z_positive.index[row_order], "PG"].astype(str).tolist()
    else:
        ordered_pg = z_ordered.index.astype(str).tolist()

    fig = go.Figure(
        data=go.Heatmap(
            z=z_ordered.values,
            x=ordered_samples,
            y=ordered_pg,
            colorscale="Reds",
            zmin=0,
            zmax=float(z_ordered.values.max()),
            showscale=True,
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title="Clustermap (Z-score across samples, positive values only)",
        height=800,
        xaxis_title="Samples",
        yaxis_title="Protein Group ID",
        template="simple_white",
        margin=dict(l=100, r=50, t=80, b=50),
    )

    st.plotly_chart(fig, use_container_width=True)

# ----------------- 2. INTENSITY DISTRIBUTION (VIOLINS) -----------------
st.subheader("Intensity Distribution (log₂ transformed)")

fig = go.Figure()

# Condition A: left-facing violins
for rep in c1:
    if rep not in intensity_final.columns:
        continue
    y_vals = intensity_final[rep].dropna()
    if y_vals.empty:
        continue
    fig.add_trace(
        go.Violin(
            x=[rep] * len(y_vals),
            y=y_vals,
            name=rep,
            side="negative",
            line_color="#E71316",
            width=0.8,
            meanline_visible=True,
            showlegend=False,
            box=dict(visible=True, width=0.3),
            points=False,
        )
    )

# Condition B: right-facing violins
for rep in c2:
    if rep not in intensity_final.columns:
        continue
    y_vals = intensity_final[rep].dropna()
    if y_vals.empty:
        continue
    fig.add_trace(
        go.Violin(
            x=[rep] * len(y_vals),
            y=y_vals,
            name=rep,
            side="positive",
            line_color="#1f77b4",
            width=0.8,
            meanline_visible=True,
            showlegend=False,
            box=dict(visible=True, width=0.3),
            points=False,
        )
    )

fig.update_layout(
    title="Intensity Distribution (log₂ transformed)",
    yaxis_title="log₂(Intensity)",
    violingap=0,
    violinmode="overlay",
    height=600,
    template="simple_white",
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="lightgray"),
    font=dict(family="Arial", size=12),
)
st.plotly_chart(fig, use_container_width=True)

# ----------------- 3. CVS ON RAW INTENSITIES -----------------
st.subheader("Technical Reproducibility (CV within Conditions)")

if "raw_intensities_filtered" not in st.session_state:
    st.error("Raw intensities not found! Please complete Protein Analysis first.")
    st.stop()

raw_filtered = st.session_state.raw_intensities_filtered.copy()

cv_data = []
mean_cvs = {}
median_cvs = {}

for label, reps in [("Condition A", c1), ("Condition B", c2)]:
    reps = [r for r in reps if r in raw_filtered.columns]
    if len(reps) < 2:
        st.warning(f"{label} has <2 replicates — CV skipped")
        continue

    raw_vals = raw_filtered[reps]
    cv_per_protein = raw_vals.std(axis=1) / raw_vals.mean(axis=1) * 100
    cv_per_protein = cv_per_protein.replace([np.inf, -np.inf], np.nan).dropna()

    if cv_per_protein.empty:
        continue

    mean_cv = cv_per_protein.mean()
    median_cv = cv_per_protein.median()

    mean_cvs[label] = mean_cv
    median_cvs[label] = median_cv

    cv_data.extend([{"Condition": label, "CV (%)": v} for v in cv_per_protein])

col1, col2 = st.columns(2)
with col1:
    if "Condition A" in mean_cvs:
        st.metric("Condition A — Mean CV", f"{mean_cvs['Condition A']:.1f}%")
        st.metric("Condition A — Median CV", f"{median_cvs['Condition A']:.1f}%")
    else:
        st.metric("Condition A — Mean CV", "N/A")
        st.metric("Condition A — Median CV", "N/A")

with col2:
    if "Condition B" in mean_cvs:
        st.metric("Condition B — Mean CV", f"{mean_cvs['Condition B']:.1f}%")
        st.metric("Condition B — Median CV", f"{median_cvs['Condition B']:.1f}%")
    else:
        st.metric("Condition B — Mean CV", "N/A")
        st.metric("Condition B — Median CV", "N/A")

if cv_data:
    cv_df = pd.DataFrame(cv_data)
    fig = go.Figure()
    for label, color in [("Condition A", "#E71316"), ("Condition B", "#1f77b4")]:
        data_vals = cv_df[cv_df["Condition"] == label]["CV (%)"]
        if data_vals.empty:
            continue
        fig.add_trace(
            go.Violin(
                y=data_vals,
                name=label,
                line_color=color,
                meanline_visible=True,
                box=dict(visible=True, width=0.3),
                points=False,
            )
        )

    fig.update_layout(
        title="Coefficient of Variation (CV %) Within Conditions — Raw Intensities",
        yaxis_title="CV (%)",
        height=600,
        template="simple_white",
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------- 4. PROTEINS DETECTED PER SAMPLE PER SPECIES -----------------
st.subheader("Proteins Detected per Sample per Species")

if "Species" not in df_final.columns:
    st.warning("Species column not found; skipping species-level counts.")
else:
    # Count proteins with intensity > 1 (non-imputed)
    detected = (intensity_final[all_reps] > 1).astype(int)
    detected_df = detected.join(df_final[["Species"]])

    count_data = []
    for rep in all_reps:
        for species in detected_df["Species"].dropna().unique():
            sub = detected_df[detected_df["Species"] == species]
            count = int(sub[rep].sum())
            count_data.append(
                {"Sample": rep, "Species": species, "Proteins": count}
            )

    count_df = pd.DataFrame(count_data)
    if count_df.empty:
        st.warning("No non-imputed detections to count.")
    else:
        fig = px.bar(
            count_df,
            x="Sample",
            y="Proteins",
            color="Species",
            title="Number of Proteins Detected per Sample per Species",
            text="Proteins",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(textposition="inside", textfont_size=12)
        fig.update_layout(height=600, barmode="stack", template="simple_white")
        st.plotly_chart(fig, use_container_width=True)

# ----------------- FINAL SNAPSHOT INDEXED BY PG -----------------
if "PG" in df_final.columns:
    df_final_pg = df_final.set_index("PG").copy()
    intensity_pg = df_final_pg[all_reps]
else:
    df_final_pg = df_final.copy()
    intensity_pg = intensity_final.copy()

st.subheader("Final Processed Data (5-row snapshot)")
st.write("Index: Protein Group ID | Transformation: log₂ | Filtering: Applied")
st.dataframe(intensity_pg.head(5).round(3), use_container_width=True)

# ----------------- FINAL ACCEPT / EXPORT -----------------
if st.button("Complete Analysis & Export Results", type="primary"):
    st.success("Analysis complete! Ready for export / publication.")
    st.balloons()
