# pages/3_Statistical_EDA.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Statistical EDA", layout="wide")

st.title("📈 Statistical EDA: Species & Variability")

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
protein_data = st.session_state.get("protein_data")
if protein_data is None:
    st.error("No data found. Please complete the EDA page first.")
    st.stop()

df: pd.DataFrame = protein_data.raw
numeric_cols = protein_data.numeric_cols
species_col = protein_data.species_col
species_mapping = protein_data.species_mapping

if not numeric_cols or species_col is None or not species_mapping:
    st.error("Missing species information or numeric columns.")
    st.stop()

st.success(
    f"{len(df):,} proteins × {len(numeric_cols)} samples | "
    f"Transform: **{st.session_state.get('selected_transform_method', 'raw')}**"
)

# ----------------------------------------------------------------------
# 1) Configuration: Assign samples to conditions
# ----------------------------------------------------------------------
st.subheader("1️⃣ Configuration")

n_cols = len(numeric_cols)
mid = n_cols // 2

col1, col2 = st.columns(2)
with col1:
    cond_a_cols = st.multiselect(
        "Condition A samples",
        options=numeric_cols,
        default=numeric_cols[:mid],
        key="stat_eda_cond_a",
    )
with col2:
    cond_b_cols = st.multiselect(
        "Condition B samples",
        options=numeric_cols,
        default=numeric_cols[mid:],
        key="stat_eda_cond_b",
    )

if not cond_a_cols or not cond_b_cols:
    st.warning("Select at least one sample for each condition.")
    st.stop()

all_samples = cond_a_cols + cond_b_cols

# ----------------------------------------------------------------------
# 2) Species composition per sample (stacked bar chart)
# ----------------------------------------------------------------------
st.subheader("2️⃣ Species Composition per Sample")

# Count proteins per species per sample (presence/absence or intensity-weighted)
species_counts = []

for sample in all_samples:
    # For each sample, count how many proteins of each species are detected (non-zero/non-NaN)
    sample_data = df[[species_col, sample]].copy()
    sample_data = sample_data[sample_data[sample].notna() & (sample_data[sample] > 0)]
    
    counts = sample_data[species_col].value_counts().to_dict()
    
    species_counts.append({
        "Sample": sample,
        "HUMAN": counts.get("HUMAN", 0),
        "YEAST": counts.get("YEAST", 0),
        "ECOLI": counts.get("ECOLI", 0),
    })

species_df = pd.DataFrame(species_counts)

# Stacked bar chart
fig_species = go.Figure()

fig_species.add_trace(go.Bar(
    name='HUMAN',
    x=species_df['Sample'],
    y=species_df['HUMAN'],
    marker_color='#1f77b4',
))

fig_species.add_trace(go.Bar(
    name='YEAST',
    x=species_df['Sample'],
    y=species_df['YEAST'],
    marker_color='#ff7f0e',
))

fig_species.add_trace(go.Bar(
    name='ECOLI',
    x=species_df['Sample'],
    y=species_df['ECOLI'],
    marker_color='#2ca02c',
))

fig_species.update_layout(
    barmode='stack',
    title="Detected Proteins per Species per Sample",
    xaxis_title="Sample",
    yaxis_title="Number of Proteins",
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig_species, use_container_width=True)

# Summary table
st.dataframe(species_df.set_index("Sample"), use_container_width=True)

# ----------------------------------------------------------------------
# 3) Within-condition CV (Coefficient of Variation) violin plots
# ----------------------------------------------------------------------
st.subheader("3️⃣ Within-Condition Variability (CV)")

st.markdown(
    """
    **Coefficient of Variation (CV)** = (std / mean) × 100%  
    Lower CV indicates more consistent measurements within a condition.
    """
)

# Compute CV per protein per condition
def compute_cv(row, cols):
    vals = row[cols]
    vals = vals[vals > 0]  # exclude zeros/negatives
    if len(vals) < 2:
        return np.nan
    return (vals.std() / vals.mean()) * 100 if vals.mean() > 0 else np.nan

df['CV_CondA'] = df.apply(lambda row: compute_cv(row, cond_a_cols), axis=1)
df['CV_CondB'] = df.apply(lambda row: compute_cv(row, cond_b_cols), axis=1)

cv_a = df['CV_CondA'].dropna()
cv_b = df['CV_CondB'].dropna()

# Violin plot
fig_cv = go.Figure()

fig_cv.add_trace(go.Violin(
    y=cv_a,
    name='Condition A',
    box_visible=True,
    meanline_visible=True,
    fillcolor='#1f77b4',
    opacity=0.6,
    x0='Condition A',
))

fig_cv.add_trace(go.Violin(
    y=cv_b,
    name='Condition B',
    box_visible=True,
    meanline_visible=True,
    fillcolor='#ff7f0e',
    opacity=0.6,
    x0='Condition B',
))

fig_cv.update_layout(
    title="Coefficient of Variation (CV) Distribution per Condition",
    yaxis_title="CV (%)",
    xaxis_title="Condition",
    height=500,
    showlegend=False,
)

st.plotly_chart(fig_cv, use_container_width=True)

# Summary stats
col_stats1, col_stats2 = st.columns(2)

with col_stats1:
    st.metric("Condition A - Median CV", f"{cv_a.median():.1f}%")
    st.metric("Condition A - Mean CV", f"{cv_a.mean():.1f}%")

with col_stats2:
    st.metric("Condition B - Median CV", f"{cv_b.median():.1f}%")
    st.metric("Condition B - Mean CV", f"{cv_b.mean():.1f}%")

# Optional: CV per species
with st.expander("View CV by Species"):
    st.markdown("### CV Distribution by Species")
    
    for species in ["HUMAN", "YEAST", "ECOLI"]:
        species_proteins = df[df[species_col] == species]
        
        if len(species_proteins) == 0:
            continue
        
        cv_a_sp = species_proteins['CV_CondA'].dropna()
        cv_b_sp = species_proteins['CV_CondB'].dropna()
        
        st.markdown(f"#### {species}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"Cond A - Median CV ({species})", f"{cv_a_sp.median():.1f}%" if len(cv_a_sp) > 0 else "N/A")
        with col2:
            st.metric(f"Cond B - Median CV ({species})", f"{cv_b_sp.median():.1f}%" if len(cv_b_sp) > 0 else "N/A")
