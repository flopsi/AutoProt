"""
pages/2_Visual_EDA.py

Visual Exploratory Data Analysis
- Data quality assessment
- Per-species protein counts
- Missing value distribution analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from helpers.core import ProteinData
from helpers.analysis import (
    count_valid_proteins_per_species_sample,
    count_missing_per_protein,
    count_proteins_by_species
)
from helpers.audit import log_event

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(page_title="Visual EDA", layout="wide")

# ============================================================================
# CHECK DATA LOADED
# ============================================================================

if "protein_data" not in st.session_state or not st.session_state.get("data_locked"):
    st.warning("⚠️ No data loaded. Please upload data on the **Data Upload** page first.")
    st.stop()

# Load cached protein data
protein_data: ProteinData = st.session_state.protein_data

st.title("📊 Visual Exploratory Data Analysis")

st.info(f"""
**Loaded Data:**
- **File**: {protein_data.file_path} ({protein_data.file_format})
- **Proteins**: {protein_data.n_proteins:,}
- **Samples**: {protein_data.n_samples}
- **Conditions**: {protein_data.n_conditions}
""")

# ============================================================================
# SECTION 1: TOTAL PROTEINS
# ============================================================================

st.subheader("1️⃣ Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Total Proteins", f"{protein_data.n_proteins:,}")

with c2:
    st.metric("Total Samples", protein_data.n_samples)

with c3:
    species_count = len(set(protein_data.species_mapping.values()))
    st.metric("Species", species_count)

with c4:
    proteins_per_species = count_proteins_by_species(
        protein_data.raw, 
        protein_data.species_mapping
    )
    avg_per_species = int(protein_data.n_proteins / species_count) if species_count > 0 else 0
    st.metric("Avg Proteins/Species", avg_per_species)

# ============================================================================
# SECTION 2: VALID PROTEINS PER SPECIES PER REPLICATE
# ============================================================================

st.subheader("2️⃣ Valid Proteins per Species per Sample")

st.info("**Valid = intensity ≠ 1.00** (missing value threshold). Shows data completeness by species.")

# Calculate valid proteins per species per sample
valid_per_sample = count_valid_proteins_per_species_sample(
    protein_data.raw,
    protein_data.numeric_cols,
    protein_data.species_mapping,
    missing_value==1.0
)

# Display as table with simple formatting (no gradient - no matplotlib dependency)
st.dataframe(
    valid_per_sample.style.format("{:,}"),
    use_container_width=True
)

# Download option
csv = valid_per_sample.to_csv()
st.download_button(
    label="📥 Download Valid Counts (CSV)",
    data=csv,
    file_name="valid_proteins_per_species.csv",
    mime="text/csv"
)

# ============================================================================
# SECTION 3: MISSING VALUE DISTRIBUTION BY SPECIES (BOXPLOT)
# ============================================================================

st.subheader("3️⃣ Missing Value Distribution by Species")

st.info("**Distribution of missing count per protein**: Shows how many samples have intensity = 1.00 for each protein, grouped by species.")

# Calculate missing values per protein (one row per protein)
missing_per_protein_df = count_missing_per_protein(
    protein_data.raw,
    protein_data.numeric_cols,
    protein_data.species_mapping,
    missing_value=1.0
)

# Ensure unique proteins only (no duplicates)
missing_per_protein_df = missing_per_protein_df.drop_duplicates(subset=['protein_id'])

# Create boxplot
fig = go.Figure()

# Species order: HUMAN (first/largest), YEAST, ECOLI
species_order = ["HUMAN", "YEAST", "ECOLI"]
species_in_data = sorted(missing_per_protein_df['species'].unique())

# Use defined order, but only include species in data
ordered_species = [s for s in species_order if s in species_in_data]

# Color scheme: progressively lighter/different for stacking visual
color_map = {
    "HUMAN": "#1f77b4",   # Blue (bottom/base)
    "YEAST": "#ff7f0e",   # Orange (middle)
    "ECOLI": "#2ca02c"    # Green (top)
}

for species in ordered_species:
    species_data = missing_per_protein_df[missing_per_protein_df['species'] == species]
    missing_counts = species_data['missing_count'].values
    
    fig.add_trace(go.Box(
        y=missing_counts,
        name=species,
        marker=dict(color=color_map.get(species, "#808080")),
        boxmean='sd',  # Show mean and std dev
        jitter=0.3,
        pointpos=-1.8,
        hovertemplate=f"{species}<br>Missing Count: %{{y}}<extra></extra>"
    ))

fig.update_layout(
    title="Distribution of Missing Values per Protein by Species",
    yaxis_title="Number of Missing Values (intensity = 1.00)",
    xaxis_title="Species",
    height=500,
    showlegend=True,
    hovermode="closest",
    plot_bgcolor="rgba(240,240,240,0.5)",
    yaxis=dict(gridcolor="rgba(200,200,200,0.3)"),
    font=dict(size=11)
)

st.plotly_chart(fig, use_container_width=True)

# Statistics table (formatted without matplotlib gradient)
st.markdown("**Summary Statistics by Species:**")

stats_by_species = missing_per_protein_df.groupby('species')['missing_count'].agg([
    ('Count', 'count'),
    ('Mean', 'mean'),
    ('Median', 'median'),
    ('Std Dev', 'std'),
    ('Min', 'min'),
    ('Max', 'max')
]).round(2)

# Simple formatting without gradient
st.dataframe(
    stats_by_species.style.format("{:.2f}"),
    use_container_width=True
)

# ============================================================================
# SECTION 4: DATA SUMMARY EXPORT
# ============================================================================

st.subheader("4️⃣ Export Data Quality Report")

# Create summary report (count each protein once)
unique_protein_count = missing_per_protein_df['protein_id'].nunique()
mean_missing = missing_per_protein_df['missing_count'].mean()
median_missing = missing_per_protein_df['missing_count'].median()

report_data = {
    "Metric": [
        "Total Proteins",
        "Total Samples",
        "Number of Species",
        "Missing Value Threshold",
        "Mean Missing Count (all proteins)",
        "Median Missing Count (all proteins)"
    ],
    "Value": [
        unique_protein_count,
        protein_data.n_samples,
        len(set(protein_data.species_mapping.values())),
        "1.00",
        f"{mean_missing:.2f}",
        f"{median_missing:.0f}"
    ]
}

report_df = pd.DataFrame(report_data)

col1, col2 = st.columns(2)

with col1:
    st.download_button(
        label="📊 Download Quality Report (CSV)",
        data=report_df.to_csv(index=False),
        file_name="data_quality_report.csv",
        mime="text/csv"
    )

with col2:
    st.download_button(
        label="📋 Download Missing Value Details (CSV)",
        data=missing_per_protein_df.to_csv(index=False),
        file_name="missing_values_per_protein.csv",
        mime="text/csv"
    )


# ============================================================================
# Navigation
# ============================================================================

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    if st.button("← Back to Upload", use_container_width=True):
        st.switch_page("pages/1_Data_Upload.py")

with col2:
    st.info("**Next:** Statistical transformation & differential expression analysis (coming soon)")
