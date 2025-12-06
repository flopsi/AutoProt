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
from helpers.viz import create_protein_count_stacked_bar
from helpers.core import get_theme
# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(page_title="Visual EDA", layout="wide")

# ============================================================================
# CHECK DATA LOADED
# ============================================================================

if "protein_data" not in st.session_state:
    st.warning("⚠️ No data loaded")
    if st.button("← Go to Upload"):
        st.switch_page("pages/1_Data_Upload.py")
    st.stop()

protein_data: ProteinData = st.session_state.protein_data
df_raw = protein_data.raw
numeric_cols = protein_data.numeric_cols
species_mapping = protein_data.species_mapping


st.info(f"📁 **{protein_data.file_path}** | {protein_data.n_proteins:,} proteins × {protein_data.n_samples} samples")
st.markdown("---")

st.title("📊 Visual Exploratory Data Analysis")



# ============================================================================
# SECTION 2: VALID PROTEINS PER SPECIES PER REPLICATE
# ============================================================================

st.subheader("2️⃣ Valid Proteins per Species per Sample")

st.info("**Valid = intensity ≠ 1.00** (missing value threshold). Each cell shows count of valid proteins.")

# Create table: species (rows) × samples (columns)
table_data = {}

for species in sorted(set(protein_data.species_mapping.values())):
    table_data[species] = {}
    
    for sample in protein_data.numeric_cols:
        # Get proteins of this species
        species_proteins = [
            pid for pid, sp in protein_data.species_mapping.items() 
            if sp == species
        ]
        
        # Count valid (intensity ≠ 1.0) for this species in this sample
        valid_count = 0
        for protein_id in species_proteins:
            if protein_id in protein_data.raw.index:
                intensity = protein_data.raw.loc[protein_id, sample]
                if pd.notna(intensity) and intensity != 1.0:
                    valid_count += 1
        
        table_data[species][sample] = valid_count

# Convert to DataFrame
df_valid = pd.DataFrame(table_data).T

# Add Total row
df_valid.loc['Total'] = df_valid.sum()

# Display table
st.dataframe(df_valid, use_container_width=True)

# Download
csv_data = df_valid.to_csv()
st.download_button(
    label="📥 Download Table",
    data=csv_data,
    file_name="valid_proteins_per_species.csv",
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
