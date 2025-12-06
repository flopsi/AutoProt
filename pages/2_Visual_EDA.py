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

st.info("**Valid = intensity ≠ 1.00**. Each cell shows count of valid proteins.")

# DEBUG
st.write("**DEBUG INFO:**")
st.write(f"Raw index (first 5): {protein_data.raw.index[:5].tolist()}")
st.write(f"Species mapping (first 5): {list(protein_data.species_mapping.items())[:5]}")
st.write(f"Numeric cols: {protein_data.numeric_cols}")
st.write(f"Sample data A1: {protein_data.raw[protein_data.numeric_cols[0]].head()}")
st.write(f"Value types: {protein_data.raw[protein_data.numeric_cols[0]].dtype}")
st.write(f"Unique values in A1: {protein_data.raw[protein_data.numeric_cols[0]].unique()[:20]}")

# Build table from ProteinData
table_data = {}

# Get unique species from species_mapping
unique_species = sorted(set(protein_data.species_mapping.values()))

for species in unique_species:
    table_data[species] = {}
    
    for sample in protein_data.numeric_cols:
        # Get protein IDs for this species
        proteins_in_species = [
            protein_id for protein_id, sp in protein_data.species_mapping.items()
            if sp == species
        ]
        
        # Count valid intensities (≠ 1.0, not NaN) in this sample for this species
        valid_count = 0
        for protein_id in proteins_in_species:
            intensity = protein_data.raw.loc[protein_id, sample]
            if pd.notna(intensity) and intensity != 1.0:
                valid_count += 1
        
        table_data[species][sample] = valid_count

# Convert to DataFrame
df_valid = pd.DataFrame(table_data).T
df_valid.loc['Total'] = df_valid.sum()

# Display
st.dataframe(df_valid, use_container_width=True)
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
