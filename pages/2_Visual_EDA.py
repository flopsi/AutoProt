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

# Build table using ProteinData structure
table_data = {}

for species in sorted(set(protein_data.species_mapping.values())):
    table_data[species] = {}
    
    for sample in protein_data.numeric_cols:
        # Mask: proteins of this species
        species_mask = protein_data.raw.index.map(
            lambda x: protein_data.species_mapping.get(x) == species
        )
        
        # Mask: valid intensities (≠ 1.0 and not NaN)
        valid_mask = (protein_data.raw[sample] != 1.0) & (protein_data.raw[sample].notna())
        
        # Count
        valid_count = (species_mask & valid_mask).sum()
        table_data[species][sample] = int(valid_count)

# Convert to DataFrame
df_valid = pd.DataFrame(table_data).T
df_valid.loc['Total'] = df_valid.sum()

# Display
st.dataframe(df_valid, use_container_width=True)

# Download
st.download_button(
    label="📥 Download Table",
    data=df_valid.to_csv(),
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
