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

st.info("**Valid = intensity ≠ 1.00**. Stacked bar chart shows composition by species.")

# Prepare data for viz helper: convert missing values (1.0) to NaN
df_for_viz = protein_data.raw[protein_data.numeric_cols].copy()

for col in protein_data.numeric_cols:
    df_for_viz.loc[df_for_viz[col] == 1.0, col] = np.nan

# Get theme
from helpers.core import get_theme
theme = get_theme("light")

# Use optimized viz helper to create stacked bar chart and summary
from helpers.viz import create_protein_count_stacked_bar

fig, summary_df = create_protein_count_stacked_bar(
    df_for_viz,
    protein_data.numeric_cols,
    protein_data.species_mapping,
    theme
)

# Display stacked bar chart
st.plotly_chart(fig, use_container_width=True)

# Display summary table
st.markdown("**Summary Statistics:**")
st.dataframe(summary_df, use_container_width=True, height=250)

# Download
st.download_button(
    label="📥 Download Summary (CSV)",
    data=summary_df.to_csv(index=False),
    file_name="protein_counts_summary.csv",
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
