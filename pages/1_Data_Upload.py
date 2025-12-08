"""
pages/1_Data_Upload.py - OPTIMIZED
Unified data upload interface for proteins and peptides with tab-based selection
Uses helpers for validation and data loading
FIXED: Ensures df_filtered is always defined before section 7️⃣ to prevent NameError.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import gc
import polars as pl

# Import helpers
#from helpers.io import detect_numeric_columns, convert_string_numbers_to_float
#from helpers.core import ProteinData, PeptideData
#from helpers.naming import rename_columns_for_display 
#from helpers.naming import standardize_condition_names 

# ============================================================================
# PAGE CONFIG
# ============================================================================
def init_session_state(key: str, default_value):
    """Initialize session state variable if not already set."""
    if key not in st.session_state:
        st.session_state[key] = default_value

# Ensure core variables are initialized
init_session_state("data_type", "protein")
init_session_state("protein_data", None)
init_session_state("peptide_data", None)
init_session_state("selected_data", None)
init_session_state("selected_columns", [])  # Track selected columns

# ============================================================================
# FILE UPLOAD
# ============================================================================

st.subheader("2️⃣ Upload File")

col1, col2 = st.columns([2, 1])
with col1:
    st.caption(f"Upload your {st.session_state.data_type} abundance data (CSV or Excel)")

with col2:
    st.caption("Supported: .csv, .xlsx, .xls")
    
peptides = st.toggle("Toggle if Peptide Data")

if peptides:
    st.session_state.data_type = "peptide"
else:
    st.session_state.data_type = "protein"

uploaded_file = st.file_uploader(
    f"Choose a {st.session_state.data_type} data file",
    type=["csv", "xlsx", "xls"],
    key=f"file_upload_{st.session_state.selected_data}"
)
    
if uploaded_file is None:
    st.info(f"👆 Upload a {st.session_state.data_type} abundance file to begin analysis")
    st.stop()

# ============================================================================
# LOAD FILE
# ============================================================================

try:
    with st.spinner(f"Loading {st.session_state.data_type} data..."):
        if uploaded_file.name.endswith('.csv'):
            df_raw = pl.read_csv(uploaded_file, has_header=True, null_values="#NUM!")
        else:
            df_raw = pl.read_excel(uploaded_file, sheet_id=0)
        
        st.success(f"✅ Loaded {len(df_raw):,} rows × {len(df_raw.columns)} columns")
except Exception as e:
    st.error(f"❌ Error loading file: {str(e)}")
    st.stop()

# ============================================================================
# SELECT COLUMNS
# ============================================================================

# ============================================================================
# SELECT COLUMNS - STEP 1: METADATA
# ============================================================================

st.subheader("3️⃣ Select Metadata Columns")
st.caption("Click headers to select ID, gene names, descriptions, etc.")

event_metadata = st.dataframe(
    df_raw,
    key="metadata_selector",
    on_select="rerun",
    selection_mode="multi-column",
)

metadata_cols = event_metadata.selection.columns

if metadata_cols:
    st.session_state.metadata_columns = metadata_cols
    st.success(f"✅ Selected {len(metadata_cols)} metadata column(s): {', '.join(metadata_cols)}")
else:
    st.info("👆 Select metadata columns first")
    st.stop()

# ============================================================================
# SELECT COLUMNS - STEP 2: NUMERICAL
# ============================================================================

st.subheader("4️⃣ Select Numerical Columns")
st.caption("Click headers to select abundance/intensity columns for analysis")

event_numerical = st.dataframe(
    df_raw,
    key="numerical_selector",
    on_select="rerun",
    selection_mode="multi-column",
)

numerical_cols = event_numerical.selection.columns

if numerical_cols:
    st.session_state.numerical_columns = numerical_cols
    st.success(f"✅ Selected {len(numerical_cols)} numerical column(s): {', '.join(numerical_cols)}")
    
    # Combine selections into working dataframe
    all_cols = metadata_cols + numerical_cols
    working_df = df_raw.select(all_cols)
    
    st.subheader("Working DataFrame")
    st.dataframe(working_df, use_container_width=True)
    
    # Store by data type
    if st.session_state.data_type == "protein":
        st.session_state.protein_data = working_df
    else:
        st.session_state.peptide_data = working_df
else:
    st.info("👆 Select numerical columns to create working dataframe")

# FOOTER
# ============================================================================

col1, col2 = st.columns([1, 1])

with col1:
    st.caption("💡 **Tip:** Ensure your data has a unique ID column and numeric intensity columns for each sample.")

with col2:
    st.caption("📖 **Format:** CSV or Excel files with proteins/peptides as rows and samples as columns.")
