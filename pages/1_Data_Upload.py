"""
pages/1_Data_Upload.py - OPTIMIZED
Unified data upload interface with interactive column selection using Polars
"""

import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import gc


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Data Upload",
    page_icon="📥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📥 Data Upload")
st.markdown("Upload protein or peptide abundance data for analysis")

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state(key: str, default_value):
    """Initialize session state variable if not already set."""
    if key not in st.session_state:
        st.session_state[key] = default_value

init_session_state('data_type', 'protein')
init_session_state('protein_data', None)
init_session_state('peptide_data', None)
init_session_state('metadata_columns', [])
init_session_state('numerical_columns', [])
init_session_state('column_mapping', {})
init_session_state('reverse_mapping', {})

# ============================================================================
# FILE UPLOAD
# ============================================================================

st.subheader("1️⃣ Upload File")

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
    key=f"file_upload_{st.session_state.data_type}"
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
            df_raw = pl.read_csv(uploaded_file, has_header=True, null_values=["#NUM!"])
        else:
            df_raw = pl.read_excel(uploaded_file, sheet_id=0)
        
        st.success(f"✅ Loaded {len(df_raw):,} rows × {len(df_raw.columns)} columns")
except Exception as e:
    st.error(f"❌ Error loading file: {str(e)}")
    st.stop()

st.markdown("---")

# ============================================================================
# SELECT COLUMNS - STEP 1: METADATA
# ============================================================================

st.subheader("2️⃣ Select Metadata Columns")
st.caption("Click column headers to select ID, gene names, descriptions, species, etc.")

# Convert to pandas for display in st.dataframe
df_preview = df_raw.head(5).to_pandas()

event_metadata = st.dataframe(
    df_preview,
    key="metadata_selector",
    on_select="rerun",
    selection_mode="multi-column",
)

metadata_cols = event_metadata.selection.columns

if metadata_cols:
    st.session_state.metadata_columns = metadata_cols
    st.success(f"✅ Selected {len(metadata_cols)} metadata column(s): {', '.join(metadata_cols)}")
else:
    st.info("👆 Select metadata columns (ID, species, descriptions, etc.)")
    st.stop()

st.markdown("---")

# ============================================================================
# SELECT COLUMNS - STEP 2: NUMERICAL
# ============================================================================

st.subheader("3️⃣ Select Numerical Columns")
st.caption("Click column headers to select abundance/intensity columns for analysis")

event_numerical = st.dataframe(
    df_preview,
    key="numerical_selector",
    on_select="rerun",
    selection_mode="multi-column",
)

numerical_cols = event_numerical.selection.columns

if not numerical_cols:
    st.info("👆 Select numerical abundance columns")
    st.stop()

st.session_state.numerical_columns = numerical_cols
st.success(f"✅ Selected {len(numerical_cols)} numerical column(s): {', '.join(numerical_cols)}")

st.markdown("---")

# ============================================================================
# RENAME NUMERICAL COLUMNS
# ============================================================================

st.subheader("3.5️⃣ Rename Numerical Columns (Optional)")

col1, col2 = st.columns([1, 3])

with col1:
    rename_style = st.selectbox(
        "Renaming style:",
        options=['none', 'trim', 'clean', 'smart', 'short'],
        help="Choose how to clean/shorten column names"
    )

with col2:
    if rename_style != 'none':
        # Use pandas for rename_columns_for_display helper
        df_temp_pandas = df_raw.select(numerical_cols).to_pandas()
        df_preview_renamed, name_mapping = rename_columns_for_display(
            df_temp_pandas, 
            list(numerical_cols), 
            style=rename_style
        )
        
        # Show preview of mapping
        mapping_df = pd.DataFrame({
            'Original': list(name_mapping.keys()),
            'Renamed': list(name_mapping.values())
        })
        st.dataframe(mapping_df, use_container_width=True, height=200)
        
        # Store mapping
        st.session_state.column_mapping = name_mapping
        st.session_state.reverse_mapping = reverse_name_mapping(name_mapping)
        
        # Apply renaming
        numerical_cols_renamed = [name_mapping.get(col, col) for col in numerical_cols]
    else:
        name_mapping = {col: col for col in numerical_cols}
        numerical_cols_renamed = list(numerical_cols)
        st.session_state.column_mapping = name_mapping
        st.session_state.reverse_mapping = name_mapping

st.markdown("---")

# ============================================================================
# COMBINE AND CLEAN DATA
# ============================================================================

# Combine selections
all_cols = list(metadata_cols) + list(numerical_cols)
df_filtered = df_raw.select(all_cols)

# Apply column renaming to numerical columns only
df_filtered = df_filtered.rename(name_mapping)
numerical_cols_final = numerical_cols_renamed

# Clean and cast numerical columns to Float64
for col in numerical_cols_final:
    df_filtered = df_filtered.with_columns([
        # Convert to Float64, handling errors
        pl.col(col).cast(pl.Float64, strict=False)
        # Replace 0 and 1.0 with 1.00
        .replace([0.0, 1.0], 1.00)
        # Fill nulls with 1.00
        .fill_null(1.00)
        .alias(col)
    ])

st.info("✓ Cleaned numerical columns: NaN, 0, 1.0, and #NUM! values set to 1.00 (Float64)")

st.markdown("---")

# ============================================================================
# COLUMN CONFIGURATION
# ============================================================================

st.subheader("4️⃣ Configure Key Columns")

col1, col2, col3 = st.columns(3)

# ID Column selection (from metadata)
with col1:
    id_col = st.selectbox(
        "ID Column (required):",
        options=metadata_cols,
        key=f"id_col_{st.session_state.data_type}"
    )

# Species column (from metadata)
with col2:
    species_options = [None] + list(metadata_cols)
    species_col = st.selectbox(
        "Species Column (optional):",
        options=species_options,
        key=f"species_col_{st.session_state.data_type}"
    )

# Peptide sequence column (only for peptides, from metadata)
if st.session_state.data_type == 'peptide':
    with col3:
        sequence_col = st.selectbox(
            "Sequence Column:",
            options=metadata_cols,
            key="sequence_col"
        )
else:
    sequence_col = None

st.markdown("---")

# ============================================================================
# DATA VALIDATION & PREVIEW
# ============================================================================

st.subheader("5️⃣ Data Preview & Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Rows",
        f"{len(df_filtered):,}",
        help="Number of proteins/peptides"
    )

with col2:
    st.metric(
        "Samples",
        len(numerical_cols_final),
        help="Number of abundance columns"
    )

with col3:
    # Count 1.00 values (cleaned)
    cleaned_count = sum([
        (df_filtered[col] == 1.00).sum() for col in numerical_cols_final
    ])
    total_values = len(df_filtered) * len(numerical_cols_final)
    cleaned_pct = (cleaned_count / total_values) * 100
    st.metric(
        "Cleaned %",
        f"{cleaned_pct:.1f}%",
        help="Values set to 1.00 (originally NaN, 0, or 1.0)"
    )

with col4:
    st.metric(
        "Data Type",
        st.session_state.data_type.capitalize(),
        help=f"Processing as {st.session_state.data_type} data"
    )

# Show preview
with st.expander("📊 Data Preview", expanded=False):
    preview_rows = st.slider("Preview rows:", 5, min(50, len(df_filtered)), 10)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df_filtered.head(preview_rows).to_pandas(), use_container_width=True, height=400)
    
    with col2:
        st.write("**Column Info:**")
        col_info = pd.DataFrame({
            'Column': df_filtered.columns,
            'Type': [str(dtype) for dtype in df_filtered.dtypes],
            'Non-Null': [df_filtered[col].null_count() == 0 for col in df_filtered.columns]
        })
        st.dataframe(col_info, use_container_width=True, height=400)

st.markdown("---")

# ============================================================================
# VALIDATION & CONFIRMATION
# ============================================================================

st.subheader("6️⃣ Validate & Upload")

col1, col2 = st.columns(2)

with col1:
    st.write("**Validation Checks:**")
    
    checks = {
        "✅ Metadata columns selected": len(metadata_cols) > 0,
        "✅ Numerical columns selected": len(numerical_cols_final) > 0,
        "✅ ID column configured": id_col is not None,
        "✅ Data loaded": df_filtered is not None,
        "✅ Samples available": len(df_filtered) > 0,
        "✅ Numerical columns are Float64": all(df_filtered[col].dtype == pl.Float64 for col in numerical_cols_final),
    }
    
    if st.session_state.data_type == 'peptide':
        checks["✅ Sequence column selected"] = sequence_col is not None
    
    all_passed = all(checks.values())
    
    for check, status in checks.items():
        if status:
            st.success(check)
        else:
            st.error(check.replace("✅", "❌"))

with col2:
    st.write("**Summary:**")
    st.write(f"- **Type:** {st.session_state.data_type.upper()}")
    st.write(f"- **ID Column:** {id_col}")
    st.write(f"- **Species Column:** {species_col if species_col else 'None'}")
    if st.session_state.data_type == 'peptide':
        st.write(f"- **Sequence Column:** {sequence_col}")
    st.write(f"- **Metadata Columns:** {len(metadata_cols)}")
    st.write(f"- **Samples:** {len(numerical_cols_final)}")
    st.write(f"- **Total {st.session_state.data_type.title()}s:** {len(df_filtered):,}")
    st.write(f"- **Numeric dtype:** Float64")

st.markdown("---")

# ============================================================================
# UPLOAD BUTTON
# ============================================================================

if st.button(
    f"🚀 Upload {st.session_state.data_type.upper()} Data",
    type="primary",
    use_container_width=True,
    disabled=not all_passed
):
    with st.spinner(f"Processing {st.session_state.data_type} data..."):
        try:
            # Create data object (convert to pandas if needed by helpers)
            df_final_pandas = df_filtered.to_pandas()
            
            if st.session_state.data_type == 'protein':
                data_obj = ProteinData(
                    raw=df_final_pandas,
                    numeric_cols=numerical_cols_final,
                    id_col=id_col,
                    species_col=species_col,
                    file_path=str(uploaded_file.name)
                )
                st.session_state.protein_data = data_obj
                
                # Store in session for other pages (both formats)
                st.session_state.df_raw = df_final_pandas
                st.session_state.df_raw_polars = df_filtered
                st.session_state.numeric_cols = numerical_cols_final
                st.session_state.id_col = id_col
                st.session_state.species_col = species_col
                st.session_state.data_ready = True
                
            else:  # peptide
                data_obj = PeptideData(
                    raw=df_final_pandas,
                    numeric_cols=numerical_cols_final,
                    id_col=id_col,
                    species_col=species_col,
                    sequence_col=sequence_col,
                    file_path=str(uploaded_file.name)
                )
                st.session_state.peptide_data = data_obj
                
                # Store in session for other pages (both formats)
                st.session_state.df_raw = df_final_pandas
                st.session_state.df_raw_polars = df_filtered
                st.session_state.numeric_cols = numerical_cols_final
                st.session_state.id_col = id_col
                st.session_state.species_col = species_col
                st.session_state.sequence_col = sequence_col
                st.session_state.data_ready = True
            
            # Cleanup memory
            gc.collect()
            
            st.success(f"✅ {st.session_state.data_type.upper()} data uploaded successfully!")
            st.info(f"Ready for analysis. Go to **Visual EDA** page to continue.")
            
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")

st.markdown("---")

# ============================================================================
# FOOTER
# ============================================================================

col1, col2 = st.columns([1, 1])

with col1:
    st.caption("💡 **Tip:** First select metadata columns (ID, species, etc.), then numerical abundance columns.")

with col2:
    st.caption("📖 **Note:** NaN, 0, 1.0, and #NUM! values in numerical columns are automatically set to 1.00 (Float64).")
