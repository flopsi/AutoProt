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
from typing import Tuple, Optional
import gc

# Import helpers
from helpers.io import detect_numeric_columns, convert_string_numbers_to_float
from helpers.core import ProteinData, PeptideData
from helpers.naming import rename_columns_for_display 
from helpers.naming import standardize_condition_names 

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

# Ensure core variables are initialized to avoid key errors later
init_session_state("data_type", "protein")
init_session_state("protein_data", None)
init_session_state("peptide_data", None)





# ============================================================================
# FILE UPLOAD
# ============================================================================

st.subheader("2️⃣ Upload File")

col1, col2 = st.columns([2, 1])
with col1:
    st.caption(f"Upload your {st.session_state.data_type} abundance data (CSV or Excel)")

with col2:
    st.caption("Supported: .csv, .xlsx, .xls")

uploaded_file = st.file_uploader(
    f"Choose a {st.session_state.data_type} data file",
    type=["csv", "xlsx", "xls"],
    key=f"file_upload_{st.session_state.data_type}"
)
peptides = st.toggle("Peptide Data")
if peptides:
    st.session_state.data_type = 'peptide'
    st.rerun()
else:
    st.session_state.data_type = 'protein'
    st.rerun()
    
    
if uploaded_file is None:
    st.info(f"👆 Upload a {st.session_state.data_type} abundance file to begin analysis")
    st.stop()

# ============================================================================
# LOAD FILE
# ============================================================================

try:
    with st.spinner(f"Loading {st.session_state.data_type} data..."):
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file, index_col=None)
        else:
            df_raw = pd.read_excel(uploaded_file, sheet_name=0)
        
        st.success(f"✅ Loaded {len(df_raw):,} rows × {len(df_raw.columns)} columns")
except Exception as e:
    st.error(f"❌ Error loading file: {str(e)}")
    st.stop()

# ============================================================================
# DATA CLEANUP (FIX FOR '#NUM!')
# ============================================================================
try:
    # Replace common non-numeric string placeholders with NaN for correct numeric detection
    df_raw = df_raw.replace(['#NUM!', '#N/A', '#REF!', 'N/A', 'NA', ''])
    df_raw[np.isna(df_raw)]=1.0
    st.caption("🔍 Applied preliminary data cleanup (replaced common proteomics placeholders like `#NUM!` with 1.0)")
except Exception as e:
    st.warning(f"⚠️ Could not perform data cleanup: {str(e)}")

st.markdown("---")

# ============================================================================
# COLUMN CONFIGURATION
# ============================================================================

st.subheader("3️⃣ Configure Columns")

col1, col2, col3 = st.columns(3)

# ID Column selection
with col1:
    id_col = st.selectbox(
        "ID Column (protein/peptide identifier):",
        options=df_raw.columns,
        key=f"id_col_{st.session_state.data_type}"
    )

# Species/Taxonomy column selection
with col2:
    species_options = [None] + list(df_raw.columns)
    species_col = st.selectbox(
        "Species Column (optional):",
        options=species_options,
        key=f"species_col_{st.session_state.data_type}"
    )

# Peptide sequence column (only for peptides)
if st.session_state.data_type == 'peptide':
    with col3:
        sequence_col = st.selectbox(
            "Sequence Column (amino acids):",
            options=df_raw.columns,
            key="peptide_sequence_col"
        )
else:
    sequence_col = None

st.markdown("---")

# ============================================================================
# NUMERIC COLUMN DETECTION & CONVERSION
# ============================================================================

st.subheader("4️⃣ Select Numeric Columns (Abundance Data)")

numeric_cols, categorical_cols = detect_numeric_columns(df_raw)

# FIX: Convert string-formatted numbers to actual floats before filtering
df_raw = convert_string_numbers_to_float(df_raw, numeric_cols)
# END FIX

# Filter out ID and species columns from numeric candidates
exclude_cols = {id_col, species_col, sequence_col}
numeric_cols = [c for c in numeric_cols if c not in exclude_cols and c is not None]

col1, col2 = st.columns([3, 1])
with col1:
    st.caption(f"Auto-detected {len(numeric_cols)} numeric columns")

with col2:
    select_all = st.checkbox("Select all", value=True, key="select_all")

if numeric_cols:
    selected_numeric = st.multiselect(
        "Numeric columns to include:",
        options=numeric_cols,
        default=numeric_cols if select_all else [],
        key=f"numeric_cols_{st.session_state.data_type}"
    )
else:
    st.error("❌ No numeric columns detected. Check your data format.")
    st.stop()

if not selected_numeric:
    st.error("❌ Select at least one numeric column")
    st.stop()

# ============================================================================
# **FIX: INITIALIZE df_filtered**
# Ensure df_filtered is defined before conditional logic in section 6
# ============================================================================
df_filtered = df_raw.copy()
# ============================================================================

st.markdown("---")

# ============================================================================
# DATA VALIDATION & PREVIEW
# ============================================================================

st.subheader("5️⃣ Data Validation & Preview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Rows",
        f"{len(df_raw):,}",
        help="Number of proteins/peptides"
    )

with col2:
    st.metric(
        "Samples",
        len(selected_numeric),
        help="Number of abundance columns"
    )

with col3:
    missing_pct = (df_raw[selected_numeric].isna().sum().sum() / 
                   (len(df_raw) * len(selected_numeric)) * 100)
    st.metric(
        "Missing %",
        f"{missing_pct:.1f}%",
        help="Overall missing value rate"
    )

with col4:
    st.metric(
        "Data Type",
        st.session_state.data_type.capitalize(),
        help=f"Processing as {st.session_state.data_type} data"
    )

# Show preview
with st.expander("📊 Data Preview", expanded=False):
    preview_rows = st.slider("Preview rows:", 5, min(50, len(df_raw)), 10)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df_raw.iloc[:preview_rows], use_container_width=True, height=400)
    
    with col2:
        st.write("**Column Info:**")
        col_info = pd.DataFrame({
            'Column': df_raw.columns,
            'Type': df_raw.dtypes.astype(str),
            'Non-Null': df_raw.notna().sum().values
        })
        st.dataframe(col_info, use_container_width=True, height=400)

st.markdown("---")

# ============================================================================
# DATA FILTERING OPTIONS
# ============================================================================

st.subheader("6️⃣ Optional Filtering")

col1, col2, col3 = st.columns(3)

with col1:
    min_intensity = st.number_input(
        "Minimum intensity threshold:",
        min_value=0.0,
        value=1.0,
        step=0.1,
        help="Filter proteins/peptides below this value"
    )

with col2:
    max_missing = st.slider(
        "Maximum missing % per row:",
        min_value=0,
        max_value=100,
        value=100,
        step=5,
        help="Remove rows exceeding this missing rate"
    )

with col3:
    st.write("")  # Spacing
    apply_filtering = st.checkbox("Apply filtering", value=False)

if apply_filtering:
    # Filter by missing rate
    missing_rates = df_filtered[selected_numeric].isna().sum(axis=1) / len(selected_numeric) * 100
    df_filtered = df_filtered[missing_rates <= max_missing]
    
    # Filter by minimum intensity
    has_valid = (df_filtered[selected_numeric] >= min_intensity).any(axis=1)
    df_filtered = df_filtered[has_valid]
    
    rows_removed = len(df_raw) - len(df_filtered)
    st.info(f"✓ Filtering applied: {rows_removed:,} rows removed, {len(df_filtered):,} rows remaining")
else:
    # If not applying filtering, df_filtered remains the initialized raw copy.
    pass 

st.markdown("---")

# ============================================================================
# COLUMN RENAMING & FINAL DROPPING
# ============================================================================

st.subheader("7️⃣ Renaming & Final Column Selection")

st.info("The app automatically suggests **Condition_R#** names for replicates. Review and adjust below.")

# 1. Generate the initial, automatic mapping using 'smart' condition logic
initial_name_mapping = standardize_condition_names(selected_numeric)

st.write(f"**Auto-suggested Renaming for {len(selected_numeric)} Samples:**")

# 2. Manual Override Interface
manual_mapping = {}
col_chunks = st.columns(3) # Use columns for a compact layout

with st.expander("📝 Review/Edit Sample Names", expanded=True):
    # Create the manual input fields
    for i, (original_col, suggested_name) in enumerate(initial_name_mapping.items()):
        
        # Use a consistent key for the text input
        key_id = f"rename_{original_col}_{i}" 

        # Display input in one of the three columns
        with col_chunks[i % 3]:
            # Use text_input to allow manual override
            new_name = st.text_input(
                label=f"Original: `{original_col}`",
                value=suggested_name,
                key=key_id,
                help="Edit the new name (Condition_R# format recommended)"
            )
            
            # Store the user's final decision (or the auto-suggested name)
            manual_mapping[original_col] = new_name.strip()


# 3. Apply the final, user-approved mapping to the DataFrame

# Create a clean final mapping for use in the DataFrame operations
final_mapping = {
    original_col: manual_name 
    for original_col, manual_name in manual_mapping.items() 
    if original_col in df_filtered.columns # Only rename columns that still exist (after filtering/selection)
}

# Apply mapping to the DataFrame
df_filtered = df_filtered.rename(columns=final_mapping)

# Update selected numeric columns list with the new, shorter names
selected_numeric_renamed = list(final_mapping.values())

# Since ID/Species columns were not in the mapping keys, they retain their original name
id_col_renamed = id_col
species_col_renamed = species_col
sequence_col_renamed = sequence_col

if selected_numeric_renamed:
    st.caption(f"**Renamed Columns:** {len(final_mapping)} numeric columns were changed. The first column name is now **'{selected_numeric_renamed[0]}'**.")

# --- Drop Columns ---
# Determine final columns to keep
final_cols = [id_col_renamed] + selected_numeric_renamed
if species_col_renamed:
    final_cols.append(species_col_renamed)
if st.session_state.data_type == 'peptide' and sequence_col_renamed:
    final_cols.append(sequence_col_renamed)

# Drop columns not in the final list
dropped_cols = [col for col in df_filtered.columns if col not in final_cols]

if dropped_cols:
    df_filtered = df_filtered[final_cols]
    st.success(f"✅ Dropped {len(dropped_cols)} unused columns.")
else:
    st.info("No extra columns to drop.")

# Update the variables used in the final step with the renamed versions
selected_numeric = selected_numeric_renamed
id_col = id_col_renamed
species_col = species_col_renamed
sequence_col = sequence_col_renamed

st.markdown("---")

# ============================================================================
# VALIDATION & CONFIRMATION
# ============================================================================

st.subheader("8️⃣ Validate & Upload")

col1, col2 = st.columns(2)

with col1:
    st.write("**Validation Checks:**")
    
    checks = {
        "✅ ID column selected": id_col is not None,
        "✅ Numeric columns selected": len(selected_numeric) > 0,
        "✅ Data loaded": df_filtered is not None,
        "✅ Samples available": len(df_filtered) > 0,
    }
    
    if st.session_state.data_type == 'peptide':
        checks["✅ Sequence column selected"] = sequence_col is not None
    else:
        checks["✅ Species/taxonomy configured"] = True
    
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
    st.write(f"- **Samples:** {len(selected_numeric)}")
    st.write(f"- **Total {st.session_state.data_type.title()}s:** {len(df_filtered):,}")

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
            # Create data object
            if st.session_state.data_type == 'protein':
                data_obj = ProteinData(
                    raw=df_filtered,
                    numeric_cols=selected_numeric,
                    id_col=id_col, 
                    species_col=species_col, 
                    file_path=str(uploaded_file.name)
                )
                st.session_state.protein_data = data_obj
                
                # Store in session for other pages
                st.session_state.df_raw = df_filtered
                st.session_state.numeric_cols = selected_numeric
                st.session_state.id_col = id_col
                st.session_state.species_col = species_col
                st.session_state.data_ready = True
                
            else:  # peptide
                data_obj = PeptideData(
                    raw=df_filtered,
                    numeric_cols=selected_numeric,
                    id_col=id_col, 
                    species_col=species_col, 
                    sequence_col=sequence_col, 
                    file_path=str(uploaded_file.name)
                )
                st.session_state.peptide_data = data_obj
                
                # Store in session for other pages
                st.session_state.df_raw = df_filtered
                st.session_state.numeric_cols = selected_numeric
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
    st.caption("💡 **Tip:** Ensure your data has a unique ID column and numeric intensity columns for each sample.")

with col2:
    st.caption("📖 **Format:** CSV or Excel files with proteins/peptides as rows and samples as columns.")
