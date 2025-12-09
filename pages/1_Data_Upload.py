"""
pages/1_Data_Upload.py - Data Upload & Configuration (FULLY FIXED)

Handles file upload, validation, column detection, and initial configuration
Integrates with helpers.io, helpers.core, helpers.analysis modules
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Import helper modules
from helpers.io import (
    load_csv, load_excel, detect_numeric_columns,
    validate_dataframe, check_duplicates, check_missing_data,
    get_data_summary, convert_string_numbers_to_float
)
from helpers.core import ProteinData, PeptideData
from helpers.analysis import detect_conditions_from_columns, create_condition_mapping

# ============================================================================
# LOGGER SETUP
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Data Upload - AutoProt",
    page_icon="📁",
    layout="wide"
)

st.title("📁 Data Upload & Configuration")
st.caption("Load, validate, and configure your proteomics data")

# Initialize session state for this page
if "upload_step" not in st.session_state:
    st.session_state.upload_step = 1


# ============================================================================
# STEP 1: SELECT DATA TYPE
# ============================================================================

st.header("Step 1️⃣: Select Data Type")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("""
    **Select the type of data you're analyzing:**
    
    - **Protein:** Unique protein identifiers
    - **Peptide:** Peptide sequences with protein mapping
    """)

data_type = st.radio(
    "What type of data are you uploading?",
    options=["protein", "peptide"],
    format_func=lambda x: f"🧬 {x.capitalize()} Data" if x == "protein" else f"🔗 {x.capitalize()} Data",
    horizontal=True,
    key="data_type_selection"
)

st.session_state.data_type = data_type

st.markdown("---")


# ============================================================================
# STEP 2: FILE UPLOAD
# ============================================================================

st.header("Step 2️⃣: Upload Your Data File")

st.markdown("""
**Supported formats:** CSV, TSV, TXT, Excel (.xlsx, .xls)

**Requirements:**
- First column: ID column (Protein/Gene name or Peptide sequence)
- Numeric columns: Sample abundance values (counts, intensities, ratios)
- Column naming: Use format like `A1`, `A2`, `B1`, `B2` (condition_replicate)
""")

uploaded_file = st.file_uploader(
    "Choose a file",
    type=["csv", "tsv", "txt", "xlsx", "xls"],
    help="Maximum file size: 100 MB",
    key="file_uploader"
)

if uploaded_file is None:
    st.info("👈 Upload a file to continue")
    st.stop()

# Log upload
logger.info(f"File uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

st.markdown("---")


# ============================================================================
# STEP 3: LOAD & VALIDATE DATA
# ============================================================================

st.header("Step 3️⃣: Load & Validate")

try:
    with st.spinner("Loading file..."):
        # Determine file type
        file_ext = Path(uploaded_file.name).suffix.lower()
        
        # Load file
        if file_ext in [".csv", ".tsv", ".txt"]:
            # Try to detect delimiter
            delimiter = "\t" if file_ext == ".tsv" else ","
            df_raw = load_csv(uploaded_file, sep=delimiter)
        elif file_ext in [".xlsx", ".xls"]:
            df_raw = load_excel(uploaded_file)
        else:
            st.error(f"❌ Unsupported file type: {file_ext}")
            st.stop()
        
        st.success(f"✅ Loaded {len(df_raw):,} rows × {len(df_raw.columns)} columns")
        
        logger.info(f"File loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
        
        # Display preview
        with st.expander("Preview First 5 Rows", expanded=False):
            st.dataframe(df_raw.head(5), use_container_width=True)

except Exception as e:
    st.error(f"❌ Error loading file: {str(e)}")
    logger.error(f"Error loading file: {str(e)}")
    st.stop()


st.markdown("---")


# ============================================================================
# STEP 4: COLUMN DETECTION
# ============================================================================

st.header("Step 4️⃣: Column Configuration")

st.markdown("AutoProt has automatically detected column types. Review and adjust if needed:")

# Detect numeric columns
numeric_cols, categorical_cols = detect_numeric_columns(df_raw)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Numeric Columns (Samples)")
    st.caption(f"Found: {len(numeric_cols)} columns")
    
    if numeric_cols:
        st.dataframe(
            pd.DataFrame({
                "Column Name": numeric_cols,
                "Data Type": [str(df_raw[col].dtype) for col in numeric_cols]
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("⚠️ No numeric columns detected!")

with col2:
    st.subheader("🏷️ Categorical/Text Columns")
    st.caption(f"Found: {len(categorical_cols)} columns")
    
    if categorical_cols:
        st.dataframe(
            pd.DataFrame({
                "Column Name": categorical_cols,
                "Data Type": [str(df_raw[col].dtype) for col in categorical_cols]
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No categorical columns detected")

st.markdown("---")


# ============================================================================
# STEP 5: ID COLUMN SELECTION
# ============================================================================

st.header("Step 5️⃣: Select ID Column")

st.markdown("Which column contains the unique identifiers (protein/gene names or peptide sequences)?")

id_col = st.selectbox(
    "ID Column:",
    options=categorical_cols if categorical_cols else numeric_cols,
    help="This column identifies each row uniquely",
    key="id_column_select"
)

if id_col:
    st.caption(f"Selected: **{id_col}**")
    
    # Check for duplicates
    duplicates = check_duplicates(df_raw, id_col)
    
    if duplicates[0] > 0:
        st.warning(f"⚠️ Found {duplicates[0]} duplicate IDs")
        with st.expander("Show duplicate IDs"):
            st.dataframe(pd.DataFrame({"Duplicate IDs": duplicates[1]}), hide_index=True)
    else:
        st.success(f"✅ All {len(df_raw)} IDs are unique")

st.markdown("---")


# ============================================================================
# STEP 6: OPTIONAL SPECIES COLUMN
# ============================================================================

st.header("Step 6️⃣: Species (Optional)")

st.markdown("Does your data contain a species/organism annotation column?")

species_options = [None] + [col for col in categorical_cols if col != id_col]

species_col = st.selectbox(
    "Species Column:",
    options=species_options,
    format_func=lambda x: "None - I'll skip this" if x is None else x,
    key="species_column_select",
    help="Optional: Used for filtering and quality control"
)

if species_col:
    unique_species = df_raw[species_col].nunique()
    st.info(f"Found {unique_species} unique species/organisms")

st.markdown("---")


# ============================================================================
# STEP 7: OPTIONAL SEQUENCE COLUMN (PEPTIDE ONLY)
# ============================================================================

if data_type == "peptide":
    st.header("Step 7️⃣: Peptide Sequence (Peptide Data Only)")
    
    st.markdown("Which column contains the peptide sequences?")
    
    sequence_options = [None] + [col for col in categorical_cols if col != id_col and col != species_col]
    
    sequence_col = st.selectbox(
        "Sequence Column:",
        options=sequence_options,
        format_func=lambda x: "None - Use ID as sequence" if x is None else x,
        key="sequence_column_select"
    )
    
    st.markdown("---")
else:
    sequence_col = None


# ============================================================================
# STEP 8: SAMPLE NAMING & CONDITION DETECTION
# ============================================================================

st.header("Step 8️⃣: Condition Detection")

st.markdown("""
AutoProt automatically detects experimental conditions from sample column names.

**Naming convention:** `[Condition][Replicate]` (e.g., `A1`, `A2`, `B1`, `B2`)
""")

# Detect conditions
numeric_cols_filtered = [col for col in numeric_cols if col != id_col]

if numeric_cols_filtered:
    conditions = detect_conditions_from_columns(numeric_cols_filtered)
    
    if conditions:
        st.success(f"✅ Detected {len(conditions)} conditions: {', '.join(sorted(conditions))}")
        
        # Create and show condition mapping
        condition_mapping = create_condition_mapping(numeric_cols_filtered)
        
        with st.expander("View Condition Mapping", expanded=True):
            mapping_df = pd.DataFrame([
                {"Sample": col, "Condition": cond}
                for col, cond in condition_mapping.items()
            ])
            st.dataframe(mapping_df, use_container_width=True, hide_index=True)
        
        st.session_state.condition_mapping = condition_mapping
    else:
        st.warning("⚠️ Could not auto-detect conditions from column names")
        st.info("Make sure column names follow pattern: A1, A2, B1, B2 (condition_replicate)")

st.markdown("---")


# ============================================================================
# STEP 9: DATA VALIDATION
# ============================================================================

st.header("Step 9️⃣: Validation")

# Validate dataframe
numeric_cols_filtered = [col for col in numeric_cols if col != id_col]

is_valid, validation_msg = validate_dataframe(
    df_raw,
    id_col=id_col,
    numeric_cols=numeric_cols_filtered,
    min_rows=2,
    min_cols=2
)

if is_valid:
    st.success("✅ Data validation passed")
else:
    st.error(f"❌ Validation failed: {validation_msg}")
    st.stop()

# Check missing data
missing_data = check_missing_data(df_raw, numeric_cols_filtered)

# Extract values from missing_data dictionary with safe defaults
total_missing = missing_data.get('total_missing', 0)
missing_percent = missing_data.get('missing_percent', 0.0)
complete_rows = missing_data.get('complete_rows', 0)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Missing", f"{total_missing:,} cells")

with col2:
    st.metric("Missing %", f"{missing_percent:.1f}%")

with col3:
    st.metric("Complete Cases", f"{complete_rows}")

# Show detailed missing data analysis
with st.expander("Missing Data Details", expanded=False):
    missing_by_col = missing_data.get("by_column", {})
    if missing_by_col:
        missing_df = pd.DataFrame([
            {"Column": col, "Missing": count, "Percent": f"{(count/len(df_raw)*100):.1f}%"}
            for col, count in missing_by_col.items()
        ]).sort_values("Missing", ascending=False)
        
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
    else:
        st.info("No missing data details available")

st.markdown("---")


# ============================================================================
# STEP 10: DATA SUMMARY
# ============================================================================

st.header("Step 🔟: Data Summary")

# Calculate statistics directly from numeric columns
# This is more reliable than get_data_summary() which may have key mismatches
numeric_data = df_raw[numeric_cols_filtered].copy()

# Calculate statistics from actual data
n_rows = len(df_raw)
n_cols = len(numeric_cols_filtered)

# Get statistics from all numeric values (ignoring NaN)
all_values = numeric_data.values.flatten()
all_values_clean = all_values[~pd.isna(all_values)]

if len(all_values_clean) > 0:
    mean_abundance = float(np.nanmean(all_values_clean))
    median_abundance = float(np.nanmedian(all_values_clean))
    min_value = float(np.nanmin(all_values_clean))
    max_value = float(np.nanmax(all_values_clean))
else:
    mean_abundance = 0.0
    median_abundance = 0.0
    min_value = 0.0
    max_value = 0.0

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Proteins/Peptides", f"{n_rows:,}")
    st.metric("Samples", n_cols)

with col2:
    st.metric("Mean Abundance", f"{mean_abundance:.1f}")
    st.metric("Median Abundance", f"{median_abundance:.1f}")

with col3:
    if min_value > 0:
        st.metric("Min Value", f"{min_value:.2e}")
    else:
        st.metric("Min Value", "0.0")
    
    if max_value > 0:
        st.metric("Max Value", f"{max_value:.2e}")
    else:
        st.metric("Max Value", "0.0")

# Intensity distribution
with st.expander("Intensity Distribution", expanded=False):
    st.caption("Note: 1.0 values are treated as missing data (preprocessing artifact)")

st.markdown("---")


# ============================================================================
# STEP 11: FINALIZE & UPLOAD
# ============================================================================

st.header("Step 1️⃣1️⃣: Finalize Upload")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Before uploading, verify:**
    
    - ✓ ID column selected
    - ✓ Numeric columns detected
    - ✓ Data validated
    - ✓ No major quality issues
    """)

with col2:
    st.markdown(f"""
    **Summary:**
    
    - Type: **{data_type.title()}**
    - Rows: **{len(df_raw):,}**
    - Samples: **{len(numeric_cols_filtered)}**
    - Missing: **{missing_percent:.1f}%**
    - Status: **✅ Ready**
    """)

# IMPORTANT: Handle 1.0 replacement as intentional preprocessing correction
st.info("""
**ℹ️ Data Preprocessing Note**

The platform treats abundance value of **1.0 as missing data** because this is 
a known preprocessing artifact where the instrument software outputs 1.0 as an 
invalid/null marker. This value will be replaced with NaN for proper statistical handling.
""")

col1, col2 = st.columns([2, 1])

with col1:
    confirm = st.checkbox(
        "✅ I confirm the data looks correct and I'm ready to upload",
        key="confirm_upload"
    )

with col2:
    if confirm:
        if st.button("🚀 Upload & Proceed", use_container_width=True, type="primary", key="btn_upload"):
            try:
                with st.spinner("Processing data..."):
                    # Convert 1.0 to NaN (preprocessing artifact)
                    for col in numeric_cols_filtered:
                        df_raw[col] = df_raw[col].replace(1.0, float('nan'))
                    
                    # Convert string-formatted numbers to float
                    df_raw = convert_string_numbers_to_float(df_raw, numeric_cols_filtered)
                    
                    # Create data container based on type
                    if data_type == "protein":
                        data_obj = ProteinData(
                            raw=df_raw,
                            numeric_cols=numeric_cols_filtered,
                            id_col=id_col,
                            species_col=species_col,
                            file_path=uploaded_file.name
                        )
                        st.session_state.protein_data = data_obj
                    else:  # peptide
                        data_obj = PeptideData(
                            raw=df_raw,
                            numeric_cols=numeric_cols_filtered,
                            id_col=id_col,
                            species_col=species_col,
                            sequence_col=sequence_col,
                            file_path=uploaded_file.name
                        )
                        st.session_state.peptide_data = data_obj
                    
                    # Store in session state
                    st.session_state.df_raw = df_raw
                    st.session_state.numeric_cols = numeric_cols_filtered
                    st.session_state.id_col = id_col
                    st.session_state.species_col = species_col
                    st.session_state.sequence_col = sequence_col if data_type == "peptide" else None
                    st.session_state.metadata_columns = categorical_cols
                    st.session_state.column_mapping = condition_mapping if 'condition_mapping' in st.session_state else {}
                    st.session_state.data_ready = True
                    
                    logger.info(f"Data uploaded successfully: {data_type} ({len(df_raw)} rows × {len(numeric_cols_filtered)} samples)")
                
                st.success("✅ Data uploaded successfully!")
                st.balloons()
                
                # Show next steps
                st.markdown("---")
                st.subheader("✨ What's Next?")
                st.markdown("""
                Your data is now ready for analysis! 
                
                **Next steps:**
                
                1. Go to **📊 Visual EDA** to explore distributions
                2. Choose a transformation (log2, yeo-johnson, etc.)
                3. Assess normality with Q-Q plots
                4. Proceed to **🧪 Statistical EDA** for filtering and testing
                """)
                
            except Exception as e:
                st.error(f"❌ Error processing data: {str(e)}")
                logger.error(f"Error processing data: {str(e)}")
    else:
        st.caption("👈 Check the confirmation box to proceed")

st.markdown("---")

st.caption("""
**Need help?** Check the documentation on the home page or review the 
data requirements section above.
""")
