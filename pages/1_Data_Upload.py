"""
pages/1_Data_Upload.py - ENHANCED WITH COLUMN DROPPING & INTELLIGENT RENAMING

Advanced data upload with:
- Smart column selection (keep/drop metadata columns)
- Intelligent column renaming (trim, extract, auto-generate)
- Variable file format support
- Species inference from protein names
- Direct column manipulation
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import re
from typing import Dict, List, Tuple

from helpers.io import load_csv, load_excel, detect_numeric_columns, validate_dataframe, check_duplicates, check_missing_data, convert_string_numbers_to_float
from helpers.core import ProteinData, PeptideData
from helpers.analysis import detect_conditions_from_columns, create_condition_mapping

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Data Upload - AutoProt",
    page_icon="📁",
    layout="wide"
)

st.title("📁 Data Upload & Configuration")
st.caption("Load, validate, configure, and rename your proteomics data")

if "upload_step" not in st.session_state:
    st.session_state.upload_step = 1


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def infer_species_from_protein_name(name: str) -> str:
    """Extract species from protein name (e.g., 'PROT_HUMAN' → 'HUMAN')."""
    if pd.isna(name) or name is None:
        return None
    
    s = str(name).upper()
    
    # Check common patterns
    if "HUMAN" in s:
        return "HUMAN"
    if "MOUSE" in s:
        return "MOUSE"
    if "YEAST" in s:
        return "YEAST"
    if "ECOLI" in s or "_ECOL" in s:
        return "ECOLI"
    
    # Fallback: last token after underscore
    if "_" in s:
        tail = s.split("_")[-1]
        if len(tail) >= 3:
            return tail
    
    return None


# ============================================================================
# STEP 1: SELECT DATA TYPE
# ============================================================================

st.header("Step 1️⃣: Select Data Type")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("""
    **Select the type of data:**
    
    - **Protein:** Unique protein identifiers
    - **Peptide:** Peptide sequences with protein mapping
    """)

data_type = st.radio(
    "Data type:",
    options=["protein", "peptide"],
    format_func=lambda x: f"🧬 {x.capitalize()}" if x == "protein" else f"🔗 {x.capitalize()}",
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

**Note:** Uploads are highly variable - many metadata columns that may not be needed downstream. You'll have full control to select and drop columns.
""")

uploaded_file = st.file_uploader(
    "Choose a file",
    type=["csv", "tsv", "txt", "xlsx", "xls"],
    key="file_uploader"
)

if uploaded_file is None:
    st.info("👈 Upload a file to continue")
    st.stop()

logger.info(f"File uploaded: {uploaded_file.name}")
st.markdown("---")


# ============================================================================
# STEP 3: LOAD DATA
# ============================================================================

st.header("Step 3️⃣: Load & Preview")

try:
    with st.spinner("Loading file..."):
        file_ext = Path(uploaded_file.name).suffix.lower()
        
        if file_ext in [".csv", ".tsv", ".txt"]:
            delimiter = "\t" if file_ext == ".tsv" else ","
            df_raw = load_csv(uploaded_file, sep=delimiter)
        elif file_ext in [".xlsx", ".xls"]:
            df_raw = load_excel(uploaded_file)
        else:
            st.error(f"❌ Unsupported file type: {file_ext}")
            st.stop()
        
        st.success(f"✅ Loaded {len(df_raw):,} rows × {len(df_raw.columns)} columns")
        logger.info(f"File loaded: {df_raw.shape}")
        
        with st.expander("Preview First 5 Rows", expanded=False):
            st.dataframe(df_raw.head(5), width="stretch")

except Exception as e:
    st.error(f"❌ Error loading file: {str(e)}")
    logger.error(f"Error loading file: {str(e)}")
    st.stop()

st.markdown("---")


# ============================================================================
# STEP 4: SELECT & DROP COLUMNS
# ============================================================================

st.header("Step 4️⃣: Select Columns (Keep/Drop)")

st.markdown("""
**Choose which columns to keep for downstream analysis:**

- **Metadata columns:** Protein/gene info, descriptions, species
- **Sample columns:** Abundance values you want to analyze
""")

numeric_cols, categorical_cols = detect_numeric_columns(df_raw)

col1, col2 = st.columns(2)

# Metadata columns
with col1:
    st.subheader("📋 Metadata Columns")
    st.caption(f"Found: {len(categorical_cols)} columns")
    
    if categorical_cols:
        selected_metadata = st.multiselect(
            "Select metadata to KEEP:",
            options=categorical_cols,
            default=categorical_cols,
            key="select_metadata"
        )
    else:
        selected_metadata = []
        st.info("No categorical columns detected")

# Sample columns
with col2:
    st.subheader("🧪 Sample Columns")
    st.caption(f"Found: {len(numeric_cols)} columns")
    
    if numeric_cols:
        selected_samples = st.multiselect(
            "Select samples to KEEP:",
            options=numeric_cols,
            default=numeric_cols,
            key="select_samples"
        )
    else:
        selected_samples = []
        st.warning("No numeric columns detected!")

if not selected_metadata or not selected_samples:
    st.warning("⚠️ Select at least one metadata and one sample column")
    st.stop()

st.success(f"✅ Keeping {len(selected_metadata)} metadata + {len(selected_samples)} samples = {len(selected_metadata) + len(selected_samples)} total columns")
st.markdown("---")


# ============================================================================
# STEP 5: RENAME COLUMNS INTELLIGENTLY
# ============================================================================

st.header("Step 5️⃣: Rename Columns")

st.markdown("**Rename columns for clarity and standardization**")

col1, col2 = st.columns(2)

# METADATA RENAMING
with col1:
    st.subheader("📋 Rename Metadata")
    
    if selected_metadata:
        metadata_rename = {}
        
        for col in selected_metadata:
            new_name = st.text_input(
                f"Rename '{col}':",
                value=col,
                key=f"rename_meta_{col}"
            )
            if new_name != col:
                metadata_rename[col] = new_name
        
        if metadata_rename:
            st.caption(f"✓ Renaming {len(metadata_rename)} metadata column(s)")
        else:
            st.caption("No renames - using original names")
    else:
        metadata_rename = {}

# SAMPLE RENAMING
with col2:
    st.subheader("🧪 Rename Samples")
    
    if selected_samples:
        rename_strategy = st.radio(
            "Strategy:",
            options=[
                "Keep Original",
                "Trim Prefix/Suffix",
                "Extract ID",
                "Auto-Generate (S1, S2, ...)"
            ],
            key="sample_strategy"
        )
        
        sample_rename = {}
        
        if rename_strategy == "Keep Original":
            st.caption("Using original names")
            sample_rename = {}
        
        elif rename_strategy == "Trim Prefix/Suffix":
            # Find common prefix/suffix and remove
            def common_prefix(strs):
                if not strs: return ""
                s1, s2 = min(strs), max(strs)
                for i, (c1, c2) in enumerate(zip(s1, s2)):
                    if c1 != c2: return s1[:i]
                return s1
            
            prefix = common_prefix(selected_samples)
            reversed_strs = [s[::-1] for s in selected_samples]
            suffix = common_prefix(reversed_strs)[::-1]
            
            st.info(f"Removing prefix: '{prefix[:20]}...' and suffix: '...{suffix[-10:]}'")
            
            for col in selected_samples:
                trimmed = col[len(prefix):] if prefix else col
                trimmed = trimmed[:-len(suffix)] if suffix else trimmed
                sample_rename[col] = trimmed if trimmed else col
            
            st.caption(f"✓ Trimmed {len(sample_rename)} sample(s)")
        
        elif rename_strategy == "Extract ID":
            # Extract last numeric ID (e.g., 1268 from ...1268.d)
            st.info("Extracting numeric IDs from file paths")
            
            for col in selected_samples:
                match = re.search(r'(\d{4})\.d$', col)
                if match:
                    sample_rename[col] = f"S{match.group(1)}"
                else:
                    sample_rename[col] = col
            
            st.caption(f"✓ Extracted {len(sample_rename)} sample IDs")
        
        elif rename_strategy == "Auto-Generate (S1, S2, ...)":
            st.info(f"Auto-generating: S1, S2, ..., S{len(selected_samples)}")
            sample_rename = {col: f"S{i+1}" for i, col in enumerate(selected_samples)}
            st.caption(f"✓ Generated {len(sample_rename)} new names")
    else:
        sample_rename = {}

st.markdown("---")


# ============================================================================
# STEP 6: APPLY CHANGES & VALIDATE
# ============================================================================

st.header("Step 6️⃣: Apply Changes & Validate")

# Filter columns
df_filtered = df_raw[selected_metadata + selected_samples].copy()

# Apply metadata renames
if metadata_rename:
    df_filtered.rename(columns=metadata_rename, inplace=True)
    selected_metadata = [metadata_rename.get(col, col) for col in selected_metadata]

# Apply sample renames
if sample_rename:
    df_filtered.rename(columns=sample_rename, inplace=True)
    selected_samples = [sample_rename.get(col, col) for col in selected_samples]

numeric_cols_final = selected_samples

st.success(f"✅ Filtered to {len(selected_metadata)} metadata + {len(numeric_cols_final)} samples")

# DATA SUMMARY
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Rows", f"{len(df_filtered):,}")
    st.metric("Metadata Cols", len(selected_metadata))

with col2:
    st.metric("Sample Cols", len(numeric_cols_final))
    st.metric("Total Cols", len(df_filtered.columns))

with col3:
    all_values = df_filtered[numeric_cols_final].values.flatten()
    all_values_clean = all_values[~pd.isna(all_values)]
    if len(all_values_clean) > 0:
        mean_val = float(np.nanmean(all_values_clean))
        st.metric("Mean Abundance", f"{mean_val:.1f}")

with st.expander("View Filtered Data", expanded=False):
    st.dataframe(df_filtered.head(10), width="stretch")

st.markdown("---")


# ============================================================================
# STEP 7: CONFIGURE KEY COLUMNS & INFER SPECIES
# ============================================================================

st.header("Step 7️⃣: Configure Key Columns")

col1, col2, col3 = st.columns(3)

with col1:
    id_col = st.selectbox(
        "ID Column:",
        options=selected_metadata,
        key="id_col"
    )

with col2:
    species_options = [None] + selected_metadata
    species_col = st.selectbox(
        "Species Column (optional):",
        options=species_options,
        key="species_col"
    )

if data_type == "peptide":
    with col3:
        sequence_col = st.selectbox(
            "Sequence Column:",
            options=selected_metadata,
            key="sequence_col"
        )
else:
    sequence_col = None

# Show species inference
st.subheader("🔬 Species Information")

if species_col:
    # User provided explicit species column
    st.info(f"Using species from: **{species_col}**")
    
    unique_species = df_filtered[species_col].nunique()
    st.metric("Unique Species/Values", unique_species)
    
    with st.expander("View Species Distribution", expanded=False):
        species_counts = df_filtered[species_col].value_counts()
        st.bar_chart(species_counts)
else:
    # Infer species from ID column
    st.info(f"Inferring species from ID column: **{id_col}**")
    
    inferred_species = df_filtered[id_col].apply(infer_species_from_protein_name)
    
    # Show preview
    preview_df = pd.DataFrame({
        'Protein ID': df_filtered[id_col].head(10),
        'Inferred Species': inferred_species.head(10)
    })
    
    st.dataframe(preview_df, width="stretch", hide_index=True)
    
    # Show distribution
    species_counts = inferred_species.value_counts()
    st.metric("Unique Species Detected", len(species_counts))
    
    with st.expander("View Species Distribution", expanded=False):
        st.bar_chart(species_counts)
    
    # Add as column
    df_filtered['Inferred_Species'] = inferred_species
    species_col = 'Inferred_Species'

st.markdown("---")


# ============================================================================
# STEP 8: CALCULATE STATISTICS
# ============================================================================

st.header("Step 8️⃣: Data Statistics")

numeric_data = df_filtered[numeric_cols_final].copy()
all_values = numeric_data.values.flatten()
all_values_clean = all_values[~pd.isna(all_values)]

if len(all_values_clean) > 0:
    mean_abundance = float(np.nanmean(all_values_clean))
    median_abundance = float(np.nanmedian(all_values_clean))
    min_value = float(np.nanmin(all_values_clean))
    max_value = float(np.nanmax(all_values_clean))
else:
    mean_abundance = median_abundance = min_value = max_value = 0.0

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Proteins/Peptides", f"{len(df_filtered):,}")
    st.metric("Samples", len(numeric_cols_final))

with col2:
    st.metric("Mean Abundance", f"{mean_abundance:.1f}")
    st.metric("Median Abundance", f"{median_abundance:.1f}")

with col3:
    st.metric("Min Value", f"{min_value:.2e}" if min_value > 0 else "0.0")
    st.metric("Max Value", f"{max_value:.2e}" if max_value > 0 else "0.0")

st.markdown("---")


# ============================================================================
# STEP 9: FINAL CONFIRMATION & UPLOAD
# ============================================================================

st.header("Step 9️⃣: Finalize Upload")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    **Configuration Summary:**
    
    - Type: **{data_type.title()}**
    - ID Column: **{id_col}**
    - Species Column: **{species_col if species_col else 'None'}**
    - Rows: **{len(df_filtered):,}**
    - Samples: **{len(numeric_cols_final)}**
    - Metadata: **{len(selected_metadata)}**
    """)

with col2:
    st.markdown("""
    **Before uploading, verify:**
    
    - ✓ Columns selected correctly
    - ✓ Renames look good
    - ✓ ID column configured
    - ✓ Species column shows correct values
    - ✓ No unwanted columns
    """)

confirm = st.checkbox(
    "✅ I confirm the configuration is correct",
    key="confirm_upload"
)

if confirm:
    if st.button("🚀 Upload & Proceed", type="primary", width="stretch"):
        try:
            with st.spinner("Processing data..."):
                # Convert 1.0 to NaN (preprocessing artifact)
                for col in numeric_cols_final:
                    df_filtered[col] = df_filtered[col].replace(1.0, float('nan'))
                
                # Create data object
                if data_type == "protein":
                    data_obj = ProteinData(
                        raw=df_filtered,
                        numeric_cols=numeric_cols_final,
                        id_col=id_col,
                        species_col=species_col,
                        file_path=uploaded_file.name
                    )
                    st.session_state.protein_data = data_obj
                else:
                    data_obj = PeptideData(
                        raw=df_filtered,
                        numeric_cols=numeric_cols_final,
                        id_col=id_col,
                        species_col=species_col,
                        sequence_col=sequence_col,
                        file_path=uploaded_file.name
                    )
                    st.session_state.peptide_data = data_obj
                
                # Store in session state
                st.session_state.df_raw = df_filtered
                st.session_state.numeric_cols = numeric_cols_final
                st.session_state.id_col = id_col
                st.session_state.species_col = species_col
                st.session_state.sequence_col = sequence_col if data_type == "peptide" else None
                st.session_state.metadata_columns = selected_metadata
                st.session_state.data_ready = True
                
                logger.info(f"Data uploaded: {data_type} ({len(df_filtered)} rows × {len(numeric_cols_final)} samples)")
            
            st.success("✅ Data uploaded successfully!")
            st.balloons()
            
            st.markdown("---")
            st.subheader("✨ What's Next?")
            st.markdown("""
            Your data is ready for analysis!
            
            **Next steps:**
            1. Go to **📊 Visual EDA** to explore distributions
            2. Choose a transformation (log2, yeo-johnson, etc.)
            3. Assess normality with Q-Q plots
            4. Proceed to **🧪 Statistical EDA** for filtering and testing
            """)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            logger.error(f"Error: {str(e)}")
else:
    st.caption("👈 Check the confirmation box to proceed")

st.markdown("---")
st.caption("**💡 Tip:** Use intelligent renaming to standardize column names across different file formats. Species are automatically inferred from protein names using common patterns (e.g., _HUMAN, _MOUSE).")
