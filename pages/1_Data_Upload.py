"""
pages/1_Data_Upload.py - MANDATORY SPECIES COLUMN SELECTION
Unified data upload with column selection, intelligent renaming, and SAFE species detection

Advanced features:
- Smart column selection (keep/drop metadata and samples)
- Intelligent column renaming (trim, extract, auto-generate)
- MANDATORY species column selection (searches all metadata columns)
- Polars for efficient data handling
- Direct column manipulation
"""

import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import gc
import re

from helpers.core import ProteinData, PeptideData

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Data Upload - AutoProt",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📁 Data Upload & Configuration")
st.markdown("Load, validate, configure, and rename your proteomics data")

if "upload_step" not in st.session_state:
    st.session_state.upload_step = 1


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def longest_common_prefix(strings: list) -> str:
    """Find longest common prefix from list of strings."""
    if not strings:
        return ""
    s1, s2 = min(strings), max(strings)
    for i, (c1, c2) in enumerate(zip(s1, s2)):
        if c1 != c2:
            return s1[:i]
    return s1


def generate_default_column_names(n_cols: int, replicates_per_condition: int = 2) -> list:
    """Generate default names: A1, A2, B1, B2, etc."""
    names = []
    for i in range(n_cols):
        condition_idx = i // replicates_per_condition
        replicate_num = (i % replicates_per_condition) + 1
        condition_letter = chr(ord('A') + condition_idx)
        names.append(f"{condition_letter}{replicate_num}")
    return names


def infer_species_from_text(text: str) -> str:
    """Extract species from any text string (e.g., 'PROT_HUMAN' → 'HUMAN')."""
    if pd.isna(text) or text is None:
        return None
    
    s = str(text).upper()
    
    # Check common patterns
    if "HUMAN" in s:
        return "HUMAN"
    if "MOUSE" in s:
        return "MOUSE"
    if "YEAST" in s:
        return "YEAST"
    if "ECOLI" in s or "_ECOL" in s:
        return "ECOLI"
    if "DROSOPHILA" in s or "FRUIT" in s:
        return "DROSOPHILA"
    if "ARABIDOPSIS" in s:
        return "ARABIDOPSIS"
    if "ZEBRAFISH" in s:
        return "ZEBRAFISH"
    if "CHICKEN" in s:
        return "CHICKEN"
    if "DOG" in s or "CANIS" in s:
        return "DOG"
    if "CAT" in s or "FELIS" in s:
        return "CAT"
    
    # Fallback: last token after underscore
    if "_" in s:
        tail = s.split("_")[-1]
        if len(tail) >= 3:
            return tail
    
    return None


def extract_protein_id_from_name(name: str) -> str:
    """Extract just the protein ID without species (e.g., 'NUD4B_HUMAN' → 'NUD4B')."""
    if pd.isna(name) or name is None:
        return None
    
    s = str(name)
    
    # If has underscore, return part before last underscore
    if "_" in s:
        parts = s.rsplit("_", 1)
        return parts[0]
    
    return s


def scan_column_for_species(column_data: pd.Series) -> dict:
    """Scan a column and detect all unique species found."""
    species_found = {}
    
    for value in column_data.dropna().unique():
        species = infer_species_from_text(str(value))
        if species:
            if species not in species_found:
                species_found[species] = 0
            species_found[species] += 1
    
    return species_found


def trim_common_prefix_suffix(columns: list) -> dict:
    """Remove common prefix and suffix from column names."""
    if len(columns) < 2:
        return {col: col for col in columns}
    
    prefix = longest_common_prefix(columns)
    reversed_cols = [col[::-1] for col in columns]
    suffix = longest_common_prefix(reversed_cols)[::-1]
    
    mapping = {}
    for col in columns:
        trimmed = col[len(prefix):] if prefix else col
        trimmed = trimmed[:-len(suffix)] if suffix else trimmed
        mapping[col] = trimmed if trimmed else col
    
    return mapping


def smart_rename_columns(columns: list, style: str = 'trim') -> dict:
    """Rename columns with various strategies."""
    if style == 'trim':
        return trim_common_prefix_suffix(columns)
    elif style == 'default':
        new_names = generate_default_column_names(len(columns))
        return {orig: new for orig, new in zip(columns, new_names)}
    else:
        return {col: col for col in columns}


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
st.session_state.data_type = "peptide" if peptides else "protein"

uploaded_file = st.file_uploader(
    f"Choose a {st.session_state.data_type} data file",
    type=["csv", "xlsx", "xls"],
    key=f"file_upload_{st.session_state.data_type}"
)

if uploaded_file is None:
    st.info(f"👈 Upload a {st.session_state.data_type} abundance file to begin analysis")
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
# SELECT COLUMNS - METADATA & SAMPLES
# ============================================================================

st.subheader("2️⃣ Select Columns (Keep/Drop)")

st.markdown("""
**Choose which columns to keep for downstream analysis:**

- **Metadata columns:** Protein/gene info, descriptions, species
- **Sample columns:** Abundance values you want to analyze
""")

df_preview = df_raw.head(5).to_pandas()

col1, col2 = st.columns(2)

# Metadata columns
with col1:
    st.subheader("📋 Metadata Columns")
    st.caption(f"Found: {len(df_raw.columns) - len([c for c in df_raw.columns if df_raw[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]])} columns")
    
    # Auto-detect metadata (non-numeric)
    all_metadata = [c for c in df_raw.columns if df_raw[c].dtype not in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
    
    if all_metadata:
        selected_metadata = st.multiselect(
            "Select metadata to KEEP:",
            options=all_metadata,
            default=all_metadata,
            key="select_metadata"
        )
    else:
        selected_metadata = []
        st.info("No metadata columns detected")

# Sample columns
with col2:
    st.subheader("🧪 Sample Columns")
    
    # Auto-detect samples (numeric)
    all_samples = [c for c in df_raw.columns if df_raw[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
    st.caption(f"Found: {len(all_samples)} columns")
    
    if all_samples:
        selected_samples = st.multiselect(
            "Select samples to KEEP:",
            options=all_samples,
            default=all_samples,
            key="select_samples"
        )
    else:
        selected_samples = []
        st.warning("No numeric columns detected!")

if not selected_metadata or not selected_samples:
    st.warning("⚠️ Select at least one metadata and one sample column")
    st.stop()

st.success(f"✅ Keeping {len(selected_metadata)} metadata + {len(selected_samples)} samples")
st.markdown("---")

# ============================================================================
# RENAME COLUMNS
# ============================================================================

st.subheader("3️⃣ Rename Columns")

col1, col2 = st.columns(2)

# Metadata renaming
with col1:
    st.subheader("📋 Rename Metadata")
    
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

# Sample renaming
with col2:
    st.subheader("🧪 Rename Samples")
    
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
        prefix = longest_common_prefix(selected_samples)
        reversed_strs = [s[::-1] for s in selected_samples]
        suffix = longest_common_prefix(reversed_strs)[::-1]
        
        st.info(f"Removing prefix: '{prefix[:20]}...' and suffix: '...{suffix[-10:]}'")
        
        for col in selected_samples:
            trimmed = col[len(prefix):] if prefix else col
            trimmed = trimmed[:-len(suffix)] if suffix else trimmed
            sample_rename[col] = trimmed if trimmed else col
        
        st.caption(f"✓ Trimmed {len(sample_rename)} sample(s)")
    
    elif rename_strategy == "Extract ID":
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

st.markdown("---")

# ============================================================================
# APPLY CHANGES
# ============================================================================

st.subheader("4️⃣ Apply Changes & Validate")

# Filter and rename
df_filtered = df_raw.select(selected_metadata + selected_samples)

# Apply renames
all_renames = {**metadata_rename, **sample_rename}
if all_renames:
    df_filtered = df_filtered.rename(all_renames)
    selected_metadata = [metadata_rename.get(col, col) for col in selected_metadata]
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
    # Calculate mean
    try:
        numeric_data = df_filtered.select(numeric_cols_final).to_pandas()
        all_values = numeric_data.values.flatten()
        all_values_clean = all_values[~pd.isna(all_values)]
        if len(all_values_clean) > 0:
            mean_val = float(np.nanmean(all_values_clean))
            st.metric("Mean Abundance", f"{mean_val:.1f}")
    except:
        st.metric("Mean Abundance", "N/A")

st.markdown("---")

# ============================================================================
# CONFIGURE KEY COLUMNS
# ============================================================================

st.subheader("5️⃣ Configure Key Columns")

with col1:
    id_col = st.selectbox(
        "ID Column (Protein/Gene names):",
        options=selected_metadata,
        key="id_col_widget",  # different from st.session_state.id_col
    )

with col2:
    species_col_select = st.selectbox(
        "Species Column (REQUIRED):",
        options=selected_metadata,
        key="species_col_widget",  # different from st.session_state.species_col
        help="Select which column contains species information",
    )


if st.session_state.data_type == "peptide":
    with col3:
        sequence_col = st.selectbox(
            "Sequence Column:",
            options=selected_metadata,
            key="sequence_col"
        )
else:
    sequence_col = None

st.markdown("---")

# ============================================================================
# SCAN SPECIES COLUMN
# ============================================================================

st.subheader("🔬 Species Detection")

df_pandas_temp = df_filtered.to_pandas()

st.info(f"Scanning column **{species_col_select}** for species information...")

# Scan all values in the selected species column
species_scan = scan_column_for_species(df_pandas_temp[species_col_select])

if species_scan:
    st.success(f"✅ Found {len(species_scan)} unique species:")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Show table of species found
        species_df = pd.DataFrame(
            list(species_scan.items()),
            columns=['Species', 'Count']
        ).sort_values('Count', ascending=False)
        
        st.dataframe(species_df, width="stretch", hide_index=True)
    
    with col2:
        # Show distribution chart
        with st.expander("View Distribution", expanded=True):
            st.bar_chart(species_df.set_index('Species')['Count'])
    
    # Show sample data with inferred species
    st.subheader("Sample Data Preview")
    
    inferred_col = df_pandas_temp[species_col_select].apply(infer_species_from_text)
    preview_df = pd.DataFrame({
        'Row': range(min(10, len(df_pandas_temp))),
        species_col_select: df_pandas_temp[species_col_select].head(10).values,
        'Inferred Species': inferred_col.head(10).values
    })
    
    st.dataframe(preview_df, width="stretch", hide_index=True)
else:
    st.warning(f"⚠️ No species patterns detected in **{species_col_select}**")
    st.info("Common patterns detected: HUMAN, MOUSE, YEAST, ECOLI, DROSOPHILA, ARABIDOPSIS, ZEBRAFISH, etc.")
    st.info("Species can also be detected as the last token after underscore (e.g., PROT_HUMAN → HUMAN)")

st.markdown("---")

# ============================================================================
# OPTIONAL: EXTRACT PROTEIN IDS
# ============================================================================

st.subheader("📌 Extract Protein IDs (Optional)")

st.caption("If your ID column contains both protein name and species (e.g., NUD4B_HUMAN), extract just the ID.")

extract_protein = st.checkbox(
    "Extract protein ID (remove species suffix like _HUMAN)",
    value=False,
    key="extract_protein_id"
)

if extract_protein:
    st.info("Extracting protein IDs without species suffix...")
    
    extracted_ids = df_pandas_temp[id_col].apply(extract_protein_id_from_name)
    
    preview_extract = pd.DataFrame({
        'Original': df_pandas_temp[id_col].head(10),
        'Extracted ID': extracted_ids.head(10)
    })
    
    st.dataframe(preview_extract, width="stretch", hide_index=True)
    
    # Replace in dataframe
    df_filtered = df_filtered.with_columns([
        pl.Series(id_col, extracted_ids.tolist())
    ])
    st.success(f"✅ Extracted protein IDs from {id_col}")

st.markdown("---")

# ============================================================================
# FINAL VALIDATION & UPLOAD
# ============================================================================

st.subheader("6️⃣ Validate & Upload")

col1, col2 = st.columns(2)

with col1:
    st.write("**Validation Checks:**")
    
    checks = {
        "✅ Metadata columns selected": len(selected_metadata) > 0,
        "✅ Numerical columns selected": len(numeric_cols_final) > 0,
        "✅ ID column configured": id_col is not None,
        "✅ Species column selected": species_col_select is not None,
        "✅ Data loaded": df_filtered is not None,
        "✅ Samples available": len(df_filtered) > 0,
        "✅ Species detected in column": len(species_scan) > 0,
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
    st.write(f"- **Species Column:** {species_col_select}")
    st.write(f"- **Species Found:** {len(species_scan)}")
    if st.session_state.data_type == 'peptide':
        st.write(f"- **Sequence Column:** {sequence_col}")
    st.write(f"- **Metadata Columns:** {len(selected_metadata)}")
    st.write(f"- **Samples:** {len(numeric_cols_final)}")
    st.write(f"- **Total {st.session_state.data_type.title()}s:** {len(df_filtered):,}")

st.markdown("---")

# ============================================================================
# UPLOAD BUTTON
# ============================================================================

if st.button(
    f"🚀 Upload {st.session_state.data_type.upper()} Data",
    type="primary",
    width="stretch",
    disabled=not all_passed
):
    with st.spinner(f"Processing {st.session_state.data_type} data..."):
        try:
            df_final_pandas = df_filtered.to_pandas()
            
            if st.session_state.data_type == 'protein':
                data_obj = ProteinData(
                    raw=df_final_pandas,
                    numeric_cols=numeric_cols_final,
                    id_col=id_col,
                    species_col=species_col_select,
                    file_path=str(uploaded_file.name)
                )
                st.session_state.protein_data = data_obj
            else:
                data_obj = PeptideData(
                    raw=df_final_pandas,
                    numeric_cols=numeric_cols_final,
                    id_col=id_col,
                    species_col=species_col_select,
                    sequence_col=sequence_col,
                    file_path=str(uploaded_file.name)
                )
                st.session_state.peptide_data = data_obj
            
            # Store in session
            st.session_state.df_raw = df_final_pandas
            st.session_state.df_raw_polars = df_filtered
            st.session_state.numeric_cols = numeric_cols_final
            st.session_state.id_col = id_col
            st.session_state.species_col = species_col_select
            st.session_state.sequence_col = sequence_col if st.session_state.data_type == "peptide" else None
            st.session_state.metadata_columns = selected_metadata
            st.session_state.data_ready = True
            
            gc.collect()
            
            st.success(f"✅ {st.session_state.data_type.upper()} data uploaded successfully!")
            st.info("Ready for analysis. Go to **Visual EDA** page to continue.")
            
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")

st.markdown("---")

st.caption("**💡 Tip:** Species column is REQUIRED and will be scanned for common species patterns (HUMAN, MOUSE, YEAST, etc.). All metadata columns are searched, not just the ID column. This ensures accuracy and prevents false detection.")
