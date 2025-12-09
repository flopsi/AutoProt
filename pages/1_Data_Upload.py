"""
pages/1_Data_Upload.py - PRODUCTION DATA UPLOAD
Unified data upload with column selection and species detection

Working features preserved:
- Polars efficient data handling
- Interactive column selection (metadata + numerical)
- Column renaming (trim/default)
- Float64 conversion with NaN handling
- Replicates per condition configuration

Fixed:
- Species detection now scans ALL metadata columns
"""

import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import gc

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

# ============================================================================
# HELPER FUNCTIONS (KEEP INTACT - WORKING)
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
    """Extract species from any text string."""
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
    if "DROSOPHILA" in s or "DROME" in s:
        return "DROSOPHILA"
    if "ARABIDOPSIS" in s or "ARATH" in s:
        return "ARABIDOPSIS"
    if "ZEBRAFISH" in s or "DANRE" in s:
        return "ZEBRAFISH"
    
    # Fallback: last token after underscore
    if "_" in s:
        tail = s.split("_")[-1]
        if len(tail) >= 3:
            return tail
    
    return None


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


# ============================================================================
# SESSION STATE INITIALIZATION (KEEP INTACT - WORKING)
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
# FILE UPLOAD (KEEP INTACT - WORKING)
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
# LOAD FILE (KEEP INTACT - WORKING)
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
# SELECT COLUMNS (KEEP INTACT - WORKING)
# ============================================================================

st.subheader("2️⃣ Select Columns")
st.caption("Click column headers to select metadata and numerical columns")

df_preview = df_raw.head(5)

# Metadata columns
st.markdown("**📋 Metadata Columns** (ID, gene names, descriptions, species, etc.)")
event_metadata = st.dataframe(
    df_preview,
    key="metadata_selector",
    on_select="rerun",
    selection_mode="multi-column"
)

metadata_cols = event_metadata.selection.columns

if metadata_cols:
    st.session_state.metadata_columns = metadata_cols
    st.success(f"✅ Selected {len(metadata_cols)} metadata columns")
else:
    st.info("👆 Select metadata columns above")
    st.stop()

st.markdown("---")

# Numerical columns
st.markdown("**🧪 Numerical Columns** (abundance/intensity values for analysis)")
event_numerical = st.dataframe(
    df_preview,
    key="numerical_selector",
    on_select="rerun",
    selection_mode="multi-column"
)

numerical_cols = event_numerical.selection.columns

if not numerical_cols:
    st.info("👆 Select numerical abundance columns above")
    st.stop()

st.session_state.numerical_columns = numerical_cols
st.success(f"✅ Selected {len(numerical_cols)} numerical columns")

st.markdown("---")

# ============================================================================
# RENAME NUMERICAL COLUMNS (KEEP INTACT - WORKING)
# ============================================================================

st.subheader("3️⃣ Rename Numerical Columns")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    rename_style = st.selectbox(
        "Renaming strategy:",
        options=["none", "trim", "default"],
        help="none: keep original | trim: remove common prefix/suffix | default: A1, A2, B1, B2..."
    )

with col2:
    if rename_style == "default":
        replicates_per_condition = st.number_input(
            "Replicates per condition:",
            min_value=2,
            max_value=10,
            value=2,
            help="How many samples per condition? (minimum 2)"
        )
    else:
        replicates_per_condition = 2

with col3:
    if rename_style != "none":
        if rename_style == "default":
            new_names = generate_default_column_names(
                len(numerical_cols),
                replicates_per_condition=replicates_per_condition
            )
            name_mapping = {orig: new for orig, new in zip(numerical_cols, new_names)}
        else:
            name_mapping = trim_common_prefix_suffix(list(numerical_cols))
        
        mapping_df = pd.DataFrame({
            'Original': list(name_mapping.keys()),
            'Renamed': list(name_mapping.values())
        })
        st.dataframe(mapping_df, use_container_width=True, height=200)
        
        st.session_state.column_mapping = name_mapping
        st.session_state.reverse_mapping = {v: k for k, v in name_mapping.items()}
        numerical_cols_renamed = [name_mapping.get(col, col) for col in numerical_cols]
    else:
        name_mapping = {col: col for col in numerical_cols}
        numerical_cols_renamed = list(numerical_cols)
        st.session_state.column_mapping = name_mapping
        st.session_state.reverse_mapping = name_mapping
        st.info("No renaming applied - using original column names")

st.markdown("---")

# ============================================================================
# APPLY CHANGES (KEEP INTACT - WORKING)
# ============================================================================

st.subheader("4️⃣ Apply Changes")

# Filter and rename
all_cols = list(metadata_cols) + list(numerical_cols)
df_filtered = df_raw.select(all_cols)
df_filtered = df_filtered.rename(name_mapping)
numerical_cols_final = numerical_cols_renamed

# Clean numerical columns
for col in numerical_cols_final:
    df_filtered = df_filtered.with_columns([
        pl.col(col).cast(pl.Float64, strict=False)
        .fill_null(1.00)
        .alias(col)
    ])

st.info("✅ Cleaned numerical columns: NaN set to 1.00, converted to Float64")

# DATA SUMMARY
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Rows", f"{len(df_filtered):,}")
    st.metric("Metadata Cols", len(metadata_cols))

with col2:
    st.metric("Sample Cols", len(numerical_cols_final))
    conditions = len(numerical_cols_final) // replicates_per_condition
    st.metric("Conditions", conditions)

with col3:
    st.metric("Replicates/Condition", replicates_per_condition)
    st.metric("Data Type", st.session_state.data_type.upper())

st.markdown("---")

# ============================================================================
# CONFIGURE KEY COLUMNS (KEEP INTACT - WORKING)
# ============================================================================

st.subheader("5️⃣ Configure Key Columns")

col1, col2, col3 = st.columns(3)

with col1:
    id_col = st.selectbox(
        "ID Column (required):",
        options=metadata_cols,
        key=f"id_col_{st.session_state.data_type}"
    )

with col2:
    # Species column - will be auto-detected below
    species_col_options = [None] + list(metadata_cols)
    species_col_temp = st.selectbox(
        "Species Column (optional - will auto-detect):",
        options=species_col_options,
        key=f"species_col_{st.session_state.data_type}"
    )

if st.session_state.data_type == "peptide":
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
# SPECIES DETECTION - FIXED VERSION
# ============================================================================

st.subheader("🔬 Species Detection")

df_pandas_temp = df_filtered.to_pandas()

st.info("🔍 Scanning ALL metadata columns for species information...")

# Scan ALL metadata columns for species
all_species_found = {}
for col in metadata_cols:
    species_in_col = scan_column_for_species(df_pandas_temp[col])
    if species_in_col:
        all_species_found[col] = species_in_col

if all_species_found:
    st.success(f"✅ Found species in {len(all_species_found)} column(s)!")
    
    # Show results for each column with species
    for col_name, species_dict in all_species_found.items():
        total_species_count = sum(species_dict.values())
        
        with st.expander(f"📊 Column: **{col_name}** ({len(species_dict)} species, {total_species_count} total entries)", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                species_df = pd.DataFrame(
                    list(species_dict.items()),
                    columns=['Species', 'Count']
                ).sort_values('Count', ascending=False)
                
                st.dataframe(species_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.bar_chart(species_df.set_index('Species')['Count'])
    
    # Auto-select the column with most species entries
    best_col = max(all_species_found.keys(), key=lambda k: sum(all_species_found[k].values()))
    
    st.markdown("---")
    st.subheader("Select Species Column")
    
    species_col = st.selectbox(
        "Which column contains species information?",
        options=list(all_species_found.keys()),
        index=list(all_species_found.keys()).index(best_col),
        key="species_col_final",
        help=f"Auto-detected: {best_col} has the most species entries"
    )
    
    st.success(f"✅ Using **{species_col}** as species column")
    
else:
    st.warning("⚠️ No species patterns detected in ANY metadata column")
    st.info("Common patterns: HUMAN, MOUSE, YEAST, ECOLI, DROSOPHILA, ARABIDOPSIS, ZEBRAFISH, etc.")
    
    # Fallback: let user manually select
    species_col = st.selectbox(
        "Manually select species column:",
        options=metadata_cols,
        key="species_col_manual"
    )
    st.info(f"Using **{species_col}** as species column (manual selection)")

st.markdown("---")

# ============================================================================
# VALIDATION & UPLOAD (KEEP INTACT - WORKING)
# ============================================================================

st.subheader("6️⃣ Validate & Upload")

col1, col2 = st.columns(2)

with col1:
    st.write("**Validation Checks:**")
    
    checks = {
        "✅ Metadata columns selected": len(metadata_cols) > 0,
        "✅ Numerical columns selected": len(numerical_cols_final) > 0,
        "✅ ID column configured": id_col is not None,
        "✅ Species column detected": species_col is not None,
        "✅ Data loaded": df_filtered is not None,
        "✅ Samples available": len(df_filtered) > 0,
        "✅ Min 2 replicates/condition": replicates_per_condition >= 2,
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
    st.write(f"- **Species Column:** {species_col}")
    if st.session_state.data_type == 'peptide':
        st.write(f"- **Sequence Column:** {sequence_col}")
    st.write(f"- **Metadata Columns:** {len(metadata_cols)}")
    st.write(f"- **Samples:** {len(numerical_cols_final)}")
    st.write(f"- **Conditions:** {len(numerical_cols_final) // replicates_per_condition}")
    st.write(f"- **Replicates/Condition:** {replicates_per_condition}")
    st.write(f"- **Total {st.session_state.data_type.title()}s:** {len(df_filtered):,}")

st.markdown("---")

# ============================================================================
# UPLOAD BUTTON (KEEP INTACT - WORKING)
# ============================================================================

if st.button(
    f"🚀 Upload {st.session_state.data_type.upper()} Data",
    type="primary",
    use_container_width=True,
    disabled=not all_passed
):
    with st.spinner(f"Processing {st.session_state.data_type} data..."):
        try:
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
            else:
                data_obj = PeptideData(
                    raw=df_final_pandas,
                    numeric_cols=numerical_cols_final,
                    id_col=id_col,
                    species_col=species_col,
                    sequence_col=sequence_col,
                    file_path=str(uploaded_file.name)
                )
                st.session_state.peptide_data = data_obj
            
            # Store in session
            st.session_state.df_raw = df_final_pandas
            st.session_state.df_raw_polars = df_filtered
            st.session_state.numeric_cols = numerical_cols_final
            st.session_state.id_col = id_col
            st.session_state.species_col = species_col
            st.session_state.sequence_col = sequence_col if st.session_state.data_type == "peptide" else None
            st.session_state.metadata_columns = metadata_cols
            st.session_state.data_ready = True
            st.session_state.replicates_per_condition = replicates_per_condition
            
            gc.collect()
            
            st.success(f"✅ {st.session_state.data_type.upper()} data uploaded successfully!")
            st.info("Ready for analysis. Go to **Visual EDA** page to continue.")
            
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")

st.markdown("---")

st.caption("**💡 Tip:** Species scanner checks ALL metadata columns automatically. For your prot3.csv file, it will find YEAST, HUMAN, ECOLI in the 'name' column.")
