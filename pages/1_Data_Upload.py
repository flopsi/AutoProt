"""
pages/1_Data_Upload.py - PRODUCTION DATA UPLOAD
Efficient data loading with automatic species detection and caching
Streamlit 1.47+ with modern API
"""

import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import gc

from helpers.core import ProteinData, PeptideData

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Data Upload",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📁 Data Upload")
st.markdown("Load and configure your proteomics data")

# ============================================================================
# CACHED HELPER FUNCTIONS
# ============================================================================

@st.cache_data(show_spinner=False)
def load_csv_file(file_bytes: bytes, filename: str) -> pl.DataFrame:
    """Load CSV file with caching."""
    import io
    return pl.read_csv(io.BytesIO(file_bytes), has_header=True, null_values=["#NUM!"])

@st.cache_data(show_spinner=False)
def load_excel_file(file_bytes: bytes, filename: str) -> pl.DataFrame:
    """Load Excel file with caching."""
    import io
    return pl.read_excel(io.BytesIO(file_bytes), sheet_id=0)

def longest_common_prefix(strings: list) -> str:
    """Find longest common prefix."""
    if not strings:
        return ""
    s1, s2 = min(strings), max(strings)
    for i, (c1, c2) in enumerate(zip(s1, s2)):
        if c1 != c2:
            return s1[:i]
    return s1

def generate_default_column_names(n_cols: int, replicates_per_condition: int = 2) -> list:
    """Generate A1, A2, B1, B2 naming."""
    names = []
    for i in range(n_cols):
        condition_idx = i // replicates_per_condition
        replicate_num = (i % replicates_per_condition) + 1
        condition_letter = chr(ord('A') + condition_idx)
        names.append(f"{condition_letter}{replicate_num}")
    return names

def infer_species_from_text(text: str) -> str:
    """Extract species from text."""
    if pd.isna(text) or text is None:
        return None
    
    s = str(text).upper()
    
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
    
    if "_" in s:
        tail = s.split("_")[-1]
        if len(tail) >= 3:
            return tail
    
    return None

def scan_column_for_species(column_data: pd.Series) -> Dict[str, int]:
    """Count proteins per species in column."""
    species_counts = {}
    
    for value in column_data.dropna():
        species = infer_species_from_text(str(value))
        if species:
            species_counts[species] = species_counts.get(species, 0) + 1
    
    return species_counts

def trim_common_prefix_suffix(columns: list) -> dict:
    """Remove common prefix/suffix."""
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
# SESSION STATE
# ============================================================================

if 'data_type' not in st.session_state:
    st.session_state.data_type = 'protein'
if 'file_hash' not in st.session_state:
    st.session_state.file_hash = None

# ============================================================================
# FILE UPLOAD
# ============================================================================

st.subheader("1️⃣ Upload File")

peptides = st.toggle("Toggle if Peptide Data", key="peptide_toggle")
st.session_state.data_type = "peptide" if peptides else "protein"

uploaded_file = st.file_uploader(
    f"Choose {st.session_state.data_type} data file (CSV or Excel)",
    type=["csv", "xlsx", "xls"],
    key=f"file_upload_{st.session_state.data_type}"
)

if uploaded_file is None:
    st.info("👆 Upload file to begin")
    st.stop()

# Load with caching
try:
    file_bytes = uploaded_file.read()
    current_hash = hash(file_bytes)
    
    # Reset selections if new file
    if st.session_state.file_hash != current_hash:
        st.session_state.file_hash = current_hash
        if 'metadata_cols' in st.session_state:
            del st.session_state.metadata_cols
        if 'numerical_cols' in st.session_state:
            del st.session_state.numerical_cols
    
    with st.spinner("Loading..."):
        if uploaded_file.name.endswith('.csv'):
            df_raw = load_csv_file(file_bytes, uploaded_file.name)
        else:
            df_raw = load_excel_file(file_bytes, uploaded_file.name)
    
    st.success(f"✅ Loaded {len(df_raw):,} rows × {len(df_raw.columns)} columns")
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.stop()

st.markdown("---")

# ============================================================================
# SELECT COLUMNS
# ============================================================================

st.subheader("2️⃣ Select Columns")

df_preview = df_raw.head(5)

# Metadata columns
st.markdown("**Metadata** (ID, species, descriptions)")
event_meta = st.dataframe(
    df_preview, 
    key=f"meta_sel_{st.session_state.data_type}_{st.session_state.file_hash}",
    on_select="rerun", 
    selection_mode="multi-column",
    width="stretch"
)

metadata_cols = event_meta.selection.columns if event_meta.selection.columns else []

if metadata_cols:
    st.session_state.metadata_cols = metadata_cols
    st.success(f"✅ {len(metadata_cols)} metadata columns")
else:
    if 'metadata_cols' in st.session_state and st.session_state.metadata_cols:
        metadata_cols = st.session_state.metadata_cols
        st.success(f"✅ {len(metadata_cols)} metadata columns")
    else:
        st.info("👆 Click column headers to select metadata")
        st.stop()

st.markdown("---")

# Numerical columns
st.markdown("**Numerical** (abundance/intensity)")
event_num = st.dataframe(
    df_preview, 
    key=f"num_sel_{st.session_state.data_type}_{st.session_state.file_hash}",
    on_select="rerun", 
    selection_mode="multi-column",
    width="stretch"
)

numerical_cols = event_num.selection.columns if event_num.selection.columns else []

if numerical_cols:
    st.session_state.numerical_cols = numerical_cols
    st.success(f"✅ {len(numerical_cols)} numerical columns")
else:
    if 'numerical_cols' in st.session_state and st.session_state.numerical_cols:
        numerical_cols = st.session_state.numerical_cols
        st.success(f"✅ {len(numerical_cols)} numerical columns")
    else:
        st.info("👆 Click column headers to select numerical")
        st.stop()

st.markdown("---")

# ============================================================================
# RENAME NUMERICAL COLUMNS
# ============================================================================

st.subheader("3️⃣ Rename Numerical Columns")

col1, col2 = st.columns(2)

with col1:
    rename_style = st.selectbox(
        "Strategy:",
        options=["none", "trim", "default"],
        help="none: keep | trim: remove prefix/suffix | default: A1,A2,B1,B2",
        key="rename_style"
    )

with col2:
    if rename_style == "default":
        replicates_per_condition = st.number_input(
            "Replicates/condition:",
            min_value=2,
            max_value=10,
            value=2,
            key="replicates_input"
        )
    else:
        replicates_per_condition = 2

if rename_style != "none":
    if rename_style == "default":
        new_names = generate_default_column_names(len(numerical_cols), replicates_per_condition)
        name_mapping = {orig: new for orig, new in zip(numerical_cols, new_names)}
    else:
        name_mapping = trim_common_prefix_suffix(list(numerical_cols))
    
    numerical_cols_renamed = [name_mapping[col] for col in numerical_cols]
else:
    name_mapping = {col: col for col in numerical_cols}
    numerical_cols_renamed = list(numerical_cols)

st.markdown("---")

# ============================================================================
# APPLY CHANGES
# ============================================================================

st.subheader("4️⃣ Process Data")

all_cols = list(metadata_cols) + list(numerical_cols)
df_filtered = df_raw.select(all_cols).rename(name_mapping)

# Clean numerical columns
for col in numerical_cols_renamed:
    df_filtered = df_filtered.with_columns([
        pl.col(col).cast(pl.Float64, strict=False).fill_null(1.00).alias(col)
    ])

# Summary metrics
col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{len(df_filtered):,}")
col1.metric("Metadata", len(metadata_cols))
col2.metric("Samples", len(numerical_cols_renamed))
col2.metric("Conditions", len(numerical_cols_renamed) // replicates_per_condition)
col3.metric("Replicates/Cond", replicates_per_condition)
col3.metric("Type", st.session_state.data_type.upper())

st.markdown("---")

# ============================================================================
# CONFIGURE COLUMNS
# ============================================================================

st.subheader("5️⃣ Configure Columns")

if st.session_state.data_type == "peptide":
    col1, col2 = st.columns(2)
    with col1:
        id_col = st.selectbox("ID Column:", options=metadata_cols, key="id_col_select")
    with col2:
        sequence_col = st.selectbox("Sequence Column:", options=metadata_cols, key="seq_col_select")
else:
    id_col = st.selectbox("ID Column:", options=metadata_cols, key="id_col_select")
    sequence_col = None

st.markdown("---")

# ============================================================================
# SPECIES DETECTION
# ============================================================================

st.subheader("🔬 Species Detection")

df_pandas = df_filtered.to_pandas()

st.info("🔍 Scanning all metadata columns...")

all_species = {}
for col in metadata_cols:
    species_counts = scan_column_for_species(df_pandas[col])
    if species_counts:
        all_species[col] = species_counts

if all_species:
    st.success(f"✅ Found species in {len(all_species)} column(s)")
    
    for col_name, species_dict in all_species.items():
        total = sum(species_dict.values())
        
        with st.expander(f"📊 {col_name} ({len(species_dict)} species, {total} proteins)", expanded=True):
            species_df = pd.DataFrame(
                list(species_dict.items()),
                columns=['Species', 'Count']
            ).sort_values('Count', ascending=False)
            
            st.dataframe(species_df, width="stretch", hide_index=True)
    
    # Auto-select best column
    best_col = max(all_species.keys(), key=lambda k: sum(all_species[k].values()))
    
    st.markdown("**Select Species Column:**")
    species_col = st.selectbox(
        "Column:",
        options=list(all_species.keys()),
        index=list(all_species.keys()).index(best_col),
        label_visibility="collapsed",
        key="species_col_select"
    )
    
    st.success(f"✅ Using **{species_col}**")
else:
    st.warning("⚠️ No species detected")
    species_col = st.selectbox("Manually select:", options=metadata_cols, key="species_col_manual")

st.markdown("---")

# ============================================================================
# CONFIRMATION & UPLOAD
# ============================================================================

st.subheader("6️⃣ Confirm & Upload")

st.write("**Configuration Summary:**")
st.write(f"- **Type:** {st.session_state.data_type.upper()}")
st.write(f"- **Rows:** {len(df_filtered):,}")
st.write(f"- **ID Column:** {id_col}")
st.write(f"- **Species Column:** {species_col}")
if st.session_state.data_type == 'peptide':
    st.write(f"- **Sequence Column:** {sequence_col}")
st.write(f"- **Samples:** {len(numerical_cols_renamed)}")
st.write(f"- **Conditions:** {len(numerical_cols_renamed) // replicates_per_condition}")
st.write(f"- **Replicates/Condition:** {replicates_per_condition}")

st.markdown("---")

confirm = st.checkbox("✅ I confirm the configuration is correct", key="confirm_upload")

if st.button("🚀 Upload Data", type="primary", width="stretch", disabled=not confirm):
    with st.spinner("Processing..."):
        try:
            df_final = df_filtered.to_pandas()
            
            if st.session_state.data_type == 'protein':
                data_obj = ProteinData(
                    raw=df_final,
                    numeric_cols=numerical_cols_renamed,
                    id_col=id_col,
                    species_col=species_col,
                    file_path=uploaded_file.name
                )
                st.session_state.protein_data = data_obj
            else:
                data_obj = PeptideData(
                    raw=df_final,
                    numeric_cols=numerical_cols_renamed,
                    id_col=id_col,
                    species_col=species_col,
                    sequence_col=sequence_col,
                    file_path=uploaded_file.name
                )
                st.session_state.peptide_data = data_obj
            
            # Store in session
            st.session_state.df_raw = df_final
            st.session_state.df_raw_polars = df_filtered
            st.session_state.numeric_cols = numerical_cols_renamed
            st.session_state.id_col = id_col
            st.session_state.species_col = species_col
            st.session_state.sequence_col = sequence_col
            st.session_state.data_ready = True
            st.session_state.replicates_per_condition = replicates_per_condition
            
            gc.collect()
            
            st.success("✅ Upload successful!")
            st.info("→ Go to **Visual EDA** page")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
