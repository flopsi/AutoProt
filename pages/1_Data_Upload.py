"""
pages/1_Data_Upload.py

Data upload, configuration, and initial validation
Leverages existing helper functions for maximum code reuse
Persistent caching for downstream access
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import time
from helpers.io import (
    read_file,
    detect_numeric_columns,
    detect_quantitative_columns,
    detect_protein_id_column,
    detect_species_column,
    clean_species_name,
    drop_proteins_with_invalid_intensities
)
from helpers.core import ProteinData
from helpers.audit import log_event, init_audit_session

# ============================================================================
# HELPER FUNCTIONS (only those not in helpers)
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

def generate_default_column_names(n_cols: int, replicates_per_condition: int = 3) -> list:
    """
    Generate default names: A1, A2, A3, B1, B2, B3, etc.
    
    Args:
        n_cols: Total number of columns
        replicates_per_condition: Samples per condition
        
    Returns:
        List of generated names
    """
    names = []
    for i in range(n_cols):
        condition_idx = i // replicates_per_condition
        replicate_num = (i % replicates_per_condition) + 1
        condition_letter = chr(ord('A') + condition_idx)
        names.append(f"{condition_letter}{replicate_num}")
    return names

def infer_species_from_protein_name(name: str) -> str:
    """Extract species from protein name (e.g., 'PROT_HUMAN' → 'HUMAN')."""
    if pd.isna(name):
        return None
    s = str(name).upper()
    
    # Check common patterns
    if "_HUMAN" in s:
        return "HUMAN"
    if "_MOUSE" in s:
        return "MOUSE"
    if "_YEAST" in s:
        return "YEAST"
    if "_ECOLI" in s or "_ECOL" in s:
        return "ECOLI"
    
    # Fallback: last token after underscore
    if "_" in s:
        tail = s.split("_")[-1]
        if len(tail) >= 3:
            return tail
    
    return None

# ============================================================================
# PERSISTENT DATA CACHING
# Resource cached at app session level - survives page navigation
# ============================================================================

@st.cache_resource
def cache_protein_data(
    raw_df: pd.DataFrame,
    numeric_cols: list,
    species_col: str,
    species_mapping: dict,
    index_col: str,
    file_path: str,
    file_format: str
) -> ProteinData:
    """
    Create and cache ProteinData object for persistent access across pages.
    
    Cached at resource level (survives navigation & reruns).
    
    Args:
        raw_df: Processed DataFrame (after cleaning & column selection)
        numeric_cols: List of selected quantitative column names
        species_col: Species annotation column name
        species_mapping: Dict mapping protein ID → species
        index_col: Protein/peptide ID column name
        file_path: Source file path/name
        file_format: File format ("CSV", "TSV", "Excel")
        
    Returns:
        Cached ProteinData object available across all pages
    """
    protein_data = ProteinData(
        raw=raw_df.copy(),
        numeric_cols=numeric_cols.copy(),
        species_col=species_col,
        species_mapping=species_mapping.copy(),
        index_col=index_col,
        file_path=file_path,
        file_format=file_format
    )
    return protein_data

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(page_title="Data Upload", layout="wide")

# Initialize audit session
init_audit_session()

with st.sidebar:
    st.title("📊 Upload Settings")
    st.info("""
**Supported Formats:**
- CSV (.csv)
- TSV (.tsv, .txt)
- Excel (.xlsx)

**Requirements:**
- Float64 columns only for quantitative data
- ≥4 samples minimum
- One protein/gene ID column
""")

# ============================================================================
# MAIN PAGE
# ============================================================================

st.title("📊 Data Upload & Configuration")

# === STEP 1: FILE UPLOAD ===
st.subheader("1️⃣ Upload File")

uploaded_file = st.file_uploader(
    "Choose a proteomics data file",
    type=["csv", "tsv", "txt", "xlsx"],
    help="Supports CSV, TSV, and Excel formats"
)

if uploaded_file is None:
    st.warning("⚠️ Please upload a data file to continue")
    st.stop()

# === STEP 2: READ FILE (using helper) ===
st.subheader("2️⃣ Loading Data...")

try:
    filename = uploaded_file.name.lower()
    df = read_file(uploaded_file)  # Helper function with caching!
    
    # Determine format
    if filename.endswith(".xlsx"):
        file_format = "Excel"
    elif filename.endswith((".tsv", ".txt")):
        file_format = "TSV"
    else:
        file_format = "CSV"
    
    st.success(f"✅ Loaded {file_format}: {len(df):,} rows × {len(df.columns)} columns")
    
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.stop()

# === STEP 3: DATA CLEANING ===
st.subheader("3️⃣ Data Cleaning")

n_nan_before = df.isna().sum().sum()
n_zero_before = (df == 0).sum().sum()

st.info("Replacing NaN and 0 with 1.0 for log transformation...")

# Detect all numeric columns (not just float64 yet)
all_numeric_cols = detect_numeric_columns(df)

# Clean data: replace NaN and 0 with 1.0
for col in all_numeric_cols:
    df[col] = df[col].fillna(1.0)
    df.loc[df[col] == 0, col] = 1.0

c1, c2 = st.columns(2)
c1.metric("NaN Replaced", n_nan_before, delta=f"-{n_nan_before}")
c2.metric("Zeros Replaced", n_zero_before, delta=f"-{n_zero_before}")

st.success("✅ Cleaning complete")

# === STEP 4: SELECT QUANTITATIVE COLUMNS ===
st.subheader("4️⃣ Select Quantitative Columns")

# Detect float64 columns for preselection (but allow manual override)
float64_cols = detect_quantitative_columns(df)

st.info(f"**ℹ️ Found {len(float64_cols)} float64 columns** (default for quantitative data). Other numeric types can be selected manually.")

# Create editor dataframe with float64 preselection
df_cols = pd.DataFrame({
    "Select": [col in float64_cols for col in df.columns],
    "Column": df.columns.tolist(),
    "Type": [str(df[col].dtype) for col in df.columns],
    "Sample": [str(df[col].iloc[0])[:30] if len(df) > 0 else "" for col in df.columns]
})

edited = st.data_editor(
    df_cols,
    column_config={
        "Select": st.column_config.CheckboxColumn("✓", width="small"),
        "Column": st.column_config.TextColumn("Column", disabled=True),
        "Type": st.column_config.TextColumn("Type", width="small", disabled=True),
        "Sample": st.column_config.TextColumn("Sample", disabled=True)
    },
    hide_index=True,
    use_container_width=True
)

# Get selected quantitative columns
numeric_cols = edited[edited["Select"]]["Column"].tolist()

if len(numeric_cols) < 4:
    st.warning(f"⚠️ Need ≥4 columns. Selected: {len(numeric_cols)}")
    st.stop()

st.success(f"✅ Selected {len(numeric_cols)} columns for analysis")

# === STEP 5: RENAME COLUMNS ===
st.subheader("5️⃣ Rename Columns")

c1, c2 = st.columns(2)

with c1:
    replicates = st.number_input("Replicates per condition:", 1, 10, 3)

with c2:
    use_default = st.checkbox("Use default names (A1, A2, B1...)", value=True)

rename_dict = {}

if use_default:
    default_names = generate_default_column_names(len(numeric_cols), replicates)
    rename_dict = dict(zip(numeric_cols, default_names))
    st.info(f"✅ Generated names: {', '.join(default_names[:6])}...")

# Optional: allow editing
if st.checkbox("Edit names"):
    with st.expander("Edit Individual Names"):
        for idx, (old, new) in enumerate(rename_dict.items()):
            c1, c2, c3 = st.columns([2, 1, 2])
            c1.text(old)
            c2.write("→")
            custom = c3.text_input("", new, label_visibility="collapsed", key=f"rn{idx}")
            if custom != new:
                rename_dict[old] = custom

# Apply renames only to selected columns
if rename_dict:
    df = df.rename(columns=rename_dict)
    numeric_cols = [rename_dict.get(c, c) for c in numeric_cols]

# === STEP 6: IDENTIFY METADATA ===
st.subheader("6️⃣ Metadata Columns")

# Only consider columns NOT in numeric_cols (dropped columns excluded)
non_numeric = [c for c in df.columns if c not in numeric_cols]

c1, c2 = st.columns(2)

with c1:
    protein_id_col = detect_protein_id_column(df)
    
    if protein_id_col not in non_numeric and non_numeric:
        protein_id_col = non_numeric[0]
    
    if non_numeric:
        protein_id_col = st.selectbox(
            "🔍 Protein/Peptide ID",
            non_numeric,
            index=non_numeric.index(protein_id_col) if protein_id_col in non_numeric else 0
        )
    else:
        st.warning("⚠️ No non-numeric columns available. Using first column as ID.")
        protein_id_col = df.columns[0] if len(df.columns) > 0 else "ID"

with c2:
    species_col = detect_species_column(df)
    options = ["(None)"] + non_numeric
    idx = options.index(species_col) if species_col in options else 0
    species_col = st.selectbox("🧬 Species (optional)", options, index=idx)
    
    if species_col == "(None)":
        species_col = None
    elif species_col:
        df[species_col] = df[species_col].apply(clean_species_name)

# === STEP 7: KEEP ONLY NECESSARY COLUMNS ===
# Ensure dropped columns are NOT carried forward
columns_to_keep = [protein_id_col] + numeric_cols
if species_col and species_col not in columns_to_keep:
    columns_to_keep.append(species_col)

df = df[columns_to_keep].copy()  # Explicit copy; dropped columns purged

# === STEP 8: PREVIEW ===
st.subheader("7️⃣ Preview")

df_preview = df.head(10).copy()

for col in df_preview.select_dtypes(include=['float']).columns:
    df_preview[col] = df_preview[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")

st.dataframe(df_preview, use_container_width=True, height=350)

# === STEP 9: BUILD SPECIES MAPPING & CALCULATE INITIAL STATS ===
st.subheader("8️⃣ Statistics")

# Create species mapping
if species_col:
    species_series = df[species_col]
else:
    species_series = df[protein_id_col].apply(infer_species_from_protein_name)
    # Add inferred species as temporary column
    species_col = "__INFERRED_SPECIES__"
    df[species_col] = species_series

species_mapping = dict(zip(df[protein_id_col], species_series))

# Calculate stats (before optional filtering)
n_proteins = len(df)
n_samples = len(numeric_cols)
n_conditions = max(1, n_samples // replicates)
missing_count = sum((df[c].isna().sum() + (df[c] == 1.0).sum()) for c in numeric_cols)
missing_rate = (missing_count / (n_proteins * n_samples) * 100) if n_proteins > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Proteins", f"{n_proteins:,}")
c2.metric("Samples", n_samples)
c3.metric("Conditions", n_conditions)
c4.metric("Missing %", f"{missing_rate:.1f}%")

# === STEP 10: OPTIONAL FILTERING ===
st.subheader("9️⃣ Optional Filtering")

drop_invalid = st.checkbox("Drop proteins with any NaN or 1.00 intensity", value=False)

rows_dropped = 0
if drop_invalid:
    before = len(df)
    df = drop_proteins_with_invalid_intensities(df, numeric_cols, 1.0, True)
    rows_dropped = before - len(df)
    st.info(f"✅ Dropped {rows_dropped} proteins. Remaining: {len(df)}")

# === RECALCULATE STATS (after filtering) ===
n_proteins = len(df)
species_mapping = dict(zip(df[protein_id_col], df[species_col]))
missing_count = sum((df[c].isna().sum() + (df[c] == 1.0).sum()) for c in numeric_cols)
missing_rate = (missing_count / (n_proteins * n_samples) * 100) if n_proteins > 0 else 0

st.markdown("**Updated Statistics:**")
uc1, uc2, uc3, uc4 = st.columns(4)
uc1.metric("Proteins", f"{n_proteins:,}", delta=f"-{rows_dropped}")
uc2.metric("Samples", n_samples)
uc3.metric("Conditions", n_conditions)
uc4.metric("Missing %", f"{missing_rate:.1f}%")

# ============================================================================
# CONFIRMATION & DATA CACHING
# ============================================================================

st.markdown("---")
st.subheader("✅ Confirm & Save Configuration")

st.info(f"""
**Data Summary:**
- **File**: `{uploaded_file.name}`
- **Format**: {file_format}
- **Proteins**: {n_proteins:,}
- **Samples**: {n_samples}
- **Conditions**: {n_conditions}
- **Species Detected**: {len(species_series.unique())}
- **Missing Data**: {missing_rate:.1f}%

Click below to **cache your data** and proceed to analysis.
""")

if st.button("🎯 Confirm & Cache Data", type="primary", use_container_width=True):
    
    # === PERSIST DATA VIA CACHE ===
    try:
        # Call caching function to store ProteinData at resource level
        protein_data_cached = cache_protein_data(
            raw_df=df,
            numeric_cols=numeric_cols,
            species_col=species_col,
            species_mapping=species_mapping,
            index_col=protein_id_col,
            file_path=uploaded_file.name,
            file_format=file_format
        )
        
        # Also store in session state for immediate access
        st.session_state.protein_data = protein_data_cached
        st.session_state.column_mapping = rename_dict
        st.session_state.data_locked = True  # Lock flag to prevent re-uploads
        
        # Log completion event
        log_event(
            page="1_Data_Upload",
            action="data_confirmed",
            details={
                "filename": uploaded_file.name,
                "format": file_format,
                "n_proteins": n_proteins,
                "n_samples": n_samples,
                "n_conditions": n_conditions,
                "missing_rate": missing_rate,
                "nan_replaced": n_nan_before,
                "zeros_replaced": n_zero_before,
                "proteins_dropped": rows_dropped,
                "species_count": len(species_series.unique())
            }
        )
        
        st.success("🎉 Data successfully cached and locked!")
        st.info("✅ Configuration complete. Proceeding to Visual EDA...")
        
        time.sleep(1.5)
        st.switch_page("pages/2_Visual_EDA.py")
        
    except Exception as e:
        st.error(f"❌ Error caching data: {str(e)}")

# === NAVIGATION HINT ===
st.markdown("---")
st.info("**Next Step:** Navigate to **📊 Visual EDA** in the sidebar to begin analysis")
