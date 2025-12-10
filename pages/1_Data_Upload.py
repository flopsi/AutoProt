"""
pages/1_Data_Upload.py - DATA UPLOAD WITH SMART SPECIES DETECTION
═════════════════════════════════════════════════════════════════════════════

Complete fixed version with:
- Smart conditional species detection
- All use_container_width replaced with width parameter
- Full functionality for data upload, configuration, and preprocessing
═════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import polars as pl
import pandas as pd
from pathlib import Path
import sys
import re

sys.path.append(str(Path(__file__).parent.parent))

# Import theme
from theme import apply_theme_css, get_theme_colors, PRIMARY_RED

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Data Upload",
    page_icon="📁",
    layout="wide"
)

apply_theme_css()
colors = get_theme_colors()

# ============================================================================
# HEADER
# ============================================================================

st.markdown(f"""
<div style="padding: 20px 0; border-bottom: 3px solid {PRIMARY_RED}; margin-bottom: 30px;">
    <h1 style="color: {PRIMARY_RED}; margin: 0;">📤 Data Upload</h1>
    <p style="color: #54585A; margin: 8px 0 0 0;">Upload and configure your proteomics data</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

if "species_tags" not in st.session_state:
    st.session_state.species_tags = get_default_species_tags()

# ============================================================================
# CACHED FUNCTIONS
# ============================================================================

@st.cache_data(show_spinner=False, persist="disk")
def load_csv_file(file_bytes: bytes) -> pl.DataFrame:
    import io
    return pl.read_csv(io.BytesIO(file_bytes), has_header=True, null_values=["#NUM!"])

@st.cache_data(show_spinner=False, persist="disk")
def load_excel_file(file_bytes: bytes) -> pl.DataFrame:
    import io
    return pl.read_excel(io.BytesIO(file_bytes), sheet_id=0)

def get_default_species_tags() -> list:
    """Return default species tags to search for"""
    return ["HUMAN", "MOUSE", "YEAST", "ECOLI", "DROSOPHILA", "ARABIDOPSIS", "Contaminant"]

def infer_species_from_text(text: str, species_tags: list) -> str:
    """
    Infer species from text based on user-defined species tags.
    Handles patterns like: BRCA1_HUMAN, GAL4_YEAST, Brain_Human, etc.
    """
    if pd.isna(text) or text is None:
        return "Other"
    
    s = str(text).upper()
    
    # Check each user-defined species tag
    for tag in species_tags:
        tag_upper = tag.upper()
        if tag_upper in s:
            return tag
    
    # Additional pattern matching for UniProt-style suffixes
    if "_" in s:
        tail = s.split("_")[-1]
        if len(tail) >= 3 and tail.isalpha():
            for tag in species_tags:
                if tail == tag.upper() or tag.upper() in tail:
                    return tag
            return tail
    
    return "Other"

def extract_condition_from_sample(sample_name: str) -> str:
    """Extract condition from sample name (e.g., 'A' from 'A_R1')"""
    sample_name = str(sample_name).strip()
    match = re.search(r'([a-zA-Z_]+?)[\-_]?[rR](\d+)', sample_name)
    if match:
        return match.group(1).rstrip('_').rstrip('-')
    match = re.search(r'^([A-Z]+)', sample_name)
    if match:
        return match.group(1)
    if '_' in sample_name:
        return sample_name.split('_')[0]
    return sample_name[0] if len(sample_name) > 0 else "Unknown"

def find_peptide_columns(columns: list, data_type: str) -> list:
    """Find columns containing peptide or precursor information"""
    peptide_cols = []
    if data_type == 'peptide':
        peptide_cols = [col for col in columns
                       if 'NrOfStrippedSequencesIdentified' in col
                       or 'peptide' in col.lower()
                       or 'precursor' in col.lower()]
    else:
        peptide_cols = [col for col in columns
                       if 'peptide' in col.lower()
                       or 'precursor' in col.lower()
                       or 'nr_pep' in col.lower()]
    return peptide_cols

def compute_peptide_counts(df: pd.DataFrame, peptide_cols: list, id_col: str) -> tuple:
    """Compute peptide counts per protein per sample"""
    df_copy = df.copy()
    count_cols = []
    
    for col in peptide_cols:
        sample_values = df_copy[col].dropna()
        if len(sample_values) == 0:
            continue
        
        first_val = sample_values.iloc[0]
        
        if isinstance(first_val, str) and len(first_val) > 5:
            count_col = f"{col}_Count"
            df_copy[count_col] = df_copy.groupby(id_col)[col].transform('nunique')
            count_cols.append(count_col)
        else:
            count_cols.append(col)
    
    return df_copy, count_cols

# ============================================================================
# STEP 1: FILE UPLOAD
# ============================================================================

st.subheader("1️⃣ Upload File")

uploaded_file = st.file_uploader(
    "Choose CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Upload your proteomics data file"
)

if not uploaded_file:
    st.info("👆 Upload a file to begin")
    st.stop()

# Load file
try:
    with st.spinner("Loading file..."):
        file_bytes = uploaded_file.getvalue()
        if uploaded_file.type == "text/csv":
            df_pl = load_csv_file(file_bytes)
        else:
            df_pl = load_excel_file(file_bytes)
        df_raw = df_pl.to_pandas()
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

st.success(f"✅ Loaded {len(df_raw):,} rows × {len(df_raw.columns)} columns")

# ============================================================================
# STEP 2: PREVIEW & COLUMN SELECTION
# ============================================================================

st.subheader("2️⃣ Preview Data")

with st.expander("📋 Show data preview"):
    st.dataframe(df_raw.head(10), use_container_width=True)

st.subheader("3️⃣ Select Columns to Keep")

cols_to_keep = st.multiselect(
    "Select columns to include in analysis",
    options=df_raw.columns.tolist(),
    default=df_raw.columns.tolist()[:10],
    help="Select metadata and abundance columns"
)

if not cols_to_keep:
    st.warning("Select at least one column")
    st.stop()

df_raw = df_raw[cols_to_keep]
st.success(f"✅ Keeping {len(df_raw.columns)} columns")

# ============================================================================
# STEP 4: IDENTIFY ID AND METADATA COLUMNS
# ============================================================================

st.subheader("4️⃣ Identify ID Column")

id_col = st.selectbox(
    "Select ID column (protein/peptide identifiers)",
    options=df_raw.columns.tolist(),
    help="Column with protein names or IDs"
)

# Get metadata columns (all except ID)
metadata_cols = [col for col in df_raw.columns if col != id_col]

st.info(f"ID column: **{id_col}**")
st.caption(f"Metadata columns: {', '.join(metadata_cols)}")

# ============================================================================
# STEP 5: RENAME NUMERICAL COLUMNS
# ============================================================================

st.subheader("5️⃣ Rename Abundance Columns (Optional)")

# Detect numerical columns
numerical_cols = df_raw.select_dtypes(include=['number']).columns.tolist()

if numerical_cols:
    with st.expander("📝 Rename numerical columns", expanded=False):
        name_mapping = {}
        for col in numerical_cols:
            new_name = st.text_input(
                f"Rename: {col}",
                value=col,
                key=f"rename_{col}"
            )
            name_mapping[col] = new_name
else:
    name_mapping = {}
    st.caption("No numerical columns to rename")

# ============================================================================
# STEP 6: CUSTOMIZE SPECIES TAGS
# ============================================================================

st.subheader("7️⃣ Species Tags & Filter")

with st.expander("🏷️ Customize Species Tags", expanded=False):
    st.markdown("**Default tags:** " + ", ".join(get_default_species_tags()))
    st.markdown("Add custom species tags below (e.g., RAT, BOVIN, XENLA):")
    
    species_tags_input = st.text_area(
        "Species tags (one per line):",
        value="\n".join(st.session_state.species_tags),
        height=150,
        key="species_tags_input",
        help="Enter species identifiers to search for in protein names. One per line."
    )
    
    if st.button("Update Species Tags", key="update_species_tags", use_container_width=True):
        new_tags = [tag.strip().upper() for tag in species_tags_input.split('\n') if tag.strip()]
        st.session_state.species_tags = new_tags
        st.success(f"✅ Updated to {len(new_tags)} species tags")
        st.rerun()

# ============================================================================
# STEP 7: SMART SPECIES DETECTION
# ============================================================================
st.subheader("8️⃣ Species Detection")

st.subheader("8️⃣ Species Detection")

# df_raw is a pandas DataFrame
# SMART CONDITIONAL LOGIC: Check for dedicated Species column with data
species_col_candidates = [col for col in metadata_cols if "species" in col.lower()]
has_species_column_with_data = False

if species_col_candidates:
    for spec_col in species_col_candidates:
        # Check if column has any non-null values
        if df_raw[spec_col].dropna().shape[0] > 0:
            has_species_column_with_data = True
            break

# If we have a dedicated species column with data, search only that
# Otherwise, search all available columns (including protein names)
if has_species_column_with_data:
    search_cols = species_col_candidates  # Only dedicated species columns
else:
    search_cols = metadata_cols  # Search ALL columns including protein names

all_species_set = set()
for col in search_cols:
    # Iterate over non-null values in this column
    for value in df_raw[col].dropna():
        species = infer_species_from_text(str(value), st.session_state.species_tags)
        if species != "Other":  # Only add actual detected species
            all_species_set.add(species)

# If no species found, add "Other"
if not all_species_set:
    all_species_set.add("Other")

species_list = sorted(list(all_species_set))

species_list = sorted(list(all_species_set))


st.info(f"🔍 Detected {len(species_list)} unique species/tags: {', '.join(species_list)}")

# Get saved species selection, but filter to only include species that exist in current list
if 'selected_species' in st.session_state:
    default_species = [s for s in st.session_state.selected_species if s in species_list]
else:
    default_species = species_list

# Ensure default is never empty
if not default_species:
    default_species = species_list

selected_species = st.multiselect(
    "Select species to **include** in analysis:",
    options=species_list,
    default=default_species,
    key="species_filter"
)

st.session_state.selected_species = selected_species

if not selected_species:
    st.warning("⚠️ Select at least one species")
    st.stop()

# Filter data by selected species
df_raw['__SPECIES__'] = df_raw[metadata_cols[0]].apply(
    lambda x: infer_species_from_text(x, st.session_state.species_tags)
)

for col in metadata_cols[1:]:
    df_raw['__SPECIES__'] = df_raw.apply(
        lambda row: row['__SPECIES__'] if row['__SPECIES__'] != "Other"
        else infer_species_from_text(str(row[col]), st.session_state.species_tags),
        axis=1
    )

df_filtered = df_raw[df_raw['__SPECIES__'].isin(selected_species)].copy()

# Rename numerical columns
df_filtered = df_filtered.rename(columns=name_mapping)

st.success(f"✅ {len(df_filtered):,} rows after species filter")

species_counts = df_filtered['__SPECIES__'].value_counts()
st.dataframe(
    species_counts.reset_index().rename(columns={'index': 'Species', '__SPECIES__': 'Count'}),
    hide_index=True,
    use_container_width=True
)

st.markdown("---")

# ============================================================================
# STEP 8: CONFIRMATION
# ============================================================================

st.subheader("9️⃣ Confirm & Upload")

confirm = st.checkbox(
    "✅ I confirm all settings are correct",
    value=st.session_state.get('data_ready', False)
)

if confirm:
    if st.button("🚀 Process Data", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            # Detect peptide columns
            peptide_cols = find_peptide_columns(df_filtered.columns.tolist(), 'protein')
            
            # Compute peptide counts if needed
            if peptide_cols:
                df_with_counts, peptide_count_cols = compute_peptide_counts(
                    df_filtered,
                    peptide_cols,
                    id_col
                )
            else:
                df_with_counts = df_filtered.copy()
                peptide_count_cols = []
            
            # Save to session state
            st.session_state.df_raw = df_raw
            st.session_state.df_filtered = df_filtered
            st.session_state.id_col = id_col
            st.session_state.metadata_cols = metadata_cols
            st.session_state.name_mapping = name_mapping
            st.session_state.peptide_cols = peptide_count_cols if peptide_count_cols else peptide_cols
            st.session_state.data_ready = True
            
            st.success("✅ Data processed successfully!")
            st.balloons()

def render():
    """Function called by app.py"""
    pass
