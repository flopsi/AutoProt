"""
pages/1_Data_Upload.py - DATA UPLOAD WITH SPECIES FILTER AND MANUAL RENAMING
Manual renaming only for numerical columns
"""

import streamlit as st
import polars as pl
import pandas as pd
from pathlib import Path
import sys
import re
sys.path.append(str(Path(__file__).parent.parent))

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Data Upload",
    page_icon="📁",
    layout="wide"
)

# ============================================================================
# HEADER
# ============================================================================

st.title("📁 Data Upload")
st.markdown("---")

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

def infer_species_from_text(text: str) -> str:
    if pd.isna(text) or text is None:
        return "Other"
    
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
    if "CONTA" in s:
        return "Contaminant"
    
    if "_" in s:
        tail = s.split("_")[-1]
        if len(tail) >= 3 and tail.isalpha():
            return tail
    
    return "Other"

def extract_condition_from_sample(sample_name: str) -> str:
    """
    Extract condition letter(s) from sample name.
    Handles patterns like: A_R1, Cond_A_R2, CondA_R1, etc.
    """
    sample_name = str(sample_name).strip()
    
    # Pattern 1: Something_R# or Something-R#
    match = re.search(r'([a-zA-Z_]+?)[\-_]?[rR](\d+)', sample_name)
    if match:
        return match.group(1).rstrip('_').rstrip('-')
    
    # Pattern 2: Just a letter or letters at start
    match = re.search(r'^([A-Z]+)', sample_name)
    if match:
        return match.group(1)
    
    # Pattern 3: Something before first underscore
    if '_' in sample_name:
        return sample_name.split('_')[0]
    
    return sample_name[0] if len(sample_name) > 0 else "Unknown"

def find_peptide_columns(columns: list, data_type: str) -> list:
    """
    Find columns containing peptide or precursor information.
    For peptide data: columns with 'NrOfStrippedSequencesIdentified'
    For protein data: any numeric column that could represent peptide count
    """
    peptide_cols = []
    
    if data_type == 'peptide':
        # Look for stripped sequences identified columns
        peptide_cols = [col for col in columns 
                       if 'NrOfStrippedSequencesIdentified' in col 
                       or 'peptide' in col.lower() 
                       or 'precursor' in col.lower()]
    else:
        # For protein data, look for single columns with peptide count
        peptide_cols = [col for col in columns 
                       if 'peptide' in col.lower() 
                       or 'precursor' in col.lower()
                       or 'nr_pep' in col.lower()]
    
    return peptide_cols

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'data_type' not in st.session_state:
    st.session_state.data_type = 'protein'

# ============================================================================
# FILE UPLOAD
# ============================================================================

st.subheader("1️⃣ Upload File")

peptides = st.toggle("Toggle if Peptide Data", 
                     value=st.session_state.data_type == 'peptide',
                     key="peptide_toggle")
st.session_state.data_type = "peptide" if peptides else "protein"

uploaded_file = st.file_uploader(
    f"Choose {st.session_state.data_type} data file (CSV or Excel)",
    type=["csv", "xlsx", "xls"],
    key=f"file_upload_{st.session_state.data_type}"
)

if uploaded_file is None:
    st.info("👆 Upload file to begin")
    st.stop()

# Load file with persistent cache
try:
    file_bytes = uploaded_file.read()
    
    with st.spinner("Loading..."):
        if uploaded_file.name.endswith('.csv'):
            df_raw = load_csv_file(file_bytes)
        else:
            df_raw = load_excel_file(file_bytes)
    
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

# Metadata
st.markdown("**Metadata** (ID, species, descriptions)")

event_meta = st.dataframe(
    df_preview,
    key=f"meta_sel",
    on_select="rerun",
    selection_mode="multi-column"
)

metadata_cols = event_meta.selection.columns if event_meta.selection.columns else []
if metadata_cols:
    st.session_state.metadata_cols = metadata_cols
    st.success(f"✅ {len(metadata_cols)} metadata columns")
elif 'metadata_cols' in st.session_state:
    metadata_cols = st.session_state.metadata_cols
    st.success(f"✅ {len(metadata_cols)} metadata columns (saved)")
else:
    st.info("👆 Select metadata columns")
    st.stop()

st.markdown("---")

# Numerical
st.markdown("**Numerical** (abundance/intensity)")

event_num = st.dataframe(
    df_preview,
    key=f"num_sel",
    on_select="rerun",
    selection_mode="multi-column"
)

numerical_cols = event_num.selection.columns if event_num.selection.columns else []
if numerical_cols:
    st.session_state.numerical_cols = numerical_cols
    st.success(f"✅ {len(numerical_cols)} numerical columns")
elif 'numerical_cols' in st.session_state:
    numerical_cols = st.session_state.numerical_cols
    st.success(f"✅ {len(numerical_cols)} numerical columns (saved)")
else:
    st.info("👆 Select numerical columns")
    st.stop()

st.markdown("---")

# ============================================================================
# RENAME NUMERICAL COLUMNS (MANUAL ONLY)
# ============================================================================

st.subheader("3️⃣ Rename Sample Columns")

# Initialize manual mapping if not exists OR if columns changed
if 'manual_name_mapping' not in st.session_state or set(st.session_state.manual_name_mapping.keys()) != set(numerical_cols):
    st.session_state.manual_name_mapping = {col: col for col in numerical_cols}

# Create editable dataframe from session state
edit_df = pd.DataFrame({
    'Original': list(numerical_cols),
    'New Name': [st.session_state.manual_name_mapping[col] for col in numerical_cols]
})

# Use on_change callback to persist changes immediately
def update_manual_mapping():
    """Callback to update mapping when data_editor changes"""
    if 'name_editor' in st.session_state:
        edited = st.session_state['name_editor']
        if 'edited_rows' in edited and edited['edited_rows']:
            for idx, changes in edited['edited_rows'].items():
                if 'New Name' in changes:
                    original = edit_df.iloc[idx]['Original']
                    st.session_state.manual_name_mapping[original] = changes['New Name']

edited_df = st.data_editor(
    edit_df,
    key="name_editor",
    hide_index=True,
    use_container_width=True,
    on_change=update_manual_mapping,
    column_config={
        "Original": st.column_config.TextColumn("Original", disabled=True),
        "New Name": st.column_config.TextColumn("New Name", help="Edit to rename")
    }
)

# Get final mapping from session state
name_mapping = st.session_state.manual_name_mapping
numerical_cols_renamed = [name_mapping[col] for col in numerical_cols]

st.session_state.name_mapping = name_mapping
st.session_state.numerical_cols_renamed = numerical_cols_renamed

st.markdown("---")

# ============================================================================
# PEPTIDES PER PROTEIN COLUMNS
# ============================================================================

st.subheader("4️⃣ Peptides/Precursors per Protein Column")

# Auto-detect peptide columns
all_cols = list(metadata_cols) + list(numerical_cols)
auto_peptide_cols = find_peptide_columns(all_cols, st.session_state.data_type)

if auto_peptide_cols:
    st.info(f"🔍 Found potential peptide/precursor columns: {', '.join(auto_peptide_cols[:3])}")
    if len(auto_peptide_cols) > 3:
        st.caption(f"... and {len(auto_peptide_cols) - 3} more")
else:
    st.info("📝 No peptide/precursor columns auto-detected. Select manually below.")

# Allow user to select peptide columns
peptide_cols_selection = st.multiselect(
    f"Select column(s) with peptide/precursor info per protein per run:",
    options=all_cols,
    default=auto_peptide_cols,
    key="peptide_cols_select",
    help="For peptide data: NrOfStrippedSequencesIdentified columns | For protein data: single column with integer count"
)

if peptide_cols_selection:
    st.session_state.peptide_cols = peptide_cols_selection
    st.success(f"✅ Selected {len(peptide_cols_selection)} column(s) for filtering")
    
    # Show sample values
    st.markdown("**Sample values:**")
    sample_cols_to_show = peptide_cols_selection[:3]
    sample_data = df_raw.select(sample_cols_to_show).head(5).to_pandas()
    st.dataframe(sample_data, hide_index=True)
else:
    st.info("👆 Select at least one column for peptide information")
    if 'peptide_cols' in st.session_state:
        del st.session_state.peptide_cols
    st.stop()

st.markdown("---")

# ============================================================================
# EXPERIMENTAL DESIGN
# ============================================================================

st.subheader("5️⃣ Experimental Design")

default_reps = st.session_state.get('replicates_per_condition', 3)

replicates_per_condition = st.number_input(
    "Replicates per condition:",
    min_value=3,
    max_value=20,
    value=default_reps,
    step=1,
    key="replicates_input",
    help="Number of biological replicates per experimental condition (minimum 3)"
)

st.session_state.replicates_per_condition = replicates_per_condition

num_conditions = len(numerical_cols) // replicates_per_condition
remaining = len(numerical_cols) % replicates_per_condition

col1, col2, col3 = st.columns(3)
col1.metric("Total Samples", len(numerical_cols))
col2.metric("Conditions", num_conditions)
col3.metric("Replicates/Condition", replicates_per_condition)

if remaining > 0:
    st.warning(f"⚠️ {remaining} sample(s) don't fit evenly into conditions. Check your design.")

st.markdown("---")

# ============================================================================
# CONFIGURE COLUMNS
# ============================================================================

st.subheader("6️⃣ Configure Columns")

default_id_idx = 0
if 'id_col' in st.session_state and st.session_state.id_col in metadata_cols:
    default_id_idx = metadata_cols.index(st.session_state.id_col)

if st.session_state.data_type == "peptide":
    col1, col2 = st.columns(2)
    with col1:
        id_col = st.selectbox("ID Column:", options=metadata_cols, 
                              index=default_id_idx, key="id_col_select")
    with col2:
        default_seq_idx = 0
        if 'sequence_col' in st.session_state and st.session_state.sequence_col in metadata_cols:
            default_seq_idx = metadata_cols.index(st.session_state.sequence_col)
        sequence_col = st.selectbox("Sequence Column:", options=metadata_cols,
                                   index=default_seq_idx, key="seq_col_select")
else:
    id_col = st.selectbox("ID Column:", options=metadata_cols,
                         index=default_id_idx, key="id_col_select")
    sequence_col = None

st.session_state.id_col = id_col
if sequence_col:
    st.session_state.sequence_col = sequence_col

st.markdown("---")

# ============================================================================
# SPECIES DETECTION & FILTER
# ============================================================================

st.subheader("7️⃣ Species Filter")

df_pandas = df_raw.to_pandas()

# Detect species across all metadata columns
all_species_set = set()
for col in metadata_cols:
    for value in df_pandas[col].dropna():
        species = infer_species_from_text(str(value))
        all_species_set.add(species)

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
df_pandas['__SPECIES__'] = df_pandas[metadata_cols[0]].apply(infer_species_from_text)
for col in metadata_cols[1:]:
    df_pandas['__SPECIES__'] = df_pandas.apply(
        lambda row: row['__SPECIES__'] if row['__SPECIES__'] != "Other" 
        else infer_species_from_text(str(row[col])),
        axis=1
    )

df_filtered = df_pandas[df_pandas['__SPECIES__'].isin(selected_species)].copy()

# Rename numerical columns
df_filtered = df_filtered.rename(columns=name_mapping)

st.success(f"✅ {len(df_filtered):,} rows after species filter")

species_counts = df_filtered['__SPECIES__'].value_counts()
st.dataframe(
    species_counts.reset_index().rename(columns={'index': 'Species', '__SPECIES__': 'Species', 'count': 'Count'}),
    hide_index=True
)

st.markdown("---")

# ============================================================================
# CONFIRMATION
# ============================================================================

st.subheader("8️⃣ Confirm & Upload")

confirm = st.checkbox("✅ I confirm all settings are correct", 
                     value=st.session_state.get('data_ready', False))

if confirm:
    if st.button("🚀 Process Data", type="primary"):
        # Create sample to condition mapping for EDA
        sample_to_condition = {
            renamed: extract_condition_from_sample(renamed) 
            for renamed in numerical_cols_renamed
        }
        
        st.session_state.df_raw = df_filtered
        st.session_state.numeric_cols = numerical_cols_renamed
        st.session_state.species_col = '__SPECIES__'
        st.session_state.sample_to_condition = sample_to_condition
        st.session_state.peptide_cols = peptide_cols_selection
        st.session_state.data_ready = True
        
        st.success("✅ Data uploaded successfully! Proceed to **📊 Visual EDA**")
        st.balloons()
else:
    st.info("👆 Check the box to enable upload")

st.markdown("---")
st.markdown("---")

# ============================================================================
# RESET BUTTONS (BOTTOM)
# ============================================================================

def reset_all():
    """Clear all session state and restart from upload page"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

def reset_current_page():
    """Clear only current page's session state"""
    keys_to_delete = [
        'file_hash', 'metadata_cols', 'numerical_cols', 'selected_species',
        'df_raw', 'numeric_cols', 'id_col', 'species_col', 'data_type',
        'replicates_per_condition', 'data_ready', 'manual_name_mapping',
        'sample_to_condition', 'peptide_cols', 'sequence_col'
    ]
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

col1, col2, col3 = st.columns([3, 1, 1])
with col2:
    if st.button("🔄 Reset Page", help="Clear this page and restart", key="reset_page_bottom"):
        reset_current_page()
with col3:
    if st.button("🗑️ Reset All", help="Clear everything and start over", type="secondary", key="reset_all_bottom"):
        reset_all()
