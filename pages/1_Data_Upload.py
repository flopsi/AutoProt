"""
pages/1_Data_Upload.py - DATA UPLOAD WITH SPECIES FILTER
"""

import streamlit as st
import polars as pl
import pandas as pd
from pathlib import Path
from typing import Dict

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Data Upload",
    page_icon="📁",
    layout="wide"
)

st.title("📁 Data Upload")

# ============================================================================
# CACHED FUNCTIONS
# ============================================================================

@st.cache_data(show_spinner=False)
def load_csv_file(file_bytes: bytes) -> pl.DataFrame:
    import io
    return pl.read_csv(io.BytesIO(file_bytes), has_header=True, null_values=["#NUM!"])

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

# Load file
try:
    file_bytes = uploaded_file.read()
    current_hash = hash(file_bytes)
    
    if st.session_state.file_hash != current_hash:
        st.session_state.file_hash = current_hash
        for key in ['metadata_cols', 'numerical_cols', 'selected_species']:
            if key in st.session_state:
                del st.session_state[key]
    
    with st.spinner("Loading..."):
        if uploaded_file.name.endswith('.csv'):
            df_raw = load_csv_file(file_bytes)
        else:
            import io
            df_raw = pl.read_excel(io.BytesIO(file_bytes), sheet_id=0)
    
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
    key=f"meta_sel_{st.session_state.file_hash}",
    on_select="rerun",
    selection_mode="multi-column"
)

metadata_cols = event_meta.selection.columns if event_meta.selection.columns else []
if metadata_cols:
    st.session_state.metadata_cols = metadata_cols
    st.success(f"✅ {len(metadata_cols)} metadata columns")
else:
    if 'metadata_cols' in st.session_state:
        metadata_cols = st.session_state.metadata_cols
        st.success(f"✅ {len(metadata_cols)} metadata columns")
    else:
        st.info("👆 Select metadata columns")
        st.stop()

st.markdown("---")

# Numerical
st.markdown("**Numerical** (abundance/intensity)")
event_num = st.dataframe(
    df_preview,
    key=f"num_sel_{st.session_state.file_hash}",
    on_select="rerun",
    selection_mode="multi-column"
)

numerical_cols = event_num.selection.columns if event_num.selection.columns else []
if numerical_cols:
    st.session_state.numerical_cols = numerical_cols
    st.success(f"✅ {len(numerical_cols)} numerical columns")
else:
    if 'numerical_cols' in st.session_state:
        numerical_cols = st.session_state.numerical_cols
        st.success(f"✅ {len(numerical_cols)} numerical columns")
    else:
        st.info("👆 Select numerical columns")
        st.stop()

st.markdown("---")

# ============================================================================
# EXPERIMENTAL DESIGN
# ============================================================================

st.subheader("3️⃣ Experimental Design")

replicates_per_condition = st.number_input(
    "Replicates per condition:",
    min_value=3,
    max_value=20,
    value=3,
    step=1,
    key="replicates_input",
    help="Number of biological replicates per experimental condition (minimum 3)"
)

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

st.subheader("4️⃣ Configure Columns")

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
# SPECIES DETECTION & FILTER
# ============================================================================

st.subheader("5️⃣ Species Filter")

df_pandas = df_raw.to_pandas()

# Detect species across all metadata columns
all_species_set = set()
for col in metadata_cols:
    for value in df_pandas[col].dropna():
        species = infer_species_from_text(str(value))
        all_species_set.add(species)

species_list = sorted(list(all_species_set))

st.info(f"🔍 Detected {len(species_list)} unique species/tags: {', '.join(species_list)}")

# Species multiselect
selected_species = st.multiselect(
    "Select species to **include** in analysis:",
    options=species_list,
    default=species_list,
    key="species_filter"
)

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

st.success(f"✅ {len(df_filtered):,} rows after species filter")

# Display species counts
species_counts = df_filtered['__SPECIES__'].value_counts()
st.dataframe(
    species_counts.reset_index().rename(columns={'index': 'Species', '__SPECIES__': 'Species', 'count': 'Count'}),
    hide_index=True
)

st.markdown("---")

# ============================================================================
# CONFIRMATION
# ============================================================================

st.subheader("6️⃣ Confirm & Upload")

confirm = st.checkbox("✅ I confirm all settings are correct")

if confirm:
    if st.button("🚀 Process Data", type="primary"):
        # Store in session state
        st.session_state.df_raw = df_filtered
        st.session_state.numeric_cols = list(numerical_cols)
        st.session_state.id_col = id_col
        st.session_state.species_col = '__SPECIES__'
        st.session_state.data_type = st.session_state.data_type
        st.session_state.replicates_per_condition = replicates_per_condition
        st.session_state.data_ready = True
        
        st.success("✅ Data uploaded successfully! Proceed to **📊 Visual EDA**")
        st.balloons()
else:
    st.info("👆 Check the box to enable upload")
