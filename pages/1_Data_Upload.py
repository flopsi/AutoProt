"""
pages/1_Data_Upload.py
Simplified Polars version - minimal, efficient data upload
"""

import streamlit as st
import polars as pl
import time
from pathlib import Path

# ============================================================================
# HELPERS
# ============================================================================

def read_file(file) -> pl.DataFrame:
    """Read uploaded file into Polars DataFrame."""
    name = file.name.lower()
    if name.endswith('.csv'):
        return pl.read_csv(file)
    elif name.endswith(('.tsv', '.txt')):
        return pl.read_csv(file, separator='\t')
    elif name.endswith('.xlsx'):
        return pl.read_excel(file)
    raise ValueError(f"Unsupported format: {name}")

def generate_column_names(n: int, replicates: int = 3) -> list:
    """Generate A1, A2, A3, B1, B2, B3, ..."""
    return [f"{chr(65 + i//replicates)}{i%replicates + 1}" for i in range(n)]

def infer_species(protein_id: str) -> str:
    """Extract species from protein ID (e.g., 'P12345_HUMAN' -> 'HUMAN')."""
    if not protein_id or '_' not in protein_id:
        return 'UNKNOWN'
    return protein_id.split('_')[-1].upper()

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(page_title="Data Upload", layout="wide")

st.title("📊 Data Upload & Configuration")

# ============================================================================
# 1. UPLOAD
# ============================================================================

st.subheader("1️⃣ Upload File")
uploaded = st.file_uploader("Choose file", type=['csv', 'tsv', 'txt', 'xlsx'])

if not uploaded:
    st.warning("⚠️ Upload a file to continue")
    st.stop()

# ============================================================================
# 2. READ
# ============================================================================

st.subheader("2️⃣ Loading...")

try:
    df = read_file(uploaded)
    st.success(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.stop()

# ============================================================================
# 3. SELECT NUMERIC COLUMNS
# ============================================================================

st.subheader("3️⃣ Select Quantitative Columns")

# Auto-detect numeric columns
numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns

# Multiselect with all numeric preselected
selected = st.multiselect(
    "Select columns for analysis",
    df.columns,
    default=numeric_cols,
    help="Choose numeric intensity/abundance columns"
)

if len(selected) < 4:
    st.warning(f"⚠️ Need ≥4 columns. Selected: {len(selected)}")
    st.stop()

# ============================================================================
# 4. CLEAN DATA
# ============================================================================

st.subheader("4️⃣ Data Cleaning")

# Replace nulls and zeros with 1.0 in selected columns
df = df.with_columns([
    pl.col(c).fill_null(1.0).replace(0.0, 1.0) for c in selected
])

st.success("✅ Replaced NaN/0 with 1.0")

# ============================================================================
# 5. RENAME COLUMNS
# ============================================================================

st.subheader("5️⃣ Rename Columns")

replicates = st.number_input("Replicates per condition", 1, 10, 3)

if st.checkbox("Auto-rename (A1, A2, B1...)", value=True):
    new_names = generate_column_names(len(selected), replicates)
    rename_map = dict(zip(selected, new_names))
    df = df.rename(rename_map)
    selected = new_names
    st.info(f"✅ Renamed: {', '.join(new_names[:6])}...")

# ============================================================================
# 6. IDENTIFY METADATA
# ============================================================================

st.subheader("6️⃣ Metadata")

# Non-numeric columns for ID/species
non_numeric = [c for c in df.columns if c not in selected]

c1, c2 = st.columns(2)

with c1:
    if non_numeric:
        id_col = st.selectbox("🔍 Protein ID", non_numeric, index=0)
    else:
        st.warning("⚠️ No ID column found")
        st.stop()

with c2:
    species_col = st.selectbox("🧬 Species (optional)", ['(None)'] + non_numeric)
    if species_col == '(None)':
        species_col = None

# ============================================================================
# 7. KEEP ONLY NEEDED COLUMNS
# ============================================================================

keep_cols = [id_col] + selected
if species_col:
    keep_cols.append(species_col)

df = df.select(keep_cols)

# Infer species if not provided
if not species_col:
    df = df.with_columns(
        pl.col(id_col).map_elements(infer_species, return_dtype=pl.Utf8).alias('species')
    )
    species_col = 'species'

# ============================================================================
# 8. PREVIEW
# ============================================================================

st.subheader("7️⃣ Preview")
st.dataframe(df.head(10), use_container_width=True, height=350)

# ============================================================================
# 9. STATS
# ============================================================================

st.subheader("8️⃣ Statistics")

n_proteins = df.shape[0]
n_samples = len(selected)
n_conditions = n_samples // replicates
species_count = df[species_col].n_unique()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Proteins", f"{n_proteins:,}")
c2.metric("Samples", n_samples)
c3.metric("Conditions", n_conditions)
c4.metric("Species", species_count)

# ============================================================================
# 10. CACHE & CONFIRM
# ============================================================================

st.markdown("---")
st.subheader("✅ Confirm & Cache")

st.info(f"""
**Summary:**
- File: `{uploaded.name}`
- Proteins: {n_proteins:,}
- Samples: {n_samples}
- Conditions: {n_conditions}
- Species: {species_count}
""")

if st.button("🎯 Cache & Continue", type="primary", use_container_width=True):
    # Cache in session state
    st.session_state.df = df
    st.session_state.numeric_cols = selected
    st.session_state.id_col = id_col
    st.session_state.species_col = species_col
    st.session_state.replicates = replicates
    
    st.success("🎉 Data cached!")
    time.sleep(1)
    st.switch_page("pages/2_Visual_EDA.py")
