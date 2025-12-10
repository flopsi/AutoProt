"""
pages/1_Data_Upload.py - DATA UPLOAD WITH SPECIES FILTER, MANUAL RENAMING,
explicit species-hint column selection, peptide-count selection,
and intensity artefact cleaning.

Manual renaming only for numerical columns + Custom species tags
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
    layout="wide",
)

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

def get_default_species_tags() -> list[str]:
    """Return default species tags to search for."""
    return [
        "HUMAN",
        "MOUSE",
        "YEAST",
        "ECOLI",
        "DROSOPHILA",
        "ARABIDOPSIS",
        "Contaminant",
    ]

def infer_species_from_text(text: str, species_tags: list[str]) -> str:
    """
    Infer species from text based on user-defined species tags.

    Args:
        text: Text to search for species identifiers.
        species_tags: List of species tags to search for (user-customizable).

    Returns:
        Species tag if found, otherwise "Other".
    """
    if pd.isna(text) or text is None:
        return "Other"

    s = str(text).upper()

    # Direct tag search
    for tag in species_tags:
        tag_upper = tag.upper()
        if tag_upper in s:
            return tag

    # UniProt-style suffixes, e.g. BRCA1_HUMAN
    if "_" in s:
        tail = s.split("_")[-1]
        if len(tail) >= 3 and tail.isalpha():
            for tag in species_tags:
                if tail == tag.upper() or tag.upper() in tail:
                    return tag
            # Treat unknown tail as species code
            return tail

    return "Other"

def extract_condition_from_sample(sample_name: str) -> str:
    """
    Extract condition letter(s) from sample name.
    Handles patterns like: A_R1, Cond_A_R2, CondA_R1, etc.
    """
    sample_name = str(sample_name).strip()

    # Pattern 1: Something_R# or Something-R#
    match = re.search(r"([a-zA-Z_]+?)[\-_]?[rR](\d+)", sample_name)
    if match:
        return match.group(1).rstrip("_").rstrip("-")

    # Pattern 2: Just letters at start
    match = re.search(r"^([A-Z]+)", sample_name)
    if match:
        return match.group(1)

    # Pattern 3: Before first underscore
    if "_" in sample_name:
        return sample_name.split("_")[0]

    return sample_name[0] if len(sample_name) > 0 else "Unknown"

def find_peptide_columns(columns: list[str], data_type: str) -> list[str]:
    """
    Find columns containing peptide or precursor information.

    For peptide data: columns with 'NrOfStrippedSequencesIdentified' or similar.
    For protein data: any numeric column that could represent peptide count.
    """
    peptide_cols: list[str] = []

    if data_type == "peptide":
        peptide_cols = [
            col
            for col in columns
            if "NrOfStrippedSequencesIdentified" in col
            or "peptide" in col.lower()
            or "precursor" in col.lower()
        ]
    else:
        peptide_cols = [
            col
            for col in columns
            if "peptide" in col.lower()
            or "precursor" in col.lower()
            or "nr_pep" in col.lower()
        ]
    return peptide_cols

def compute_peptide_counts(
    df: pd.DataFrame,
    peptide_cols: list[str],
    id_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Compute peptide counts per protein per sample.

    Detects whether columns contain:
      - Sequences (strings) → Count unique sequences per ID.
      - Integers / numeric → Use directly (or coerce).

    Returns:
        (df_with_counts, count_column_names)
    """
    df_copy = df.copy()
    count_cols: list[str] = []

    for col in peptide_cols:
        sample_values = df_copy[col].dropna()
        if len(sample_values) == 0:
            continue

        first_val = sample_values.iloc[0]

        if isinstance(first_val, str) and len(first_val) > 5:
            # String sequences (possibly ';'-separated)
            count_col = f"{col}_Count"
            df_copy[count_col] = df_copy.groupby(id_col)[col].transform(
                lambda s: len(
                    set(
                        seq
                        for v in s.dropna()
                        for seq in str(v).split(";")
                        if seq != ""
                    )
                )
            )
            count_cols.append(count_col)
        elif isinstance(first_val, (int, float)):
            # Already numeric counts
            count_col = f"{col}_Count"
            df_copy[count_col] = (
                pd.to_numeric(df_copy[col], errors="coerce").fillna(0).astype(int)
            )
            count_cols.append(count_col)
        else:
            # Try coercion to numeric
            count_col = f"{col}_Count"
            df_copy[count_col] = (
                pd.to_numeric(df_copy[col], errors="coerce").fillna(0).astype(int)
            )
            count_cols.append(count_col)

    return df_copy, count_cols

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "data_type" not in st.session_state:
    st.session_state.data_type = "protein"

if "species_tags" not in st.session_state:
    st.session_state.species_tags = get_default_species_tags()

# ============================================================================
# PAGE HEADER
# ============================================================================

st.title("📁 Data Upload")
st.markdown("---")

# ============================================================================
# 1️⃣ FILE UPLOAD
# ============================================================================

st.subheader("1️⃣ Upload File")

peptides = st.toggle(
    "Toggle if Peptide Data",
    value=st.session_state.data_type == "peptide",
    key="peptide_toggle",
)
st.session_state.data_type = "peptide" if peptides else "protein"

uploaded_file = st.file_uploader(
    f"Choose {st.session_state.data_type} data file (CSV or Excel)",
    type=["csv", "xlsx", "xls"],
    key=f"file_upload_{st.session_state.data_type}",
)

if uploaded_file is None:
    st.info("👆 Upload file to begin")
    st.stop()

# Load file with persistent cache (Polars → pandas)
try:
    file_bytes = uploaded_file.read()
    with st.spinner("Loading..."):
        if uploaded_file.name.endswith(".csv"):
            df_pl = load_csv_file(file_bytes)
        else:
            df_pl = load_excel_file(file_bytes)
        df_raw = df_pl.to_pandas()
    st.success(f"✅ Loaded {len(df_raw):,} rows × {len(df_raw.columns)} columns")
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.stop()

st.markdown("---")

# ============================================================================
# 2️⃣ SELECT COLUMNS (METADATA + NUMERICAL)
# ============================================================================

st.subheader("2️⃣ Select Columns")

df_preview = df_raw.head(5)

st.markdown("**Metadata** (ID, species, descriptions)")
event_meta = st.dataframe(
    df_preview,
    key="meta_sel",
    on_select="rerun",
    selection_mode="multi-column",
)

metadata_cols = event_meta.selection.columns if event_meta.selection.columns else []
if metadata_cols:
    st.session_state.metadata_cols = metadata_cols
    st.success(f"✅ {len(metadata_cols)} metadata columns")
elif "metadata_cols" in st.session_state:
    metadata_cols = st.session_state.metadata_cols
    st.success(f"✅ {len(metadata_cols)} metadata columns (saved)")
else:
    st.info("👆 Select metadata columns")
    st.stop()

st.markdown("---")

st.markdown("**Numerical** (abundance/intensity)")
event_num = st.dataframe(
    df_preview,
    key="num_sel",
    on_select="rerun",
    selection_mode="multi-column",
)

numerical_cols = event_num.selection.columns if event_num.selection.columns else []
if numerical_cols:
    st.session_state.numerical_cols = numerical_cols
    st.success(f"✅ {len(numerical_cols)} numerical columns")
elif "numerical_cols" in st.session_state:
    numerical_cols = st.session_state.numerical_cols
    st.success(f"✅ {len(numerical_cols)} numerical columns (saved)")
else:
    st.info("👆 Select numerical columns")
    st.stop()

st.markdown("---")

# ============================================================================
# 3️⃣ RENAME NUMERICAL COLUMNS (MANUAL ONLY)
# ============================================================================

st.subheader("3️⃣ Rename Sample Columns")

if (
    "manual_name_mapping" not in st.session_state
    or set(st.session_state.manual_name_mapping.keys()) != set(numerical_cols)
):
    st.session_state.manual_name_mapping = {col: col for col in numerical_cols}

edit_df = pd.DataFrame(
    {
        "Original": list(numerical_cols),
        "New Name": [st.session_state.manual_name_mapping[col] for col in numerical_cols],
    }
)

def update_manual_mapping() -> None:
    """Callback to update mapping when data_editor changes."""
    if "name_editor" not in st
