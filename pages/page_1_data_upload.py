"""
pages/page_1_data_upload.py - OPTIMIZED Data Upload with Column Selection
===========================================================================

Key optimizations:
1. Vectorized species inference (instead of row loops)
2. Cached file loading with hash-based keys
3. Efficient condition mapping (comprehensions not loops)
4. Smart caching of computed peptide counts
5. COLUMN DESELECTION UI - User can select/deselect columns not needed downstream
6. WIDE & LONG FORMAT SUPPORT - Automatic format detection
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io
import hashlib

sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers.io import detect_numeric_columns, convert_string_numbers_to_float

# ============================================================================
# CACHED FILE LOADING - OPTIMIZED with file hash
# ============================================================================

@st.cache_data(show_spinner=False, persist="disk")
def load_csv_file(file_bytes: bytes, file_hash: str) -> pd.DataFrame:
    """Load CSV with disk persistence keyed by file hash - OPTIMIZED"""
    try:
        return pd.read_csv(
            io.BytesIO(file_bytes),
            low_memory=False,
            dtype_backend='numpy_nullable',
            na_values=['NUM!', '', 'NA', 'NaN'],
        )
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

@st.cache_data(show_spinner=False, persist="disk")
def load_excel_file(file_bytes: bytes, file_hash: str, sheet_id: int = 0) -> pd.DataFrame:
    """Load Excel with disk persistence keyed by file hash - OPTIMIZED"""
    try:
        return pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name=sheet_id,
            dtype_backend='numpy_nullable',
        )
    except Exception as e:
        st.error(f"Error loading Excel: {str(e)}")
        return None

# ============================================================================
# VECTORIZED HELPER FUNCTIONS
# ============================================================================

def get_file_hash(file_bytes: bytes) -> str:
    """Compute hash of file for cache key - OPTIMIZED"""
    return hashlib.md5(file_bytes).hexdigest()[:12]

def get_default_species_tags() -> list:
    """Return default species tags"""
    return ["HUMAN", "MOUSE", "YEAST", "ECOLI", "DROSOPHILA", "ARABIDOPSIS", "Contaminant"]

def infer_species_from_text(text: str, species_tags: list) -> str:
    """Infer species from text based on user-defined tags."""
    if pd.isna(text) or text is None:
        return "Other"
    
    s = str(text).upper()
    
    for tag in species_tags:
        tag_upper = tag.upper()
        if tag_upper in s:
            return tag
    
    if '_' in s:
        tail = s.split('_')[-1]
        if len(tail) <= 3 and tail.isalpha():
            for tag in species_tags:
                if tail == tag.upper() or tag.upper() in tail:
                    return tag
    
    return "Other"

@st.cache_data(show_spinner=False)
def extract_condition_from_sample(sample_name: str) -> str:
    """Extract condition letters from sample name."""
    import re
    sample_name = str(sample_name).strip()
    
    match = re.search(r'[a-zA-Z]+-?[rR]', sample_name)
    if match:
        return match.group()[:-1].rstrip('-').rstrip()
    
    match = re.search(r'[A-Z]', sample_name)
    if match:
        return match.group()
    
    if '_' in sample_name:
        return sample_name.split('_')[0]
    
    return sample_name[0] if len(sample_name) > 0 else "Unknown"

def find_peptide_columns(columns: list, data_type: str) -> list:
    """Find columns containing peptide or precursor information - OPTIMIZED"""
    peptide_cols = []
    
    if data_type == "peptide":
        peptide_cols = [
            col for col in columns 
            if any(keyword in col.lower() for keyword in 
                   ["NrOfStrippedSequencesIdentified", "peptide", "precursor"])
        ]
    else:
        peptide_cols = [
            col for col in columns 
            if any(keyword in col.lower() for keyword in 
                   ["peptide", "precursor", "nrpep"])
        ]
    
    return peptide_cols

@st.cache_data(show_spinner=False)
def compute_peptide_counts(
    df: pd.DataFrame,
    peptide_cols: list,
    id_col: str
) -> tuple:
    """Compute peptide counts per protein per sample - OPTIMIZED & VECTORIZED"""
    df_copy = df.copy()
    count_cols = []
    
    for col in peptide_cols:
        sample_values = df_copy[col].dropna()
        if len(sample_values) == 0:
            continue
        
        first_val = sample_values.iloc[0]
        count_col = f"{col}_Count"
        
        if isinstance(first_val, str) and len(str(first_val)) > 5:
            df_copy[count_col] = df_copy[col].apply(
                lambda x: len(set(str(x).split(';'))) if pd.notna(x) and x != '' else 0
            )
        else:
            df_copy[count_col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0).astype(int)
        
        count_cols.append(count_col)
    
    return df_copy, count_cols

def infer_species_vectorized(df: pd.DataFrame, metadata_cols: list, species_tags: list) -> pd.Series:
    """VECTORIZED species inference using pandas apply - MAJOR OPTIMIZATION"""
    species_list = pd.Series(['Other'] * len(df), index=df.index)
    
    if len(metadata_cols) > 0:
        species_list = df[metadata_cols[0]].apply(
            lambda x: infer_species_from_text(str(x), species_tags) if pd.notna(x) else 'Other'
        )
    
    for col in metadata_cols[1:]:
        mask = species_list == 'Other'
        species_list[mask] = df.loc[mask, col].apply(
            lambda x: infer_species_from_text(str(x), species_tags) if pd.notna(x) else 'Other'
        )
    
    return species_list

def detect_data_format(df: pd.DataFrame, numeric_cols: list) -> str:
    """
    Detect if data is in WIDE or LONG format.
    
    WIDE: Few columns (samples as columns), many rows (proteins)
    LONG: Many columns, pivot-like structure
    """
    n_rows = len(df)
    n_cols = len(numeric_cols)
    
    # Heuristic: if more rows than numeric columns, likely WIDE format
    # If numeric columns > rows, likely LONG format
    if n_cols > n_rows:
        return "LONG"
    else:
        return "WIDE"

# ============================================================================
# MAIN PAGE RENDER
# ============================================================================

def render():
    """Render Data Upload page"""
    st.set_page_config(page_title="Data Upload", page_icon="📤", layout="wide")
    st.title("📤 Data Upload & Preprocessing")
    st.markdown("Upload your proteomics data and configure basic settings.")
    st.markdown("---")
    
    if "datatype" not in st.session_state:
        st.session_state.datatype = "protein"
    
    if "species_tags" not in st.session_state:
        st.session_state.species_tags = get_default_species_tags()
    
    if "selected_columns" not in st.session_state:
        st.session_state.selected_columns = None
    
    # ========================================================================
    # STEP 1: DATA UPLOAD
    # ========================================================================
    
    st.header("1️⃣  Upload Data File")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Upload your proteomics intensity matrix (WIDE or LONG format)"
        )
    
    with col2:
        st.session_state.datatype = st.selectbox(
            "Data Type",
            options=["protein", "peptide"],
            help="Protein or peptide-level data"
        )
    
    if not uploaded_file:
        st.info("👈 Please upload a data file to continue")
        return
    
    # ========================================================================
    # STEP 2: LOAD DATA WITH CACHING
    # ========================================================================
    
    file_bytes = uploaded_file.read()
    file_hash = get_file_hash(file_bytes)
    
    if uploaded_file.name.endswith('.csv'):
        df_raw = load_csv_file(file_bytes, file_hash)
    else:
        df_raw = load_excel_file(file_bytes, file_hash, sheet_id=0)
    
    if df_raw is None or df_raw.empty:
        st.error("❌ Could not load file. Check format and try again.")
        return
    
    st.success(f"✅ Loaded {len(df_raw):,} rows × {len(df_raw.columns)} columns")
    
    # ========================================================================
    # STEP 3: COLUMN DETECTION & DESELECTION - NEW FEATURE
    # ========================================================================
    
    st.header("2️⃣  Column Selection & Format Detection")
    
    # VECTORIZED numeric column detection
    numeric_cols, categorical_cols = detect_numeric_columns(df_raw)
    df_raw = convert_string_numbers_to_float(df_raw, numeric_cols)
    
    # Detect data format
    data_format = detect_data_format(df_raw, numeric_cols)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Numeric Columns",
            len(numeric_cols),
            help="Abundance/sample columns"
        )
    
    with col2:
        st.metric(
            "Categorical Columns",
            len(categorical_cols),
            help="ID, metadata columns"
        )
    
    with col3:
        st.metric(
            "Data Format",
            data_format,
            help="WIDE: samples as columns | LONG: samples in rows"
        )
    
    with col4:
        st.metric(
            "Data Type",
            st.session_state.datatype.title(),
            help="Protein or peptide level"
        )
    
    # ========================================================================
    # STEP 3A: SELECT/DESELECT COLUMNS TO KEEP - NEW FEATURE
    # ========================================================================
    
    st.subheader("📋 Column Selection")
    
    tab1, tab2 = st.tabs(["Select Columns", "Preview Data"])
    
    with tab1:
        st.markdown("""
        **Choose which columns to include in analysis:**
        - Metadata columns (ID, species, etc.) are usually needed
        - Select only numeric columns you want to analyze
        - Deselect columns you don't need downstream
        """)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("🔖 Categorical Columns (Metadata)")
            selected_categorical = st.multiselect(
                "Keep these columns:",
                options=categorical_cols,
                default=categorical_cols[:min(5, len(categorical_cols))],
                key="cat_cols",
                help="ID, species, gene names, etc."
            )
        
        with col_b:
            st.subheader("📊 Numeric Columns (Samples)")
            selected_numeric = st.multiselect(
                "Keep these columns:",
                options=numeric_cols,
                default=numeric_cols,
                key="num_cols",
                help="Abundance data - select only samples you need"
            )
            
            # Quick select/deselect buttons
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            with col_btn1:
                if st.button("✅ Select All", key="sel_all"):
                    st.session_state.num_cols = numeric_cols
                    st.rerun()
            with col_btn2:
                if st.button("❌ Deselect All", key="desel_all"):
                    st.session_state.num_cols = []
                    st.rerun()
            with col_btn3:
                if st.button("🔄 Reset", key="reset_cols"):
                    st.session_state.num_cols = numeric_cols
                    st.rerun()
        
        # Keep only selected columns
        selected_all_cols = selected_categorical + selected_numeric
        df_raw = df_raw[selected_all_cols].copy()
        
        st.session_state.selected_columns = selected_all_cols
        
        st.success(f"✅ Keeping {len(selected_categorical)} metadata + {len(selected_numeric)} sample columns")
    
    with tab2:
        st.subheader("Data Preview")
        st.dataframe(df_raw.head(10), use_container_width=True, height=400)
    
    # ========================================================================
    # STEP 4: ID COLUMN SELECTION
    # ========================================================================
    
    st.header("3️⃣  Configure ID & Metadata")
    
    # Filter to only selected categorical columns
    selected_categorical = [c for c in selected_categorical if c in df_raw.columns]
    
    if not selected_categorical:
        st.error("❌ Must select at least one categorical column")
        return
    
    id_col = st.selectbox(
        "Select ID Column (Protein/Peptide name)",
        options=selected_categorical,
        index=0 if selected_categorical else None,
        help="Column with protein/peptide identifiers"
    )
    
    if not id_col or id_col not in df_raw.columns:
        st.error("❌ Must select valid ID column")
        return
    
    # ========================================================================
    # STEP 5: SPECIES TAGGING - VECTORIZED & OPTIMIZED
    # ========================================================================
    
    st.header("4️⃣  Species Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        species_tags_input = st.text_area(
            "Species Tags (comma-separated)",
            value=", ".join(st.session_state.species_tags),
            height=100,
            help="Tags to search for in metadata columns"
        )
    
    with col2:
        if st.button("Update Tags", use_container_width=True):
            new_tags = [tag.strip().upper() for tag in species_tags_input.split(',') if tag.strip()]
            st.session_state.species_tags = new_tags
            st.success(f"✅ Updated to {len(new_tags)} species tags")
            st.rerun()
    
    # VECTORIZED species inference
    metadata_cols = [c for c in selected_categorical if c != id_col]
    df_raw['SPECIES'] = infer_species_vectorized(df_raw, metadata_cols, st.session_state.species_tags)
    
    species_list = sorted(list(df_raw['SPECIES'].unique()))
    st.info(f"🔍 Detected {len(species_list)} unique species: {', '.join(species_list)}")
    
    selected_species = st.multiselect(
        "Select species to include",
        options=species_list,
        default=species_list[:min(3, len(species_list))],
        help="Choose which species to keep"
    )
    
    if not selected_species:
        st.warning("⚠️ Select at least one species")
        return
    
    df_filtered = df_raw[df_raw['SPECIES'].isin(selected_species)].copy()
    species_counts = df_filtered['SPECIES'].value_counts()
    
    st.write("Species composition:")
    st.bar_chart(species_counts)
    
    # ========================================================================
    # STEP 6: NUMERIC COLUMNS & CONDITION MAPPING
    # ========================================================================
    
    st.header("5️⃣  Sample Configuration")
    
    # Get numeric columns from selected columns
    numeric_cols_final = [c for c in selected_numeric if c in df_raw.columns]
    
    if not numeric_cols_final:
        st.error("❌ Must have at least one numeric column")
        return
    
    # VECTORIZED condition mapping
    sample_to_condition = {
        col: extract_condition_from_sample(col)
        for col in numeric_cols_final
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Sample → Condition Mapping:**")
        cond_df = pd.DataFrame(
            list(sample_to_condition.items()),
            columns=["Sample", "Condition"]
        )
        st.dataframe(cond_df, use_container_width=True, hide_index=True)
    
    with col2:
        condition_counts = cond_df['Condition'].value_counts()
        st.write("**Samples per Condition:**")
        st.bar_chart(condition_counts)
    
    # ========================================================================
    # STEP 7: PEPTIDE COUNTS (if applicable)
    # ========================================================================
    
    st.header("6️⃣  Peptide Count Configuration")
    
    peptide_cols_detected = find_peptide_columns(list(df_filtered.columns), st.session_state.datatype)
    
    if peptide_cols_detected:
        st.info(f"Found {len(peptide_cols_detected)} peptide-related columns")
        
        with st.spinner("Computing peptide counts..."):
            df_with_counts, peptide_count_cols = compute_peptide_counts(
                df_filtered, peptide_cols_detected, id_col
            )
        
        st.success(f"✅ Computed peptide counts for {len(peptide_count_cols)} samples")
        st.session_state.peptide_count_cols = peptide_count_cols
    else:
        df_with_counts = df_filtered
        st.session_state.peptide_count_cols = []
    
    # ========================================================================
    # STEP 8: SAVE & CONFIRM
    # ========================================================================
    
    st.header("7️⃣  Confirm Upload")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Proteins/Peptides", f"{len(df_with_counts):,}")
        st.metric("Samples", len(numeric_cols_final))
    
    with col2:
        st.metric("Conditions", len(set(sample_to_condition.values())))
        st.metric("Species", len(selected_species))
    
    with col3:
        st.metric("Data Format", data_format)
        st.metric("Columns Selected", len(selected_all_cols))
    
    confirm = st.checkbox(
        "✅ I confirm the settings and am ready to proceed",
        value=False
    )
    
    if confirm and st.button("Process & Save Data", type="primary", use_container_width=True):
        with st.spinner("Processing data..."):
            st.session_state.df_raw = df_with_counts
            st.session_state.numeric_cols = numeric_cols_final
            st.session_state.id_col = id_col
            st.session_state.species_col = "SPECIES"
            st.session_state.peptide_cols = peptide_cols_detected
            st.session_state.sample_to_condition = sample_to_condition
            st.session_state.selected_species = selected_species
            st.session_state.data_format = data_format
            st.session_state.data_ready = True
            
            st.success("✅ Data loaded and ready for analysis!")
            st.balloons()
            st.info("👉 Proceed to **Visual EDA** page for initial exploration")
            
            st.markdown(
                """
                <div style="padding: 20px; background: #f0f2f6; border-radius: 10px;">
                <strong>Next Step:</strong> Go to the <strong>Visual EDA</strong> page to explore 
                your data distribution and quality.
                </div>
                """,
                unsafe_allow_html=True
            )
