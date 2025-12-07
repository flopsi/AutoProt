"""
pages/1_Data_Upload.py
Upload protein and/or peptide data with tabs - MEMORY OPTIMIZED WITH SHARED CONFIG
"""

import streamlit as st
import polars as pl
import polars.selectors as cs
import time
import gc

# ============================================================================
# HELPERS
# ============================================================================

def read_file(file) -> pl.DataFrame:
    """Read uploaded file into Polars DataFrame with robust error handling."""
    name = file.name.lower()
    
    try:
        if name.endswith('.csv'):
            return pl.read_csv(
                file,
                null_values=["#NUM!", "#N/A", "#VALUE!", "#REF!", "#DIV/0!", "#NAME?", "#NULL!", ""],
                ignore_errors=True,
                infer_schema_length=10000
            )
        elif name.endswith(('.tsv', '.txt')):
            return pl.read_csv(
                file,
                separator='\t',
                null_values=["#NUM!", "#N/A", "#VALUE!", "#REF!", "#DIV/0!", "#NAME?", "#NULL!", ""],
                ignore_errors=True,
                infer_schema_length=10000
            )
        elif name.endswith('.xlsx'):
            return pl.read_excel(file)
        else:
            raise ValueError(f"Unsupported format: {name}")
    except Exception as e:
        raise ValueError(f"Error reading {name}: {str(e)}")

def generate_column_names(n: int, replicates: int = 3) -> list:
    """Generate A1, A2, A3, B1, B2, B3, ..."""
    return [f"{chr(65 + i//replicates)}{i%replicates + 1}" for i in range(n)]

def clear_temp_session_data():
    """Remove temporary data from session state to free memory."""
    keys_to_remove = [k for k in st.session_state.keys() if k.startswith(('temp_', 'plot_', 'preview_'))]
    for key in keys_to_remove:
        del st.session_state[key]
    gc.collect()

def initialize_shared_config():
    """Initialize shared configuration in session state."""
    if 'shared_replicates' not in st.session_state:
        st.session_state.shared_replicates = 3
    if 'shared_species_tags' not in st.session_state:
        st.session_state.shared_species_tags = "HUMAN, YEAST, ECOLI"
    if 'shared_autorename' not in st.session_state:
        st.session_state.shared_autorename = True

def tag_species(text: str, tags: list, has_others: bool) -> str:
    """Tag species based on keyword matching."""
    if not text or not isinstance(text, str):
        return 'OTHERS' if has_others else 'UNKNOWN'
    
    text_upper = text.upper()
    
    # Check each tag
    for tag in tags:
        if tag in text_upper:
            return tag
    
    # No match found
    return 'OTHERS' if has_others else 'UNKNOWN'

def process_dataset(uploaded_file, data_type: str, key_prefix: str):
    """Process uploaded dataset (protein or peptide) with shared configuration."""
    
    st.subheader(f"1️⃣ Upload {data_type.title()} File")
    
    if not uploaded_file:
        st.info(f"👆 Upload {data_type} data")
        return False
    
    # Load file
    try:
        df = read_file(uploaded_file)
        st.success(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    except ValueError as e:
        st.error(f"❌ {e}")
        return False
    
    st.markdown("---")
    
    # ============================================================================
    # SELECT NUMERIC COLUMNS
    # ============================================================================
    
    st.subheader("2️⃣ Select Quantitative Columns")
    
    numeric_cols = df.select(cs.numeric()).columns
    
    col_data = []
    for col in df.columns:
        is_numeric = col in numeric_cols
        last_20 = col[-20:] if len(col) > 20 else col
        sample_val = str(df[col][0])[:30] if df.shape[0] > 0 else ""
        dtype = str(df[col].dtype)
        
        col_data.append({
            'Select': is_numeric,
            'Column (last 20)': last_20,
            'Full Name': col,
            'Type': dtype,
            'Sample': sample_val
        })
    
    df_cols = pl.DataFrame(col_data)
    
    st.info(f"**ℹ️ Auto-detected {len(numeric_cols)} numeric columns.** Review and adjust selection below.")
    
    edited = st.data_editor(
        df_cols.to_pandas(),
        column_config={
            'Select': st.column_config.CheckboxColumn('✓', width='small'),
            'Column (last 20)': st.column_config.TextColumn('Column (last 20)', width='medium'),
            'Full Name': st.column_config.TextColumn('Full Name', disabled=True),
            'Type': st.column_config.TextColumn('Type', width='small', disabled=True),
            'Sample': st.column_config.TextColumn('Sample', disabled=True)
        },
        hide_index=True,
        width='stretch',
        height=400,
        key=f"{key_prefix}_col_editor"
    )
    
    selected = edited[edited['Select']]['Full Name'].tolist()
    
    # Free memory
    del col_data, df_cols
    gc.collect()
    
    if len(selected) < 4:
        st.warning(f"⚠️ Need ≥4 columns. Selected: {len(selected)}")
        return False
    
    st.success(f"✅ Selected {len(selected)} columns for analysis")
    st.markdown("---")
    
    # ============================================================================
    # DATA QUALITY CHECK
    # ============================================================================
    
    st.subheader("3️⃣ Data Quality Check")
    
    missing_stats = []
    
    for c in selected:
        n_null = df[c].null_count()
        
        try:
            n_nan_string = df.filter(
                pl.col(c).cast(pl.Utf8).str.to_uppercase() == "NAN"
            ).shape[0]
        except:
            n_nan_string = 0
        
        n_zero = df.filter(pl.col(c) == 0.0).shape[0]
        n_one = df.filter(pl.col(c) == 1.0).shape[0]
        
        missing_stats.append({
            'column': c,
            'null': n_null,
            'nan_string': n_nan_string,
            'zero': n_zero,
            'one': n_one
        })
    
    n_null = sum(s['null'] for s in missing_stats)
    n_nan_string = sum(s['nan_string'] for s in missing_stats)
    n_zero = sum(s['zero'] for s in missing_stats)
    n_one = sum(s['one'] for s in missing_stats)
    total_missing = n_null + n_nan_string + n_zero + n_one
    total_values = df.shape[0] * len(selected)
    missing_pct = total_missing / total_values * 100 if total_values > 0 else 0
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Null", f"{n_null:,}")
    c2.metric("'NaN' string", f"{n_nan_string:,}")
    c3.metric("Zero", f"{n_zero:,}")
    c4.metric("Value = 1.0", f"{n_one:,}")
    c5.metric("Missing %", f"{missing_pct:.1f}%")
    
    st.info("**Note:** All missing values will be normalized to 1.0 for log2 transformation.")
    
    df = df.with_columns([
        pl.when(pl.col(c).is_null())
        .then(1.0)
        .when(pl.col(c) == 0.0)
        .then(1.0)
        .otherwise(pl.col(c))
        .alias(c)
        for c in selected
    ])
    
    # Free memory
    del missing_stats
    gc.collect()
    
    st.success("✅ All missing values normalized to 1.0")
    st.markdown("---")
    
    # ============================================================================
    # RENAME COLUMNS - USE SHARED CONFIG
    # ============================================================================
    
    st.subheader("4️⃣ Rename Columns")
    
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        replicates = st.number_input(
            "Replicates per condition", 
            1, 10, 
            value=st.session_state.shared_replicates,
            key=f"{key_prefix}_replicates"
        )
        
        # Update shared config
        if replicates != st.session_state.shared_replicates:
            st.session_state.shared_replicates = replicates
            st.caption("✅ Saved for other tab")
    
    with col_b:
        autorename = st.checkbox(
            "Auto-rename (A1, A2, B1...)", 
            value=st.session_state.shared_autorename, 
            key=f"{key_prefix}_autorename"
        )
        
        # Update shared config
        if autorename != st.session_state.shared_autorename:
            st.session_state.shared_autorename = autorename
    
    if autorename:
        new_names = generate_column_names(len(selected), replicates)
        rename_map = dict(zip(selected, new_names))
        df = df.rename(rename_map)
        selected = new_names
        st.info(f"✅ Renamed: {', '.join(new_names[:6])}...")
    
    st.markdown("---")
    
    # ============================================================================
    # IDENTIFY METADATA
    # ============================================================================
    
    st.subheader("5️⃣ Metadata")
    
    non_numeric = [c for c in df.columns if c not in selected]
    
    if not non_numeric:
        st.warning("⚠️ No metadata columns found")
        return False
    
    # PROTEIN ID
    col1, col2 = st.columns(2)
    
    with col1:
        id_col = st.selectbox(
            "🔍 Protein ID", 
            non_numeric, 
            index=0,
            key=f"{key_prefix}_id"
        )
    
    # SPECIES INFO COLUMN
    with col2:
        species_col = st.selectbox(
            "🧬 Species Info Column", 
            non_numeric,
            help="Column containing species information",
            key=f"{key_prefix}_species"
        )
    
    # PEPTIDE SEQUENCE (only for peptide data)
    sequence_col = None
    if data_type == 'peptide':
        st.markdown("**Peptide-Specific:**")
        sequence_col = st.selectbox(
            "🔬 Peptide Sequence",
            non_numeric,
            help="Column containing peptide sequences",
            key=f"{key_prefix}_sequence"
        )
    
    # ============================================================================
    # SPECIES TAGGING SYSTEM - USE SHARED CONFIG
    # ============================================================================
    
    st.markdown("---")
    st.markdown("**🏷️ Species Tagging**")
    
    st.info("""
    Define species tags to identify in the data. Use **OTHERS** to group all remaining entries (e.g., contaminants).
    
    💡 **Configuration is shared between protein and peptide tabs**
    """)
    
    # Show sample values
    sample_values = df[species_col].unique().to_list()[:10]
    with st.expander(f"📋 Sample values from '{species_col}' column"):
        st.write(", ".join([f"`{v}`" for v in sample_values]))
    
    # Species tag input - USE SHARED
    species_tags_input = st.text_input(
        "Species tags (comma-separated):",
        value=st.session_state.shared_species_tags,
        help="Enter keywords to identify species. Add 'OTHERS' to group remaining entries. Shared between tabs.",
        key=f"{key_prefix}_tags"
    )
    
    # Update shared config
    if species_tags_input != st.session_state.shared_species_tags:
        st.session_state.shared_species_tags = species_tags_input
        st.caption("✅ Saved for other tab")
    
    # Parse tags
    species_tags = [tag.strip().upper() for tag in species_tags_input.split(',') if tag.strip()]
    
    if not species_tags:
        st.warning("⚠️ Please enter at least one species tag")
        return False
    
    # Check if OTHERS is included
    has_others = 'OTHERS' in species_tags
    
    if has_others:
        st.info("✅ **OTHERS** tag detected - all unmatched entries will be grouped as contaminants")
        species_tags.remove('OTHERS')  # Remove for processing
    
    # ============================================================================
    # APPLY SPECIES TAGGING
    # ============================================================================
    
    # Keep only needed columns
    keep_cols = [id_col, species_col] + selected
    if sequence_col:
        keep_cols.append(sequence_col)
    
    df = df.select(keep_cols)
    
    # Apply tagging
    df = df.with_columns(
        pl.col(species_col).map_elements(
            lambda x: tag_species(x, species_tags, has_others), 
            return_dtype=pl.Utf8
        ).alias('species')
    )
    
    species_col_final = 'species'
    
    # ============================================================================
    # VALIDATION & PREVIEW
    # ============================================================================
    
    # Count species distribution
    species_counts = df.group_by('species').agg(
        pl.len().alias('count')
    ).sort('count', descending=True)
    
    st.success(f"✅ Tagged {df.shape[0]:,} rows into {species_counts.shape[0]} groups:")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Show distribution
        species_counts_with_pct = species_counts.with_columns(
            (pl.col('count') / df.shape[0] * 100).round(1).alias('percentage')
        )
        
        st.dataframe(
            species_counts_with_pct.to_pandas().style.format({'percentage': '{:.1f}%'}),
            hide_index=True,
            width='stretch'
        )
    
    with col2:
        # Validation metrics
        n_unknown = df.filter(pl.col('species') == 'UNKNOWN').shape[0]
        n_others = df.filter(pl.col('species') == 'OTHERS').shape[0]
        
        if n_unknown > 0:
            st.warning(f"⚠️ {n_unknown} rows → UNKNOWN")
            st.caption("Add more tags or use OTHERS")
        
        if n_others > 0:
            st.info(f"ℹ️ {n_others} rows → OTHERS")
            st.caption("Contaminants grouped")
        
        if n_unknown == 0 and (has_others or n_others == 0):
            st.success("✅ All rows tagged")
    
    # Show examples for each species
    with st.expander("🔍 Sample entries per species"):
        for species_name in species_counts['species'].to_list():
            sample_entries = df.filter(pl.col('species') == species_name).select([id_col, species_col]).head(3)
            st.markdown(f"**{species_name}:**")
            st.dataframe(sample_entries.to_pandas(), hide_index=True, width='stretch')
    
    # Clean up
    del species_counts, species_counts_with_pct
    gc.collect()
    
    species_col = species_col_final
    
    st.markdown("---")
    
    # ============================================================================
    # PREVIEW
    # ============================================================================
    
    st.subheader("6️⃣ Preview")
    st.dataframe(df.head(10), width='stretch', height=350)
    st.markdown("---")
    
    # ============================================================================
    # STATS
    # ============================================================================
    
    st.subheader("7️⃣ Statistics")
    
    n_rows = df.shape[0]
    n_samples = len(selected)
    n_conditions = n_samples // replicates
    species_count = df[species_col].n_unique()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{data_type.title()}s", f"{n_rows:,}")
    c2.metric("Samples", n_samples)
    c3.metric("Conditions", n_conditions)
    c4.metric("Species", species_count)
    
    if data_type == 'peptide' and sequence_col:
        n_unique_peptides = df[sequence_col].n_unique()
        st.metric("Unique Sequences", f"{n_unique_peptides:,}")
    
    st.markdown("---")
    
    # ============================================================================
    # CACHE TO SESSION STATE
    # ============================================================================
    
    st.subheader("8️⃣ Confirm")
    
    st.info(f"""
    **Summary:**
    - File: `{uploaded_file.name}`
    - Rows: {n_rows:,}
    - Samples: {n_samples}
    - Conditions: {n_conditions}
    - Species: {species_count}
    - Missing: {missing_pct:.1f}%
    """)
    
    if st.button(f"✅ Cache {data_type.title()} Data", type="primary", width='stretch', key=f"{key_prefix}_cache"):
        # Store dataset
        st.session_state[f'df_{data_type}'] = df
        st.session_state[f'{data_type}_cols'] = selected
        st.session_state[f'{data_type}_id_col'] = id_col
        st.session_state[f'{data_type}_species_col'] = species_col
        st.session_state[f'{data_type}_replicates'] = replicates
        
        # Store shared config
        st.session_state[f'{data_type}_shared_config'] = {
            'replicates': replicates,
            'species_tags': species_tags_input,
            'autorename': autorename
        }
        
        if sequence_col:
            st.session_state[f'{data_type}_sequence_col'] = sequence_col
        
        # FREE MEMORY
        del edited
        clear_temp_session_data()
        gc.collect()
        
        st.success(f"🎉 {data_type.title()} data cached!")
        st.info("💡 Configuration saved for other tab")
        return True
    
    return False

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(page_title="Data Upload", layout="wide")
st.title("📊 Data Upload")

# Initialize shared configuration
initialize_shared_config()

# Clear any leftover temp data
clear_temp_session_data()

# ============================================================================
# SHOW SHARED CONFIG STATUS
# ============================================================================

st.info("**Upload protein-level data, peptide-level data, or both**")

with st.expander("⚙️ Shared Configuration", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Replicates", st.session_state.shared_replicates)
    
    with col2:
        st.metric("Auto-rename", "✅ Yes" if st.session_state.shared_autorename else "❌ No")
    
    with col3:
        tags_preview = st.session_state.shared_species_tags[:30] + "..." if len(st.session_state.shared_species_tags) > 30 else st.session_state.shared_species_tags
        st.caption(f"**Species Tags:**")
        st.caption(tags_preview)
    
    st.caption("*These settings are shared between protein and peptide tabs*")

st.markdown("---")

# ============================================================================
# TABS
# ============================================================================

tab_protein, tab_peptide = st.tabs(["🧬 Protein Data", "🔬 Peptide Data"])

with tab_protein:
    protein_file = st.file_uploader(
        "Upload protein matrix:",
        type=['csv', 'tsv', 'txt', 'xlsx'],
        key='protein_upload'
    )
    
    if protein_file:
        process_dataset(protein_file, 'protein', 'prot')

with tab_peptide:
    peptide_file = st.file_uploader(
        "Upload peptide matrix:",
        type=['csv', 'tsv', 'txt', 'xlsx'],
        key='peptide_upload'
    )
    
    if peptide_file:
        process_dataset(peptide_file, 'peptide', 'pep')

# ============================================================================
# VALIDATION & CONTINUE
# ============================================================================

st.markdown("---")
st.markdown("---")

st.header("✅ Ready to Continue")

has_protein = 'df_protein' in st.session_state
has_peptide = 'df_peptide' in st.session_state

if not has_protein and not has_peptide:
    st.warning("⚠️ Upload and cache at least one dataset to continue")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    if has_protein:
        st.success(f"✅ **Protein data:** {st.session_state.df_protein.shape[0]:,} proteins")
    else:
        st.info("ℹ️ No protein data uploaded")

with col2:
    if has_peptide:
        st.success(f"✅ **Peptide data:** {st.session_state.df_peptide.shape[0]:,} peptides")
    else:
        st.info("ℹ️ No peptide data uploaded")

if st.button("🎯 Continue to Analysis", type="primary", width='stretch'):
    st.session_state.data_type = 'both' if (has_protein and has_peptide) else ('protein' if has_protein else 'peptide')
    time.sleep(0.5)
    st.switch_page("pages/2_Visual_EDA.py")
