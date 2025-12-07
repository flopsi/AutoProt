"""
pages/1_Data_Upload.py
Upload protein and/or peptide data with tabs
"""

import streamlit as st
import polars as pl
import time

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

def infer_species(protein_id: str) -> str:
    """Extract species from protein ID (e.g., 'P12345_HUMAN' -> 'HUMAN')."""
    if not protein_id or '_' not in protein_id:
        return 'UNKNOWN'
    return protein_id.split('_')[-1].upper()

def process_dataset(uploaded_file, data_type: str, key_prefix: str):
    """Process uploaded dataset (protein or peptide) - returns True if successful."""
    
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
    
    numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
    
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
        use_container_width=True,
        height=400,
        key=f"{key_prefix}_col_editor"
    )
    
    selected = edited[edited['Select']]['Full Name'].tolist()
    
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
            'column
