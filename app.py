# app.py
import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="Proteomics Data Upload", layout="wide")

# --- Helper Functions ---
def extract_condition_replicate(col_name: str) -> str:
    """Extract condition and replicate from column name like '..._Y05-E45_01.raw...'"""
    match = re.search(r'([A-Z]\d{2}-[A-Z]\d{2})_(\d{2})', col_name)
    if match:
        condition, rep = match.groups()
        return f"{condition}_{int(rep)}"
    return col_name

def detect_and_rename_numeric_cols(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Auto-detect numeric columns and rename them."""
    rename_map = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            new_name = extract_condition_replicate(col)
            rename_map[col] = new_name
    return df.rename(columns=rename_map), rename_map

def filter_by_species(df: pd.DataFrame, col: str, species_tags: list[str]) -> pd.DataFrame:
    """Filter rows where any species tag appears in the specified column."""
    if not species_tags or not col:
        return df
    pattern = '|'.join(re.escape(tag) for tag in species_tags)
    return df[df[col].astype(str).str.contains(pattern, case=False, na=False)]

# --- Session State Initialization ---
defaults = {
    "uploaded_df": None,
    "processed_df": None,
    "column_names": {},
    "protein_data": None,
    "peptide_data": None,
    "upload_key": 0,  # For clearing file uploader
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.title("🧬 Proteomics Data Upload & Configuration")

# --- Show cached data status ---
col_status1, col_status2 = st.columns(2)
with col_status1:
    if st.session_state.protein_data is not None:
        st.success(f"✅ Protein data cached: {len(st.session_state.protein_data)} rows")
with col_status2:
    if st.session_state.peptide_data is not None:
        st.success(f"✅ Peptide data cached: {len(st.session_state.peptide_data)} rows")

# --- File Upload ---
uploaded_file = st.file_uploader(
    "Upload proteomics CSV", 
    type=["csv"], 
    key=f"uploader_{st.session_state.upload_key}"
)

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    renamed_df, rename_map = detect_and_rename_numeric_cols(raw_df)
    
    # Init column names in session state if new upload
    if st.session_state.uploaded_df is None or not renamed_df.columns.equals(st.session_state.uploaded_df.columns):
        st.session_state.uploaded_df = renamed_df.copy()
        st.session_state.column_names = {c: c for c in renamed_df.columns}
    
    st.success(f"Loaded {len(renamed_df)} rows, {len(renamed_df.columns)} columns")
    
    # --- Configuration Panel ---
    with st.expander("⚙️ Column Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        non_numeric_cols = [c for c in renamed_df.columns if not pd.api.types.is_numeric_dtype(renamed_df[c])]
        
        # 1. Protein Group Column Selector
        with col1:
            st.subheader("Protein Group Column")
            protein_group_col = st.selectbox(
                "Select column containing Protein Group IDs",
                options=["None"] + non_numeric_cols,
                index=1 if non_numeric_cols else 0,
                help="Only one column can be designated as Protein Group"
            )
            protein_group_col = None if protein_group_col == "None" else protein_group_col
        
        # 2. Sequence Column Selector (Peptide vs Protein level)
        with col2:
            st.subheader("Sequence Column (Peptide Data)")
            sequence_col = st.selectbox(
                "Select column containing peptide sequences",
                options=["None"] + non_numeric_cols,
                index=0,  # Default None = protein level
                help="If None → Protein-level data. If selected → Peptide-level data"
            )
            sequence_col = None if sequence_col == "None" else sequence_col
            
            # Display data level indicator
            is_peptide_data = sequence_col is not None
            if is_peptide_data:
                st.warning("📋 **Peptide-level data**")
            else:
                st.info("🧬 **Protein-level data**")
        
        # 3. Species Tags
        st.subheader("Species Filter Tags")
        col_sp1, col_sp2 = st.columns([2, 1])
        with col_sp1:
            available_species = ["_HUMAN", "_YEAST", "_ECOLI", "_MOUSE"]
            species_tags = st.multiselect(
                "Select species to include",
                options=available_species,
                default=["_HUMAN"],
            )
        with col_sp2:
            custom_species = st.text_input("Custom tag")
            if custom_species and custom_species not in species_tags:
                species_tags.append(custom_species)
    
    # 4. Column Name Editor
    with st.expander("✏️ Edit Column Names", expanded=False):
        numeric_cols = [c for c in renamed_df.columns if pd.api.types.is_numeric_dtype(renamed_df[c])]
        
        edited_names = {}
        cols_per_row = 3
        for i in range(0, len(numeric_cols), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j, col in enumerate(numeric_cols[i:i+cols_per_row]):
                with row_cols[j]:
                    edited_names[col] = st.text_input(
                        f"`{col}`",
                        value=st.session_state.column_names.get(col, col),
                        key=f"edit_{col}"
                    )
        
        if st.button("Apply Column Renames"):
            st.session_state.column_names.update(edited_names)
            st.rerun()
    
    # --- Apply Configurations ---
    final_rename = {k: v for k, v in st.session_state.column_names.items() if k != v}
    processed_df = renamed_df.rename(columns=final_rename)
    
    # Determine filter column
    filter_col = None
    for potential in ["PG.ProteinNames", "ProteinNames", "Protein.Names"]:
        if potential in processed_df.columns:
            filter_col = potential
            break
    
    # Apply species filter
    if species_tags and filter_col:
        processed_df = filter_by_species(processed_df, filter_col, species_tags)
        st.info(f"Filtered to {len(processed_df)} rows matching: {', '.join(species_tags)}")
    
    st.session_state.processed_df = processed_df
    
    # --- Preview ---
    st.subheader("📊 Data Preview")
    
    numeric_final = [c for c in processed_df.columns if pd.api.types.is_numeric_dtype(processed_df[c])]
    meta_cols = [c for c in processed_df.columns if c not in numeric_final]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", len(processed_df))
    c2.metric("Numeric Columns", len(numeric_final))
    c3.metric("Metadata Columns", len(meta_cols))
    c4.metric("Data Level", "Peptide" if is_peptide_data else "Protein")
    
    st.dataframe(processed_df.head(100), use_container_width=True, height=300)
    
    # --- Confirm & Cache ---
    st.divider()
    data_type = "peptide" if is_peptide_data else "protein"
    
    # Check if this data type is already cached
    existing_key = f"{data_type}_data"
    if st.session_state[existing_key] is not None:
        st.warning(f"⚠️ {data_type.capitalize()} data already cached. Confirming will overwrite.")
    
    st.subheader(f"💾 Cache as {data_type.capitalize()} Data?")
    
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button(f"✅ Confirm & Cache", type="primary"):
            st.session_state[existing_key] = processed_df.copy()
            # Reset for next upload
            st.session_state.uploaded_df = None
            st.session_state.processed_df = None
            st.session_state.column_names = {}
            st.session_state.upload_key += 1  # Clear file uploader
            st.success(f"{data_type.capitalize()} data cached!")
            st.rerun()
    
    with col_btn2:
        if st.button("❌ Cancel"):
            st.session_state.uploaded_df = None
            st.session_state.processed_df = None
            st.session_state.column_names = {}
            st.session_state.upload_key += 1
            st.rerun()

else:
    st.info("👆 Upload a CSV file to begin")

# --- Display Cached Data (below uploader) ---
if st.session_state.protein_data is not None or st.session_state.peptide_data is not None:
    st.divider()
    st.subheader("📦 Cached Datasets")
    
    tab1, tab2 = st.tabs(["Protein Data", "Peptide Data"])
    
    with tab1:
        if st.session_state.protein_data is not None:
            st.dataframe(st.session_state.protein_data.head(50), use_container_width=True)
            if st.button("🗑️ Clear Protein Data"):
                st.session_state.protein_data = None
                st.rerun()
        else:
            st.caption("No protein data uploaded yet")
    
    with tab2:
        if st.session_state.peptide_data is not None:
            st.dataframe(st.session_state.peptide_data.head(50), use_container_width=True)
            if st.button("🗑️ Clear Peptide Data"):
                st.session_state.peptide_data = None
                st.rerun()
        else:
            st.caption("No peptide data uploaded yet")
