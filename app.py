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
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "processed_df" not in st.session_state:
    st.session_state.processed_df = None
if "column_names" not in st.session_state:
    st.session_state.column_names = {}

st.title("🧬 Proteomics Data Upload & Configuration")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload proteomics CSV", type=["csv"])

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
        
        # 1. Protein Group Column Selector (only one allowed)
        with col1:
            st.subheader("Protein Group Column")
            non_numeric_cols = [c for c in renamed_df.columns if not pd.api.types.is_numeric_dtype(renamed_df[c])]
            protein_group_col = st.selectbox(
                "Select column containing Protein Group IDs",
                options=["None"] + non_numeric_cols,
                index=1 if non_numeric_cols else 0,
                help="Only one column can be designated as Protein Group"
            )
            protein_group_col = None if protein_group_col == "None" else protein_group_col
        
        # 2. Species Tags
        with col2:
            st.subheader("Species Filter Tags")
            available_species = ["_HUMAN", "_YEAST", "_ECOLI", "_MOUSE"]
            species_tags = st.multiselect(
                "Select species to include (matches against Protein Names)",
                options=available_species,
                default=["_HUMAN"],
                help="Filter rows by species identifier in protein names"
            )
            # Custom species input
            custom_species = st.text_input("Add custom species tag (e.g., _RAT)")
            if custom_species and custom_species not in species_tags:
                species_tags.append(custom_species)
    
    # 3. Column Name Editor
    with st.expander("✏️ Edit Column Names", expanded=False):
        st.caption("Edit numeric column names (condition_replicate format)")
        numeric_cols = [c for c in renamed_df.columns if pd.api.types.is_numeric_dtype(renamed_df[c])]
        
        edited_names = {}
        cols_per_row = 3
        for i in range(0, len(numeric_cols), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j, col in enumerate(numeric_cols[i:i+cols_per_row]):
                with row_cols[j]:
                    edited_names[col] = st.text_input(
                        f"Original: `{col}`",
                        value=st.session_state.column_names.get(col, col),
                        key=f"edit_{col}"
                    )
        
        if st.button("Apply Column Renames"):
            st.session_state.column_names.update(edited_names)
            st.rerun()
    
    # --- Apply Configurations ---
    # Apply user-edited column names
    final_rename = {k: v for k, v in st.session_state.column_names.items() if k != v}
    processed_df = renamed_df.rename(columns=final_rename)
    
    # Determine filter column (use Protein Names if available for species filtering)
    filter_col = None
    for potential in ["PG.ProteinNames", "ProteinNames", "Protein.Names"]:
        if potential in processed_df.columns:
            filter_col = potential
            break
    
    # Apply species filter
    if species_tags and filter_col:
        processed_df = filter_by_species(processed_df, filter_col, species_tags)
        st.info(f"Filtered to {len(processed_df)} rows matching species: {', '.join(species_tags)}")
    
    # Cache the processed dataframe
    st.session_state.processed_df = processed_df
    
    # --- Preview ---
    st.subheader("📊 Data Preview")
    
    # Show column summary
    numeric_final = [c for c in processed_df.columns if pd.api.types.is_numeric_dtype(processed_df[c])]
    meta_cols = [c for c in processed_df.columns if c not in numeric_final]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows", len(processed_df))
    c2.metric("Numeric Columns", len(numeric_final))
    c3.metric("Metadata Columns", len(meta_cols))
    
    if protein_group_col:
        st.caption(f"**Protein Group Column:** `{protein_group_col}`")
    
    # Data editor for final review
    st.dataframe(
        processed_df.head(100),
        use_container_width=True,
        height=400,
        column_config={
            protein_group_col: st.column_config.TextColumn("Protein Groups", width="medium")
        } if protein_group_col else None
    )
    
    # --- Export Cached Data ---
    st.divider()
    st.subheader("💾 Cached Data for Downstream")
    
    @st.cache_data
    def get_numeric_matrix(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
        return df[numeric_cols].copy()
    
    @st.cache_data  
    def get_metadata(df: pd.DataFrame, meta_cols: list) -> pd.DataFrame:
        return df[meta_cols].copy()
    
    # Store separate dataframes
    numeric_matrix = get_numeric_matrix(processed_df, numeric_final)
    metadata_df = get_metadata(processed_df, meta_cols)
    
    tab1, tab2 = st.tabs(["Numeric Matrix", "Metadata"])
    with tab1:
        st.dataframe(numeric_matrix.head(50), use_container_width=True)
        st.download_button("Download Numeric Matrix", numeric_matrix.to_csv(index=False), "numeric_matrix.csv")
    with tab2:
        st.dataframe(metadata_df.head(50), use_container_width=True)
        st.download_button("Download Metadata", metadata_df.to_csv(index=False), "metadata.csv")

else:
    st.info("👆 Upload a CSV file to begin")
