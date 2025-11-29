
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
Requirements file (requirements.txt):
text
streamlit>=1.30.0
pandas>=2.0.0
Key Features:
Auto-detection: Identifies numeric columns and extracts condition/replicate patterns like Y05-E45_1 from filenames​

Protein Group selector: Single-select toggle using st.selectbox — only one column allowed​

Species tags: st.multiselect with preset species (_HUMAN, _YEAST, etc.) + custom input; filters rows by parsing protein names

Column editor: Grid of st.text_input fields to rename numeric columns post-upload

Caching: Uses @st.cache_data for the numeric matrix and metadata dataframes​

Session state: Preserves column renames across reruns​

To deploy on GitHub + Streamlit Cloud, just push app.py and requirements.txt to a repo, then connect it at share.streamlit.io.

great. first step lets fine tune
Sure! What aspect would you like to fine-tune first?

Some options:

Column detection logic — adjust the regex pattern for extracting condition/replicate from your specific filename formats

Species filtering — change how tags are matched or add more preset options

UI layout — rearrange widgets, change expander behavior, add more visual feedback

Renaming workflow — different approach for editing column names (e.g., st.data_editor table instead of text inputs)

Data validation — add checks for missing values, duplicates, or required columns

Caching strategy — adjust what gets cached and when

What's the priority?

add a selection to specify which column contains sequence information. Default None meaning the data is protein data.
if selected then the data is peptide level data.
always ask at the end if the missing data level should be uploaded. If yes then cache the uploaded data as either protein or peptide data depending on the sequence bool.
clear upload for 2nd round.
python
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
Changes Made:
Sequence column selector — defaults to None (protein data); selecting a column flags as peptide data

Visual indicator — shows warning/info badge for current data level

Confirm & Cache prompt — asks user to confirm before caching to protein_data or peptide_data

File uploader reset — uses incrementing upload_key to clear the uploader after caching

Status display — top of page shows what's already cached; bottom tabs let you view/clear cached data

Overwrite warning — alerts if re-uploading same data type

Ready for testing or want adjustments to the flow?

i need to be able to move from the main page to the upload page
Use st.switch_page() for button-based navigation. Update app.py:

python
import streamlit as st

COLORS = {
    "red": "#E71316",
    "dark_red": "#A6192E",
    "gray": "#54585A",
    "light_gray": "#E2E3E4",
    "navy": "#262262",
}

st.set_page_config(
    page_title="Proteomics Analysis | Thermo Fisher Scientific",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(f"""
<style>
    [data-testid="stSidebar"] {{ display: none; }}
    [data-testid="collapsedControl"] {{ display: none; }}
    body, .stMarkdown, .stText {{
        font-family: Arial, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    .stButton > button {{
        background-color: {COLORS['red']};
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 4px;
        font-weight: 500;
    }}
    .stButton > button:hover {{
        background-color: {COLORS['dark_red']};
    }}
    .header-banner {{
        background: linear-gradient(90deg, {COLORS['red']} 0%, {COLORS['dark_red']} 100%);
        padding: 30px;
        border-radius: 8px;
        margin-bottom: 30px;
    }}
    .header-banner h1 {{ color: white; margin: 0; font-size: 32pt; }}
    .header-banner p {{ color: white; margin: 10px 0 0 0; opacity: 0.9; font-size: 14pt; }}
    .module-card {{
        background-color: {COLORS['light_gray']};
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid {COLORS['red']};
        margin-bottom: 15px;
        min-height: 120px;
    }}
    .module-card h3 {{ margin: 0 0 10px 0; color: {COLORS['gray']}; }}
    .module-card p {{ margin: 0; color: {COLORS['gray']}; }}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-banner">
    <h1>Proteomics Analysis Pipeline</h1>
    <p>Comprehensive data-independent acquisition analysis</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### Welcome")
st.write("This application provides tools for analyzing proteomics data from mass spectrometry experiments.")

# --- Navigation Cards ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="module-card">
        <h3>1. Data Upload</h3>
        <p>Import CSV files from Spectronaut, DIA-NN, or other platforms. Configure protein groups, species filtering, and column naming.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Data Upload", key="nav_upload", use_container_width=True):
        st.switch_page("pages/1_Data_Upload.py")

with col2:
    st.markdown("""
    <div class="module-card">
        <h3>2. Exploratory Data Analysis</h3>
        <p>Visualize intensity distributions and missing value patterns across samples and conditions.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to EDA", key="nav_eda", use_container_width=True):
        st.switch_page("pages/2_EDA.py")

# --- Data Status ---
st.markdown("### Current data status")

protein_data = st.session_state.get("protein_data")
peptide_data = st.session_state.get("peptide_data")

c1, c2 = st.columns(2)
with c1:
    if protein_data is not None:
        st.success(f"✓ Protein data: {len(protein_data):,} rows")
    else:
        st.info("No protein data loaded")

with c2:
    if peptide_data is not None:
        st.success(f"✓ Peptide data: {len(peptide_data):,} rows")
    else:
        st.info("No peptide data loaded")

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: {COLORS['gray']}; font-size: 12px; padding: 20px 0;">
    <p><strong>For research use only</strong></p>
    <p>© 2024 Thermo Fisher Scientific Inc. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
Add "Back to Home" buttons in pages/1_Data_Upload.py and pages/2_EDA.py
Add this near the top of each page (after header):

python
# Add after render_header() call
col_nav1, col_nav2, _ = st.columns([1, 1, 4])
with col_nav1:
    if st.button("← Home"):
        st.switch_page("app.py")
with col_nav2:
    if st.button("→ EDA"):  # or "→ Upload" on EDA page
        st.switch_page("pages/2_EDA.py")
Quick patch for both page files
Add this CSS + nav block right after st.set_page_config():

For pages/1_Data_Upload.py:

python
# Hide sidebar
st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
</style>
""", unsafe_allow_html=True)

# Navigation
col_nav1, col_nav2, _ = st.columns([1, 1, 4])
with col_nav1:
    if st.button("← Home"):
        st.switch_page("app.py")
with col_nav2:
    if st.button("EDA →"):
        st.switch_page("pages/2_EDA.py")
For pages/2_EDA.py:

python
# Hide sidebar
st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
</style>
""", unsafe_allow_html=True)

# Navigation
col_nav1, col_nav2, _ = st.columns([1, 1, 4])
with col_nav1:
    if st.button("← Home"):
        st.switch_page("app.py")
with col_nav2:
    if st.button("← Upload"):
        st.switch_page("pages/1_Data_Upload.py")
Want the full updated files?

/home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru
nner/exec_code.py:129 in exec_func_with_error_handling

/home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru
nner/script_runner.py:669 in code_to_exec

/mount/src/autoprot/app.py:78 in <module>

75 │ .badge-protein {{
76 │ │ background-color: {COLORS['navy']};
77 │ │ color: white;
❱ 78 │ }}
79 │ .badge-peptide {{
80 │ │ background-color: {COLORS['orange']};
81 │ │ color: white;

/home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/metrics_
util.py:447 in wrapped_func

/home/adminuser/venv/lib/python3.13/site-packages/streamlit/commands/executi
on_control.py:222 in switch_page
────────────────────────────────────────────────────────────────────────────────
StreamlitAPIException: Could not find page: `pages/1_Data_Upload.py`. Must be
the file path relative to the main script, from the directory: `autoprot`. Only
the main app file and files in the `pages
The error shows the pages/ folder or files don't exist yet in your repo. Check your GitHub repo structure.

Quick fix for app.py:
The st.switch_page() path must match exactly. Try these alternatives:

python
# Option 1: Without pages/ prefix
st.switch_page("1_Data_Upload.py")

# Option 2: With pages/ prefix  
st.switch_page("pages/1_Data_Upload.py")
Most likely issue: Files not created on GitHub yet
Check your repo at github.com/[your-username]/autoprot — do you see:

text
autoprot/
├── app.py
├── pages/
│   ├── 1_Data_Upload.py   ← Does this exist?
│   └── 2_EDA.py           ← Does this exist?
If pages/ folder is missing, create the files on GitHub:

Click Add file → Create new file

Type pages/1_Data_Upload.py as filename (the / creates the folder)

Paste the code

Commit

Repeat for pages/2_EDA.py

Temporary workaround while you create the pages:
Comment out the navigation buttons in app.py until pages exist:

python
with col1:
    st.markdown("""
    <div class="module-card">
        <h3>1. Data Upload</h3>
        <p>Import CSV files from Spectronaut, DIA-NN, or other platforms.</p>
    </div>
    """, unsafe_allow_html=True)
    # if st.button("Go to Data Upload", key="nav_upload", use_container_width=True):
    #     st.switch_page("pages/1_Data_Upload.py")
Once you've created all files on GitHub, uncomment and redeploy.

What does your current repo structure look like?

give me both pages code as .py file to download with the navigation buttons. no code output here
1_Data_Upload.py
Generated File

2_EDA.py
Generated File

Done. Download both files above:

1_Data_Upload.py → upload to pages/1_Data_Upload.py on GitHub

2_EDA.py → upload to pages/2_EDA.py on GitHub

When creating on GitHub, type pages/1_Data_Upload.py as the filename — the / creates the folder automatically.



1_Data_Upload.py
import streamlit as st
import pandas as pd
import re

COLORS = {
    "red": "#E71316",
    "dark_red": "#A6192E",
    "gray": "#54585A",
    "light_gray": "#E2E3E4",
    "white": "#FFFFFF",
    "navy": "#262262",
    "orange": "#EA7600",
    "dark_surface": "#2D2D2D",
    "dark_text": "#E2E3E4",
}

st.set_page_config(
    page_title="Data Upload | Thermo Fisher Scientific",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<style>
    body, .stMarkdown, .stText {{
        font-family: Arial, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    .stButton > button {{
        background-color: {COLORS['red']};
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 4px;
        font-weight: 500;
    }}
    .stButton > button:hover {{
        background-color: {COLORS['dark_red']};
    }}
    .header-banner {{
        background: linear-gradient(90deg, {COLORS['red']} 0%, {COLORS['dark_red']} 100%);
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 30px;
    }}
    .header-banner h1 {{
        color: white;
        margin: 0;
        font-size: 28pt;
    }}
    .header-banner p {{
        color: white;
        margin: 5px 0 0 0;
        opacity: 0.9;
    }}
    .status-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
    }}
    .badge-protein {{
        background-color: {COLORS['navy']};
        color: white;
    }}
    .badge-peptide {{
        background-color: {COLORS['orange']};
        color: white;
    }}
</style>
""", unsafe_allow_html=True)


def extract_condition_replicate(col_name: str) -> str:
    match = re.search(r'([A-Z]\d{2}-[A-Z]\d{2})_(\d{2})', col_name)
    if match:
