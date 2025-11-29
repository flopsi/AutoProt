import streamlit as st
import pandas as pd
import re

# --- Thermo Fisher Brand Colors ---
COLORS = {
    "red": "#E71316",
    "dark_red": "#A6192E",
    "gray": "#54585A",
    "light_gray": "#E2E3E4",
    "white": "#FFFFFF",
    "navy": "#262262",
    "green": "#B5BD00",
    "orange": "#EA7600",
}

st.set_page_config(
    page_title="Proteomics data upload | Thermo Fisher Scientific",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (Thermo Fisher Style Guide) ---
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


# --- Helper Functions ---
def extract_condition_replicate(col_name: str) -> str:
    match = re.search(r'([A-Z]\d{2}-[A-Z]\d{2})_(\d{2})', col_name)
    if match:
        condition, rep = match.groups()
        return f"{condition}_{int(rep)}"
    return col_name


def detect_and_rename_numeric_cols(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rename_map = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            rename_map[col] = extract_condition_replicate(col)
    return df.rename(columns=rename_map), rename_map


def filter_by_species(df: pd.DataFrame, col: str, species_tags: list[str]) -> pd.DataFrame:
    if not species_tags or not col:
        return df
    pattern = '|'.join(re.escape(tag) for tag in species_tags)
    return df[df[col].astype(str).str.contains(pattern, case=False, na=False)]


def render_header(title: str, subtitle: str):
    st.markdown(f"""
    <div class="header-banner">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


# --- Session State Initialization ---
defaults = {
    "uploaded_df": None,
    "processed_df": None,
    "column_names": {},
    "protein_data": None,
    "peptide_data": None,
    "protein_index_col": None,
    "peptide_index_col": None,
    "upload_key": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- Header ---
render_header("Proteomics data upload", "Import and configure mass spectrometry output matrices")

# --- Sidebar: Cached Data Status ---
with st.sidebar:
    st.markdown(f"""
    <div style='background-color: {COLORS["gray"]}; padding: 15px; margin: -60px -15px 20px -15px;'>
        <h3 style='color: white; margin: 0; text-align: center;'>Data status</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.protein_data is not None:
        n_rows = len(st.session_state.protein_data)
        st.success(f"✓ Protein data: {n_rows:,} rows")
    else:
        st.caption("No protein data cached")
    
    if st.session_state.peptide_data is not None:
        n_rows = len(st.session_state.peptide_data)
