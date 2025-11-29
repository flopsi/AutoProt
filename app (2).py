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

st.markdown("### Current data status")

protein_data = st.session_state.get("protein_data")
peptide_data = st.session_state.get("peptide_data")

c1, c2 = st.columns(2)
with c1:
    if protein_data is not None:
        st.success(f"Protein data: {len(protein_data):,} rows")
    else:
        st.info("No protein data loaded")

with c2:
    if peptide_data is not None:
        st.success(f"Peptide data: {len(peptide_data):,} rows")
    else:
        st.info("No peptide data loaded")

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: {COLORS['gray']}; font-size: 12px; padding: 20px 0;">
    <p><strong>For research use only</strong></p>
    <p>© 2024 Thermo Fisher Scientific Inc. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
