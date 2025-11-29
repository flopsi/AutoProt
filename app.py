import streamlit as st
from components import inject_custom_css, render_navbar, render_footer, COLORS

st.set_page_config(
    page_title="Proteomics Analysis | Thermo Fisher Scientific",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

inject_custom_css()
render_navbar(active_page="home")

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
render_footer()
