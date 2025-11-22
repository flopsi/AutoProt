import streamlit as st
from proteomics_modules.data_upload.module import run_upload_module
from proteomics_modules.lfqbench_analysis.lfqbench_module import run_lfqbench_module

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Data Upload"

# Sidebar navigation
st.sidebar.title("📊 AutoProt Analysis")

# Navigation options
page = st.sidebar.radio(
    "Navigate to:",
    ["Data Upload", "LFQbench Analysis"],
    index=0 if st.session_state.current_page == "Data Upload" else 1,
    key="sidebar_navigation"
)

# Update current page from sidebar selection
st.session_state.current_page = page

# Render selected page
if st.session_state.current_page == "Data Upload":
    run_upload_module()
elif st.session_state.current_page == "LFQbench Analysis":
    run_lfqbench_module()
