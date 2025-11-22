import streamlit as st
from proteomics_modules.data_upload.module import run_upload_module
from proteomics_modules.lfqbench_analysis.lfqbench_module import run_lfqbench_module

# Page config
st.set_page_config(page_title="AutoProt Analysis", page_icon="🧬", layout="wide")

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Data Upload"

# Sidebar navigation
st.sidebar.title("📊 AutoProt Analysis")

# Get current page from session state (this allows button navigation to work)
current_page = st.session_state.current_page

# Sidebar radio - update based on current page
page = st.sidebar.radio(
    "Navigate to:",
    ["Data Upload", "LFQbench Analysis"],
    index=0 if current_page == "Data Upload" else 1,
    key="sidebar_nav"
)

# Update current_page if sidebar changed
if page != st.session_state.current_page:
    st.session_state.current_page = page
    st.rerun()

# Render the selected page
if st.session_state.current_page == "Data Upload":
    run_upload_module()
elif st.session_state.current_page == "LFQbench Analysis":
    run_lfqbench_module()
