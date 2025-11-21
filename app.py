"""
AutoProt - Automated Proteomics Analysis Platform
Main application entry point
"""

import streamlit as st

# Page config MUST be first Streamlit command
st.set_page_config(
    page_title="AutoProt Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modules
from proteomics_modules.data_upload.module import run_upload_module
from proteomics_modules.lfqbench_analysis.lfqbench_module import run_lfqbench_module

# Sidebar navigation
st.sidebar.title("📊 AutoProt Analysis")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Module",
    ["🔼 Data Upload", "🧪 LFQbench Analysis"],
    key="main_navigation"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "AutoProt provides comprehensive proteomics data analysis "
    "with benchmark evaluation capabilities."
)

# Main content - ONLY ONE module runs at a time
if page == "🔼 Data Upload":
    run_upload_module()
elif page == "🧪 LFQbench Analysis":
    run_lfqbench_module()
