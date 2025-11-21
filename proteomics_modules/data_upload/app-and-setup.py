# ==================================================================================
# FINAL MODULE FILES - COMPLETE CODE
# Copy each section to create separate .py files
# ==================================================================================

# ----------------------------------------------------------------------------------
# FILE 9: app.py
# ----------------------------------------------------------------------------------
"""
Demo Streamlit Application
Proteomics Data Upload Module
"""

import streamlit as st
from proteomics_modules.data_upload import run_upload_module

# Page configuration
st.set_page_config(
    page_title="Proteomics Data Upload",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .reportview-container {
        background: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🧬 Proteomics Analysis")
    st.markdown("---")
    
    st.markdown("""
    ### Module 1: Data Upload
    
    **Features:**
    - Multi-format file support
    - Auto-detect columns
    - Species identification
    - Sample annotation
    - Workflow recommendation
    
    **Supported Formats:**
    - DIA-NN output
    - Spectronaut
    - MaxQuant
    - Custom CSV/TSV
    """)
    
    st.markdown("---")
    
    st.info("""
    **Next Modules (Coming Soon):**
    - Quality Control
    - Preprocessing
    - Statistical Analysis
    - Visualization
    """)

# Main content
run_upload_module()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Proteomics Analysis Platform v0.1.0</p>
    <p>Built with Streamlit • Modular Architecture</p>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------------
# FILE 10: setup.py
# ----------------------------------------------------------------------------------
"""
Setup script for proteomics_modules package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="proteomics_modules",
    version="0.1.0",
    author="Proteomics Team",
    description="Modular proteomics data analysis toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "streamlit>=1.20.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "plotly>=5.11.0",
        "scipy>=1.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
)
