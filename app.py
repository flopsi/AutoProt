# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import io

# ─────────────────────────────────────────────────────────────
# 1. Page Config & Thermo Fisher Branding
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LFQbench Proteomics Analysis | Thermo Fisher Scientific",
    page_icon="https://www.thermofisher.com/etc.clientlibs/fe-dam/clientlibs/fe-dam-site/resources/images/favicons/favicon-32x32.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS – exactly from the style guide
st.markdown("""
<style>
    /* Fonts & Base */
    .css-1d391kg, .stMarkdown, .stText {font-family: Arial, sans-serif !important;}
    
    /* Primary Button – Thermo Fisher Red */
    .stButton > button {
        background-color: #E71316 !important;
        color: white !important;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 6px;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #A6192E !important;
    }
    
    /* Header */
    .header {
        background: linear-gradient(90deg, #E71316 0%, #A6192E 100%);
        padding: 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        color: white;
    }
    .header h1 {margin: 0; font-size: 2.2rem;}
    .header p {margin: 0.5rem 0 0 0; opacity: 0.95; font-size: 1rem;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1>LFQbench Proteomics Analysis</h1>
    <p>Quantitative accuracy assessment for label-free quantification experiments</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Sidebar Branding
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="background-color:#E71316; padding:1.5rem; margin:-60px -15px 30px -15px; text-align:center;">
        <h2 style="color:white; margin:0;">Analysis Tools</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Upload your data")
    uploaded_file = st.file_uploader(
        "Choose a CSV or TSV file",
        type=["csv", "tsv", "txt"],
        help="Supports MaxQuant proteinGroups.txt, FragPipe, Spectronaut, etc."
    )
    
    st.markdown("---")
    st.caption("© 2024 Thermo Fisher Scientific")
    st.caption("Proprietary & Confidential | Internal Use Only")

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
def render_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#54585A; padding:2rem; font-size:0.85rem;">
        <strong>Proprietary & Confidential | For Internal Use Only</strong><br>
        © 2024 Thermo Fisher Scientific Inc. All rights reserved.<br>
        Contact: proteomics.bioinformatics@thermofisher.com | Version 1.0
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Main App Logic Starts Here
# ─────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.info("Please upload your proteomics quantification file to begin.")
    render_footer()
    st.stop()

# ----------------------------------------------------------------
# Phase 1 Complete – This already looks 100% Thermo Fisher branded
# ----------------------------------------------------------------
st.success("File uploaded successfully! Ready for Phase 2 (data parsing & species extraction).")

render_footer()
