# app.py — LFQbench Proteomics Analysis (Thermo Fisher Scientific Corporate Design)
import streamlit as st
import pandas as pd
import numpy as np
import io

# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LFQbench Analysis | Thermo Fisher Scientific",
    page_icon="https://www.thermofisher.com/etc.clientlibs/fe-dam/clientlibs/fe-dam-site/resources/images/favicons/favicon-32x32.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# Full CSS from your index.html — perfectly adapted for Streamlit
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --primary-red: #E71316;
        --primary-gray: #54585A;
        --light-gray: #E2E3E4;
        --dark-red: #A6192E;
        --navy: #262262;
        --orange: #EA7600;
        --yellow: #F1B434;
        --green: #B5BD00;
    }

    /* Global Font */
    html, body, [class*="css"]  {
        font-family: Arial, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    /* Header - Exact Match */
    .header {
        background: linear-gradient(90deg, #E71316 0%, #A6192E 100%);
        padding: 20px 40px;
        color: white;
        margin: -60px -60px 40px -60px;
        border-radius: 0;
    }
    .header h1 { margin: 0; font-size: 28px; font-weight: 600; }
    .header p { margin: 5px 0 0 0; font-size: 14px; opacity: 0.95; }

    /* Navigation Bar */
    .nav {
        background-color: white;
        border-bottom: 2px solid #E2E3E4;
        padding: 0 40px;
        display: flex;
        gap: 5px;
        margin: -20px -60px 40px -60px;
    }
    .nav-item {
        padding: 15px 25px;
        font-size: 14px;
        font-weight: 500;
        color: #54585A;
        border-bottom: 3px solid transparent;
        cursor: pointer;
        transition: all 0.3s;
    }
    .nav-item:hover { background-color: rgba(231, 19, 22, 0.05); }
    .nav-item.active {
        border-bottom: 3px solid #E71316;
        color: #E71316;
    }

    /* Module Header */
    .module-header {
        background: linear-gradient(90deg, #E71316 0%, #A6192E 100%);
        padding: 30px;
        border-radius: 8px;
        margin-bottom: 40px;
        color: white;
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .module-icon {
        width: 60px; height: 60px;
        background-color: rgba(255,255,255,0.2);
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 32px;
    }

    /* Cards */
    .step-card, .demo-section, .features-section {
        background-color: white;
        border: 1px solid #E2E3E4;
        border-radius: 8px;
        padding: 25px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: all 0.3s;
    }
    .step-card:hover, .demo-section:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }

    .step-number {
        width: 40px; height: 40px;
        background-color: #E71316; color: white;
        border-radius: 50%; display: flex;
        align-items: center; justify-content: center;
        font-weight: 600; font-size: 18px;
        margin-bottom: 15px;
    }

    /* Upload Area */
    .upload-area {
        border: 2px dashed #E2E3E4;
        border-radius: 8px;
        padding: 60px 30px;
        text-align: center;
        background-color: #fafafa;
        cursor: pointer;
        transition: all 0.3s;
    }
    .upload-area:hover {
        border-color: #E71316;
        background-color: rgba(231, 19, 22, 0.02);
    }

    /* Buttons */
    .stButton > button {
        background-color: #E71316 !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }
    .stButton > button:hover {
        background-color: #A6192E !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 30px;
        color: #54585A;
        font-size: 12px;
        border-top: 1px solid #E2E3E4;
        margin-top: 60px;
    }
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

# Navigation
st.markdown("""
<div class="nav">
    <div class="nav-item active">Module 1: Data Import</div>
    <div class="nav-item">Module 2: Quality Control</div>
    <div class="nav-item">Module 3: Preprocessing</div>
    <div class="nav-item">Module 4: Analysis</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Module Header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="module-header">
    <div class="module-icon">Analysis</div>
    <div>
        <h2>Module 1: Data Import & Validation</h2>
        <p>Import and validate your MaxQuant, FragPipe, or custom LFQ intensity matrix</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Upload Demo Section (Exact Match)
# ─────────────────────────────────────────────────────────────
with st.container():
    st.markdown("""
    <div class="demo-section">
        <h3 style="font-size:20px; margin-bottom:20px; color:#54585A;">Upload your proteomics file</h3>
        <div class="upload-area">
            <div style="font-size:64px; opacity:0.5; margin-bottom:20px;">Upload</div>
            <div style="font-size:16px; color:#54585A; margin-bottom:10px;"><strong>Drag and drop your file here</strong></div>
            <div style="font-size:13px; color:#54585A; opacity:0.7;">
                or click to browse • Supports .csv, .tsv, .txt from MaxQuant, FragPipe, Spectronaut, DIA-NN
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "", 
        type=["csv", "tsv", "txt"],
        label_visibility="collapsed"
    )

# ─────────────────────────────────────────────────────────────
# If file uploaded → Show success + preview
# ─────────────────────────────────────────────────────────────
if uploaded_file:
    # Load and parse
    content = uploaded_file.getvalue().decode("utf-8", errors="replace")
    if content.startswith('\ufeff'):
        content = content[1:]
    df = pd.read_csv(io.StringIO(content), sep=None, engine='python', low_memory=False)

    # Convert numeric columns
    for col in df.columns[2:]:  # Skip ID/name
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Extract species
    if 'name' in df.columns:
        split = df['name'].str.split(',', n=1, expand=True)
        if split.shape[1] == 2:
            df['Accession'] = split[0]
            df['Species'] = split[1]
            df = df.drop(columns=['name'] if 'name' in df.columns else [])

    st.success("Data imported successfully! Ready for column renaming and condition assignment.")
    st.dataframe(df.head(), use_container_width=True)

    # Save to session state for next steps
    st.session_state.df = df
    st.session_state.uploaded = True

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <strong>Proprietary & Confidential | For Internal Use Only</strong><br>
    © 2024 Thermo Fisher Scientific Inc. All rights reserved.<br>
    Contact: proteomics.bioinformatics@thermofisher.com | Last updated: November 2024
</div>
""", unsafe_allow_html=True)
