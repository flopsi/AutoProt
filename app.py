# app.py — LFQbench Data Import (Thermo Fisher Corporate Design) — FULLY WORKING
import streamlit as st
import pandas as pd
import numpy as np
import io
import re

# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LFQbench Data Import | Thermo Fisher Scientific",
    page_icon="https://www.thermofisher.com/etc.clientlibs/fe-dam/clientlibs/fe-dam-site/resources/images/favicons/favicon-32x32.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# Thermo Fisher CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --primary-red: #E71316; --dark-red: #A6192E; --gray: #54585A;
        --light-gray: #E2E3E4;
    }
    html, body, [class*="css"] {font-family: Arial, sans-serif !important;}
    .header {
        background: linear-gradient(90deg, #E71316 0%, #A6192E 100%);
        padding: 20px 40px; color: white; margin: -60px -60px 40px -60px;
    }
    .header h1 {margin:0; font-size:28px; font-weight:600;}
    .header p {margin:5px 0 0 0; font-size:14px; opacity:0.95;}
    .nav {
        background:white; border-bottom:2px solid #E2E3E4;
        padding:0 40px; display:flex; gap:5px; margin:-20px -60px 40px -60px;
    }
    .nav-item {padding:15px 25px; font-size:14px; font-weight:500; color:#54585A;
               border-bottom:3px solid transparent; cursor:pointer;}
    .nav-item:hover {background:rgba(231,19,22,0.05);}
    .nav-item.active {border-bottom:3px solid #E71316; color:#E71316;}
    .module-header {
        background:linear-gradient(90deg,#E71316 0%,#A6192E 100%);
        padding:30px; border-radius:8px; margin-bottom:40px; color:white;
        display:flex; align-items:center; gap:20px;
    }
    .module-icon {width:60px; height:60px; background:rgba(255,255,255,0.2);
                  border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:32px;}
    .card {
        background:white; border:1px solid #E2E3E4; border-radius:8px;
        padding:25px; box-shadow:0 2px 4px rgba(0,0,0,0.05); margin-bottom:25px;
    }
    .card:hover {box-shadow:0 4px 12px rgba(0,0,0,0.1); transform:translateY(-2px);}
    .upload-area {
        border:2px dashed #E2E3E4; border-radius:8px; padding:60px 30px;
        text-align:center; background:#fafafa; cursor:pointer;
    }
    .upload-area:hover {border-color:#E71316; background:rgba(231,19,22,0.02);}
    .stButton>button {
        background:#E71316 !important; color:white !important;
        border:none !important; padding:12px 24px !important; border-radius:6px !important;
        font-weight:500 !important;
    }
    .stButton>button:hover {background:#A6192E !important;}
    .footer {text-align:center; padding:30px; color:#54585A; font-size:12px;
             border-top:1px solid #E2E3E4; margin-top:60px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Header + Nav
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1>LFQbench Proteomics Analysis</h1>
    <p>Quantitative accuracy assessment for label-free quantification experiments</p>
</div>
<div class="nav">
    <div class="nav-item active">Module 1: Data Import</div>
    <div class="nav-item">Module 2: Data Quality</div>
    <div class="nav-item">Module 3: Preprocessing</div>
    <div class="nav-item">Module 4: Analysis</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="module-header">
    <div class="module-icon">Upload</div>
    <div>
        <h2>Module 1: Data Import & Validation</h2>
        <p>Upload your MaxQuant or FragPipe LFQ intensity matrix</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────────────────────
with st.container():
    st.markdown("""
    <div class="card">
        <div class="upload-area">
            <div style="font-size:64px; opacity:0.5; margin-bottom:20px;">Upload</div>
            <div style="font-size:16px; color:#54585A; margin-bottom:10px;">
                <strong>Drag and drop your file here</strong>
            </div>
            <div style="font-size:13px; color:#54585A; opacity:0.7;">
                Supports .csv, .tsv, .txt • MaxQuant, FragPipe, Spectronaut, DIA-NN
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["csv","tsv","txt"], label_visibility="collapsed")

if not uploaded_file:
    st.markdown("""
    <div class="footer">
        <strong>Proprietary & Confidential | For Internal Use Only</strong><br>
        © 2024 Thermo Fisher Scientific Inc. All rights reserved.
    </div>
    """, unsafe_allow_html=True)
    st.stop()
# ─────────────────────────────────────────────────────────────
# Load & Parse
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_and_parse(file):
    content = file.getvalue().decode("utf-8", errors="replace")
    if content.startswith("\ufeff"): content = content[1:]
    df = pd.read_csv(io.StringIO(content), sep=None, engine="python", dtype=str)
    
    return df

df = load_and_parse(uploaded_file)
st.session_state.df = df
st.success(f"Data imported — {len(df):,} proteins")
st.dataframe(df.head(10), use_container_width=True)
# REPLACE THE ENTIRE "Metadata Columns" block with this one:
# ─────────────────────────────────────────────────────────────
# 5. Metadata Assignment – now with checkboxes in the table
# ─────────────────────────────────────────────────────────────
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Metadata Column Assignment")

    # Auto-detect candidates
    species_candidates = []
    for col in df.columns:
        if df[col].astype(str).str.upper().fillna("").str.contains("HUMAN|YEAST|ECOLI").any():
            species_candidates.append(col)

    protein_candidates = [c for c in df.columns if "protein" in c.lower()]

    # Default selections
    default_species = "Species" if "Species" in df.columns else (species_candidates[0] if species_candidates else None)
    default_protein = protein_candidates[0] if protein_candidates else df.columns[0]

    # Build preview table for metadata
    meta_preview = []
    for col in df.columns:
        meta_preview.append({
            "Species Column": col == default_species,
            "Protein Group Column": col == default_protein,
            "Column Name": col,
            "Content Preview": " | ".join(df[col].dropna().astype(str).unique()[:4])
        })

    meta_df = pd.DataFrame(meta_preview)

    edited_meta = st.data_editor(
        meta_df,
        column_config={
            "Species Column": st.column_config.CheckboxColumn(
                "Species Column",
                help="Check exactly one column that contains species (HUMAN/YEAST/ECOLI)"
            ),
            "Protein Group Column": st.column_config.CheckboxColumn(
                "Protein Group Column",
                help="Check exactly one column that contains protein identifiers"
            ),
            "Column Name": st.column_config.TextColumn("Column Name", disabled=True),
            "Content Preview": st.column_config.TextColumn("Preview", disabled=True),
        },
        disabled=["Column Name", "Content Preview"],
        hide_index=True,
        use_container_width=True,
        key="meta_editor"
    )

    # Extract final selections
    selected_species = edited_meta[edited_meta["Species Column"]]["Column Name"].tolist()
    selected_protein = edited_meta[edited_meta["Protein Group Column"]]["Column Name"].tolist()

    if len(selected_species) != 1:
        st.error("Exactly one Species column must be selected")
        st.stop()
    if len(selected_protein) != 1:
        st.error("Exactly one Protein Group column must be selected")
        st.stop()

    st.session_state.species_col = selected_species[0]
    st.session_state.protein_col = selected_protein[0]

    st.success(f"**Species** → `{selected_species[0]}` | **Protein Group** → `{selected_protein[0]}`")
    st.markdown("</div>", unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────
# Ready for next module
# ─────────────────────────────────────────────────────────────
st.success("Data import complete! Ready for **Module 2: Data Quality**")
st.markdown("""
<div class="footer">
    <strong>Proprietary & Confidential | For Internal Use Only</strong><br>
    © 2024 Thermo Fisher Scientific Inc. All rights reserved.
</div>
""", unsafe_allow_html=True)
