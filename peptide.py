# pages/1_Peptide_Data_Import.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from pandas.api.types import is_numeric_dtype

st.set_page_config(page_title="Peptide Import | Thermo Fisher", layout="wide")

# Same Thermo Fisher styling
st.markdown("""
<style>
    :root {--primary-red: #E71316; --dark-red: #A6192E; --gray: #54585A; --light-gray: #E2E3E4;}
    html, body, [class*="css"] {font-family: Arial, sans-serif !important;}
    .header {background: linear-gradient(90deg, #E71316 0%, #A6192E 100%); padding: 20px 40px; color: white; margin: -60px -60px 40px -60px;}
    .header h1 {margin:0; font-size:28px; font-weight:600;}
    .header p {margin:5px 0 0 0; font-size:14px; opacity:0.95;}
    .nav {background:white; border-bottom:2px solid #E2E3E4; padding:0 40px; display:flex; gap:5px; margin:-20px -60px 40px -60px;}
    .nav-item {padding:15px 25px; font-size:14px; font-weight:500; color:#54585A; border-bottom:3px solid transparent; cursor:pointer;}
    .nav-item.active {border-bottom:3px solid #E71316; color:#E71316;}
    .module-header {background:linear-gradient(90deg,#E71316 0%,#A6192E 100%); padding:30px; border-radius:8px; margin-bottom:40px; color:white; display:flex; align-items:center; gap:20px;}
    .module-icon {width:60px; height:60px; background:rgba(255,255,255,0.2); border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:32px;}
    .card {background:white; border:1px solid #E2E3E4; border-radius:8px; padding:25px; box-shadow:0 2px 4px rgba(0,0,0,0.05); margin-bottom:25px;}
    .card:hover {box-shadow:0 4px 12px rgba(0,0,0,0.1); transform:translateY(-2px);}
    .upload-area {border:2px dashed #E2E3E4; border-radius:8px; padding:60px 30px; text-align:center; background:#fafafa; cursor:pointer;}
    .upload-area:hover {border-color:#E71316; background:rgba(231,19,22,0.02);}
    .stButton>button {background:#E71316 !important; color:white !important; border:none !important; padding:12px 24px !important; border-radius:6px !important; font-weight:500 !important;}
    .stButton>button:hover {background:#A6192E !important;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header"><h1>LFQbench — Peptide Data Import</h1><p>Upload peptide-level quantification matrix</p></div>
<div class="nav">
    <div class="nav-item active">1. Peptide Data Import</div>
    <div class="nav-item">2. Data Quality</div>
    <div class="nav-item">3. Preprocessing</div>
    <div class="nav-item">4. Analysis</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="module-header">
    <div class="module-icon">Peptide</div>
    <div><h2>Module 1: Peptide Data Import</h2><p>Upload peptide quantification (e.g., MaxQuant evidence.txt, FragPipe combined_peptide.tsv)</p></div>
</div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card"><div class="upload-area"><div style="font-size:64px; opacity:0.5; margin-bottom:20px;">Peptide Upload</div><div style="font-size:16px; color:#54585A; margin-bottom:10px;"><strong>Drag and drop your peptide file</strong></div><div style="font-size:13px; color:#54585A; opacity:0.7;">Supports .csv, .tsv, .txt • evidence.txt, combined_peptide.tsv, etc.</div></div></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["csv","tsv","txt"], label_visibility="collapsed")

if not uploaded_file:
    st.stop()

@st.cache_data
def load_peptide_file(file):
    content = file.getvalue().decode("utf-8", errors="replace")
    if content.startswith("\ufeff"): content = content[1:]
    df = pd.read_csv(io.StringIO(content), sep=None, engine="python", dtype=str)
    
    # Convert intensity columns to float
    intensity_cols = [c for c in df.columns if any(x in c.lower() for x in ["intensity", "lfq", "raw", "area"])]
    for col in intensity_cols:
        df[col] = pd.to_numeric(df[col].str.replace(",", ""), errors="coerce")
    return df

df = load_peptide_file(uploaded_file)
st.session_state.df_peptide = df
st.success(f"Peptide data imported — {len(df):,} peptides")
st.dataframe(df.head(8), use_container_width=True)

# ── NEW: Checkbox for sequence information ──
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Sequence Information")
    
    has_sequence = st.checkbox(
        "This file contains peptide sequence information",
        value=True,
        help="Check if your file has a column with actual peptide sequences (e.g., 'Sequence', 'Peptide Sequence', 'Modified sequence')"
    )
    
    if has_sequence:
        seq_candidates = [c for c in df.columns if any(k in c.lower() for k in ["sequence", "peptide"])]
        default_seq = seq_candidates[0] if seq_candidates else df.columns[0]
        sequence_col = st.selectbox("Select peptide sequence column", df.columns, index=df.columns.get_loc(default_seq) if default_seq in df.columns else 0)
        st.session_state.sequence_col = sequence_col
        st.success(f"Sequence column set to: `{sequence_col}`")
    else:
        st.session_state.sequence_col = None
        st.info("No sequence column will be used")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ── Unified assignment table (same logic as protein page) ──
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Column Assignment & Renaming")

    numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]

    # Auto-detect condition 1
    ratio_groups = {}
    for col in numeric_cols:
        m = re.search(r'_Y(\d{2})-E(\d{2})_', col)
        if m:
            key = f"Y{m.group(1)}-E{m.group(2)}"
            ratio_groups.setdefault(key, []).append(col)

    default_cond1 = ratio_groups.get(
        sorted(ratio_groups.keys(), key=lambda x: int(x.split('-')[0][1:]))[0],
        numeric_cols[:len(numeric_cols)//2]
    ) if ratio_groups else numeric_cols[:len(numeric_cols)//2]

    # Auto-detect protein/peptide ID column
    id_col_default = next((c for c in df.columns if any(k in c.lower() for k in ["leading", "protein", "peptide id"])), df.columns[0])

    rows = []
    for col in df.columns:
        is_intensity = col in numeric_cols
        preview = " | ".join(df[col].dropna().astype(str).unique()[:3])
        rows.append({
            "Rename": col,
            "Cond 1": col in default_cond1 and is_intensity,
            "Peptide/Protein ID": col == id_col_default,
            "Original Name": col,
            "Preview": preview,
            "Type": "Intensity" if is_intensity else "Metadata"
        })

    df_edit = pd.DataFrame(rows)

    edited = st.data_editor(
        df_edit,
        column_config={
            "Rename": st.column_config.TextColumn("Rename (optional)", required=False),
            "Cond 1": st.column_config.CheckboxColumn("Condition 1", default=True),
            "Peptide/Protein ID": st.column_config.CheckboxColumn("Peptide/Protein ID", default=True),
            "Original Name": st.column_config.TextColumn("Original Name", disabled=True),
            "Preview": st.column_config.TextColumn("Preview", disabled=True),
            "Type": st.column_config.TextColumn("Type", disabled=True),
        },
        disabled=["Original Name", "Preview", "Type"],
        hide_index=True,
        use_container_width=True,
        key="peptide_table"
    )

    # Apply renaming
    rename_map = {row["Original Name"]: row["Rename"].strip() for _, row in edited.iterrows() if row["Rename"].strip() and row["Rename"].strip() != row["Original Name"]}
    if rename_map:
        df = df.rename(columns=rename_map)
        st.session_state.df_peptide = df

    cond1_cols = edited[edited["Cond 1"]]["Original Name"].tolist()
    cond2_cols = [c for c in numeric_cols if c not in cond1_cols]
    id_col = edited[edited["Peptide/Protein ID"]]["Original Name"].tolist()[0]

    st.session_state.update({
        "cond1_cols_peptide": cond1_cols,
        "cond2_cols_peptide": cond2_cols,
        "id_col_peptide": id_col,
        "df_peptide": df
    })

    st.success("Peptide data ready!")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("**Condition 1**", f"{len(cond1_cols)} reps")
        st.write(", ".join(cond1_cols))
    with c2:
        st.metric("**Condition 2**", f"{len(cond2_cols)} reps")
        st.write(", ".join(cond2_cols))

    st.info(f"**Peptide/Protein ID** → `{id_col}` | **Sequence** → {'Yes' if has_sequence else 'No'}")
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# UNIVERSAL NAVIGATION + CLEAN INVISIBLE RESTART
# ─────────────────────────────────────────────────────────────
st.markdown("---")

col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1.8])

with col1:
    if st.button("Protein Upload", use_container_width=True):
        st.Page("app.py")

with col2:
    if st.button("Peptide Upload", use_container_width=True):
        st.Page("pages/1_Peptide_Data_Import.py")

with col3:
    if st.button("Data Quality", type="primary", use_container_width=True):
        if "df" in st.session_state or "df_peptide" in st.session_state:
            st.Page("pages/2_Data_Quality.py")
        else:
            st.error("Please upload data first")
            st.stop()

with col4:
    if st.button("Restart Analysis", type="secondary", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# FIXED: Truly invisible restart button (only the red bar is visible)
st.markdown("""
<div style="position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); z-index: 9999; cursor: pointer;">
    <div style="
        background: #E71316; 
        color: white; 
        padding: 14px 36px; 
        border-radius: 10px; 
        font-weight: 600; 
        font-size: 16px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        user-select: none;
        transition: all 0.2s;
    " onclick="document.getElementById('invisible_restart').click()">
        Restart Analysis — Clear All Data
    </div>
</div>

<!-- This is the REAL hidden button — completely invisible -->
<button id="invisible_restart" style="position:fixed; bottom:0; left:0; width:0; height:0; opacity:0; pointer-events:none;" 
        onclick="this.closest('form').submit()"></button>
""", unsafe_allow_html=True)

# The actual hidden restart trigger
if st.button("real_hidden_restart", key="real_hidden_restart"):
    st.session_state.clear()
    st.rerun()
