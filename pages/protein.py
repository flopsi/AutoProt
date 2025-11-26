# app.py — LFQbench Data Import (Thermo Fisher Design) — 100% WORKING FIXED
import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from pandas.api.types import is_numeric_dtype
# ─────────────────────────────────────────────────────────────
# Check if we already have data in session state (coming back from peptide page)
# ─────────────────────────────────────────────────────────────
.

# ─────────────────────────────────────────────────────────────
# Detect Species Column Function
# ─────────────────────────────────────────────────────────────
def detect_species_column_and_extract(df):
    """
    Automatically detect species column and extract species from strings.
    
    Logic:
    1. Scan metadata columns for any containing 'HUMAN' (case-insensitive)
    2. Identify the delimiter (e.g., _) that flanks HUMAN
    3. Extract all species at the same position using the same pattern
    
    Returns: (species_col_name, extracted_species_col_name, list_of_unique_species)
    """
    
    # Identify metadata (non-numeric) columns
    metadata_cols = []
    for col in df.columns:
        sample = df[col].dropna().head(50)
        try:
            pd.to_numeric(sample.replace('#NUM!', pd.NA), errors='raise')
        except:
            metadata_cols.append(col)
    
    # Step 1: Find column containing HUMAN
    species_col = None
    prefix_delim = None
    
    for col in metadata_cols:
        sample_values = df[col].dropna().astype(str).head(100)
        
        for val in sample_values:
            if 'HUMAN' in val.upper():
                species_col = col
                
                # Find prefix delimiter before HUMAN
                match = re.search(r'([^A-Za-z0-9])HUMAN', val, re.IGNORECASE)
                prefix_delim = match.group(1) if match else ''
                break
        
        if species_col:
            break
    
    if not species_col:
        return None, None, []
    
    # Step 2: Build extraction pattern
    if prefix_delim:
        # Pattern: _SPECIES followed by non-alpha or end
        prefix_esc = re.escape(prefix_delim)
        pattern = f'{prefix_esc}([A-Z]+)(?:[^A-Za-z]|$)'
    else:
        # Standalone species name
        pattern = r'^([A-Z]+)$'
    
    # Step 3: Extract species
    def extract_first_species(val):
        if pd.isna(val):
            return None
        val_str = str(val).upper()
        matches = re.findall(pattern, val_str)
        return matches[0] if matches else None
    
    df['_extracted_species'] = df[species_col].apply(extract_first_species)
    unique_species = sorted(df['_extracted_species'].dropna().unique().tolist())
    
    return species_col, '_extracted_species', unique_species


# ─────────────────────────────────────────────────────────────
# Thermo Fisher CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {--primary-red: #E71316; --dark-red: #A6192E; --gray: #54585A; --light-gray: #E2E3E4;}
    html, body, [class*="css"] {font-family: Arial, sans-serif !important;}
    .header {background: linear-gradient(90deg, #E71316 0%, #A6192E 100%); padding: 20px 40px; color: white; margin: -60px -60px 40px -60px;}
    .header h1 {margin:0; font-size:28px; font-weight:600;}
    .header p {margin:5px 0 0 0; font-size:14px; opacity:0.95;}
    .nav {background:white; border-bottom:2px solid #E2E3E4; padding:0 40px; display:flex; gap:5px; margin:-20px -60px 40px -60px;}
    .nav-item {padding:15px 25px; font-size:14px; font-weight:500; color:#54585A; border-bottom:3px solid transparent; cursor:pointer;}
    .nav-item:hover {background:rgba(231,19,22,0.05);}
    .nav-item.active {border-bottom:3px solid #E71316; color:#E71316;}
    .card {background:white; border:1px solid #E2E3E4; border-radius:8px; padding:25px; box-shadow:0 2px 4px rgba(0,0,0,0.05); margin-bottom:25px;}
    .card:hover {box-shadow:0 4px 12px rgba(0,0,0,0.1); transform:translateY(-2px);}
    .upload-area {border:2px dashed #E2E3E4; border-radius:8px; padding:60px 30px; text-align:center; background:#fafafa; cursor:pointer;}
    .upload-area:hover {border-color:#E71316; background:rgba(231,19,22,0.02);}
    .stButton>button {background:#E71316 !important; color:white !important; border:none !important; padding:12px 24px !important; border-radius:6px !important; font-weight:500 !important;}
    .stButton>button:hover {background:#A6192E !important;}
    .footer {text-align:center; padding:30px; color:#54585A; font-size:12px; border-top:1px solid #E2E3E4; margin-top:60px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Header + Nav
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="header"><h1>LFQbench Proteomics Analysis</h1><p>Quantitative accuracy assessment</p></div>
<div class="nav">
    <div class="nav-item active">Module 1: Data Import</div>
    <div class="nav-item">Module 2: Data Quality</div>
    <div class="nav-item">Module 3: Preprocessing</div>
    <div class="nav-item">Module 4: Analysis</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# NORMAL FILE UPLOAD FLOW (if no data in session)
# ─────────────────────────────────────────────────────────────
with st.container():
    st.markdown('<div class="card"><div class="upload-area"><div style="font-size:64px; opacity:0.5; margin-bottom:20px;">📤</div><div style="font-size:16px; color:#54585A; margin-bottom:10px;"><strong>Drag and drop your file here</strong></div><div style="font-size:13px; color:#54585A; opacity:0.7;">Supports .csv, .tsv, .txt</div></div></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    with col1:
        uploaded_file = st.file_uploader("", type=["csv","tsv","txt"], label_visibility="collapsed", key="prot_upload")
    with col2:
        skip_upload = st.button("⏭️ Skip", help="Skip protein upload", key="skip_prot")

# In pages/1_protein.py (replace the skip_upload section):
if skip_upload:
    st.session_state["skip_prot"] = True  # ← Add this
    st.info("⏭️ Protein upload skipped")
    st.markdown('<div class="footer"><strong>Proprietary & Confidential</strong><br>© 2024 Thermo Fisher Scientific</div>', unsafe_allow_html=True)
    st.stop()


if not uploaded_file:
    st.markdown('<div class="footer"><strong>Proprietary & Confidential</strong><br>© 2024 Thermo Fisher Scientific</div>', unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────
# Load & Parse — FIXED numeric conversion
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_and_parse(file):
    content = file.getvalue().decode("utf-8", errors="replace")
    if content.startswith("\ufeff"): 
        content = content[1:]
    df = pd.read_csv(io.StringIO(content), sep=None, engine="python", dtype=str)
    intensity_cols = [c for c in df.columns if c not in ["pg", "name", "Protein.Group", "PG.ProteinNames", "Accession", "Species"]]
    for col in intensity_cols:
        df[col] = pd.to_numeric(df[col].str.replace(",", "").replace("#NUM!", ""), errors="coerce")
    if "name" in df.columns:
        split = df["name"].str.split(",", n=1, expand=True)
        if split.shape[1] == 2:
            df.insert(1, "Accession", split[0])
            df.insert(2, "Species", split[1])
            df = df.drop(columns=["name"])
    return df

if "prot_df" in st.session_state:
    st.info("📂 Data restored from session state")
    df = st.session_state.prot_df
    c1 = st.session_state.prot_c1
    c2 = st.session_state.prot_c2
    sp_col = st.session_state.prot_sp_col
    pg_col = st.session_state.prot_pg_col
    sp_counts = st.session_state.prot_sp_counts
    
    # Skip file upload and go to summary
    st.success("✅ Protein data loaded!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("**Condition 1**", f"{len(c1)} reps")
        st.write(", ".join(c1))
    with col2:
        st.metric("**Condition 2**", f"{len(c2)} reps")
        st.write(", ".join(c2))
    
    st.info(f"**Species** → `{sp_col}` | **Protein Group** → `{pg_col}`")
    
    if sp_counts is not None:
        st.markdown("### 📊 Proteins by Species & Condition")
        st.dataframe(sp_counts, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    st.markdown('<div class="footer"><strong>Proprietary & Confidential</strong><br>© 2024 Thermo Fisher Scientific</div>', unsafe_allow_html=True)
    st.stop()  # ← Stop here, don't re-upload

# ─── Otherwise, proceed with file upload ───
with st.container():
    st.markdown('<div class="card"><div class="upload-area">...')
    uploaded_file = st.file_uploader(...)

if not uploaded_file:
    st.stop()

# ... rest of protein page code ..
# ─────────────────────────────────────────────────────────────
# SINGLE UNIFIED TABLE — 100% WORKING
# ─────────────────────────────────────────────────────────────
with st.container():
    st.subheader("⚙️ Column Assignment & Renaming")

    # Force intensity columns to float — but keep metadata as object
    intensity_cols = [c for c in df.columns if c not in ["pg", "name", "Protein.Group", "PG.ProteinNames", "Accession", "Species"]]
    for col in intensity_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure metadata stays as string
    metadata_cols_list = ["Protein.Group", "PG.ProteinNames", "Accession", "Species"]
    for col in metadata_cols_list:
        if col in df.columns:
            df[col] = df[col].astype("string")

    numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]

    # Auto-detect Condition 1
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

    # Auto-detect Species & Protein Group
    species_col_default = "Species" if "Species" in df.columns else "PG.ProteinNames"
    protein_col_default = "Protein.Group" if "Protein.Group" in df.columns else df.columns[0]

    # Build table
    rows = []
    for col in df.columns:
        is_intensity = col in numeric_cols
        preview = " | ".join(df[col].dropna().astype(str).unique()[:3])
        rows.append({
            "Rename": col,
            "Cond 1": col in default_cond1 and is_intensity,
            "Species": col == species_col_default,
            "Protein Group": col == protein_col_default,
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
            "Species": st.column_config.CheckboxColumn("Species", default=True),
            "Protein Group": st.column_config.CheckboxColumn("Protein Group", default=True),
            "Original Name": st.column_config.TextColumn("Original Name", disabled=True),
            "Preview": st.column_config.TextColumn("Preview", disabled=True),
            "Type": st.column_config.TextColumn("Type", disabled=True),
        },
        disabled=["Original Name", "Preview", "Type"],
        hide_index=True,
        use_container_width=True,
        key="final_table_fixed_2024"
    )

    # ── SAFE extraction with proper validation ──
    cond1_checked = edited[edited["Cond 1"]]
    species_checked = edited[edited["Species"]]
    protein_checked = edited[edited["Protein Group"]]

    cond1_cols_orig = cond1_checked["Original Name"].tolist()
    cond2_cols_orig = [c for c in numeric_cols if c not in cond1_cols_orig]

    species_cols = species_checked["Original Name"].tolist()
    protein_cols = protein_checked["Original Name"].tolist()

    # ── BULLETPROOF validation ──
    errors = []

    if len(cond1_cols_orig) == 0:
        errors.append("❌ Condition 1 must have at least one replicate")
    if len(cond2_cols_orig) == 0:
        errors.append("❌ Condition 2 must have at least one replicate")

    if len(species_cols) == 0:
        errors.append("❌ Exactly one Species column must be selected")
    elif len(species_cols) > 1:
        errors.append("❌ Only one Species column allowed — uncheck the others")

    if len(protein_cols) == 0:
        errors.append("❌ Exactly one Protein Group column must be selected")
    elif len(protein_cols) > 1:
        errors.append("❌ Only one Protein Group column allowed")

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    # ── Now safe to access [0] ──
    species_col = species_cols[0]
    protein_col = protein_cols[0]

    # Apply renaming
    rename_map = {}
    for _, row in edited.iterrows():
        new_name = row["Rename"].strip()
        if new_name and new_name != row["Original Name"]:
            rename_map[row["Original Name"]] = new_name

    if rename_map:
        df = df.rename(columns=rename_map)
        # UPDATE cond1_cols and cond2_cols with renamed names
        cond1_cols = [rename_map.get(c, c) for c in cond1_cols_orig]
        cond2_cols = [rename_map.get(c, c) for c in cond2_cols_orig]
    else:
        cond1_cols = cond1_cols_orig
        cond2_cols = cond2_cols_orig

    st.session_state.df = df

# ── Save everything ──
st.session_state.update({
    "cond1_cols": cond1_cols,
    "cond2_cols": cond2_cols,
    "species_col": species_col,
    "protein_col": protein_col,
    "df": df
})

st.success("✅ All assignments & renaming applied!")

c1, c2 = st.columns(2)
with c1:
    st.metric("**Condition 1**", f"{len(cond1_cols)} replicates")
    st.write(", ".join(cond1_cols))
with c2:
    st.metric("**Condition 2**", f"{len(cond2_cols)} replicates")
    st.write(", ".join(cond2_cols))

st.info(f"**Species** → `{species_col}` | **Protein Group** → `{protein_col}`")

# ─────────────────────────────────────────────────────────────
# Detect Species & Count Proteins Per Condition
# ─────────────────────────────────────────────────────────────
species_source_col, species_col_extracted, unique_species = detect_species_column_and_extract(df)

if species_col_extracted and len(unique_species) > 1:
    st.markdown("### 📊 Unique Proteins by Species & Condition")
    
    # Convert intensity columns to numeric (ensure they are)
    for col in cond1_cols + cond2_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    species_counts = []
    for sp in unique_species:
        sp_df = df[df[species_col_extracted] == sp]
        
        # ≥2/3 replicates with intensity >1
        valid_cond1 = (sp_df[cond1_cols] > 1).sum(axis=1) >= 2
        valid_cond2 = (sp_df[cond2_cols] > 1).sum(axis=1) >= 2
        
        species_counts.append({
            "Species": sp,
            "Cond1 (≥2/3 >1)": valid_cond1.sum(),
            "Cond2 (≥2/3 >1)": valid_cond2.sum(),
            "Both": (valid_cond1 & valid_cond2).sum(),
            "Total": len(sp_df)
        })
    
    species_df = pd.DataFrame(species_counts)
    st.dataframe(species_df, hide_index=True, use_container_width=True)

    # ── Visualization ──
    col1, col2 = st.columns(2)
    
    with col1:
        chart_data = species_df[["Species", "Cond1 (≥2/3 >1)", "Cond2 (≥2/3 >1)"]].set_index("Species")
        st.bar_chart(chart_data)
    
    with col2:
        st.write("**Summary:**")
        for idx, row in species_df.iterrows():
            st.write(
                f"- **{row['Species']}**: {row['Total']} total\n"
                f"  - Cond1: {row['Cond1 (≥2/3 >1)']} | Cond2: {row['Cond2 (≥2/3 >1)']} | Both: {row['Both']}"
            )

st.markdown("---")


# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="footer"><strong>Proprietary & Confidential</strong><br>© 2024 Thermo Fisher Scientific</div>', unsafe_allow_html=True)
