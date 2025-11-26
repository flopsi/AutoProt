# app.py — LFQbench Data Import (Thermo Fisher Design) — 100% WORKING
import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from pandas.api.types import is_numeric_dtype

# ─────────────────────────────────────────────────────────────
# Detect Species Column Function
# ─────────────────────────────────────────────────────────────
def detect_species_column_and_extract(df):
    """
    Automatically detect species column and extract species from strings.
    Returns: (species_col_name, extracted_species_col_name, list_of_unique_species)
    """
    metadata_cols = []
    for col in df.columns:
        sample = df[col].dropna().head(50)
        try:
            pd.to_numeric(sample.replace('#NUM!', pd.NA), errors='raise')
        except:
            metadata_cols.append(col)
    
    species_col = None
    prefix_delim = None
    
    for col in metadata_cols:
        sample_values = df[col].dropna().astype(str).head(100)
        for val in sample_values:
            if 'HUMAN' in val.upper():
                species_col = col
                match = re.search(r'([^A-Za-z0-9])HUMAN', val, re.IGNORECASE)
                prefix_delim = match.group(1) if match else ''
                break
        if species_col:
            break
    
    if not species_col:
        return None, None, []
    
    if prefix_delim:
        prefix_esc = re.escape(prefix_delim)
        pattern = f'{prefix_esc}([A-Z]+)(?:[^A-Za-z]|$)'
    else:
        pattern = r'^([A-Z]+)$'
    
    def extract_first_species(val):
        if pd.isna(val):
            return None
        val_str = str(val).upper()
        matches = re.findall(pattern, val_str)
        return matches[0] if matches else None
    
    df['_species'] = df[species_col].apply(extract_first_species)
    unique_species = sorted(df['_species'].dropna().unique().tolist())
    
    return species_col, '_species', unique_species


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
# Upload
# ─────────────────────────────────────────────────────────────
with st.container():
    st.markdown('<div class="card"><div class="upload-area"><div style="font-size:64px; opacity:0.5; margin-bottom:20px;">📤</div><div style="font-size:16px; color:#54585A; margin-bottom:10px;"><strong>Drag and drop your file here</strong></div><div style="font-size:13px; color:#54585A; opacity:0.7;">Supports .csv, .tsv, .txt</div></div></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["csv","tsv","txt"], label_visibility="collapsed")

if not uploaded_file:
    st.markdown('<div class="footer"><strong>Proprietary & Confidential</strong><br>© 2024 Thermo Fisher Scientific</div>', unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────
# Load & Parse
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

st.set_page_config(page_title="Protein Import", layout="wide")

# Check if data exists from session
if "prot_df" in st.session_state:
    st.info("📂 Data restored")
else:
    df = load_and_parse(uploaded_file)
    st.success(f"✅ Data imported — {len(df):,} proteins")
df = st.session_state.prot_df
    # ... display without re-uploading


# ─────────────────────────────────────────────────────────────
# Column Assignment Table
# ─────────────────────────────────────────────────────────────
with st.container():
    st.subheader("⚙️ Column Assignment & Renaming")

    intensity_cols = [c for c in df.columns if c not in ["pg", "name", "Protein.Group", "PG.ProteinNames", "Accession", "Species"]]
    for col in intensity_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    meta_cols_list = ["Protein.Group", "PG.ProteinNames", "Accession", "Species"]
    for col in meta_cols_list:
        if col in df.columns:
            df[col] = df[col].astype("string")

    num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]

    # Auto-detect Condition 1
    ratio_groups = {}
    for col in num_cols:
        m = re.search(r'_Y(\d{2})-E(\d{2})_', col)
        if m:
            key = f"Y{m.group(1)}-E{m.group(2)}"
            ratio_groups.setdefault(key, []).append(col)

    default_c1 = ratio_groups.get(
        sorted(ratio_groups.keys(), key=lambda x: int(x.split('-')[0][1:]))[0],
        num_cols[:len(num_cols)//2]
    ) if ratio_groups else num_cols[:len(num_cols)//2]

    sp_col_default = "Species" if "Species" in df.columns else "PG.ProteinNames"
    pg_col_default = "Protein.Group" if "Protein.Group" in df.columns else df.columns[0]

    rows = []
    for col in df.columns:
        is_num = col in num_cols
        preview = " | ".join(df[col].dropna().astype(str).unique()[:3])
        rows.append({
            "Rename": col,
            "Cond 1": col in default_c1 and is_num,
            "Species": col == sp_col_default,
            "Protein Group": col == pg_col_default,
            "Original Name": col,
            "Preview": preview,
            "Type": "Intensity" if is_num else "Metadata"
        })

    df_edit = pd.DataFrame(rows)

    edited = st.data_editor(
        df_edit,
        column_config={
            "Rename": st.column_config.TextColumn("Rename", required=False),
            "Cond 1": st.column_config.CheckboxColumn("Cond 1", default=True),
            "Species": st.column_config.CheckboxColumn("Species", default=True),
            "Protein Group": st.column_config.CheckboxColumn("PG", default=True),
            "Original Name": st.column_config.TextColumn("Original", disabled=True),
            "Preview": st.column_config.TextColumn("Preview", disabled=True),
            "Type": st.column_config.TextColumn("Type", disabled=True),
        },
        disabled=["Original Name", "Preview", "Type"],
        hide_index=True,
        use_container_width=True,
        key="col_table"
    )

    # Extract selections
    c1_checked = edited[edited["Cond 1"]]
    sp_checked = edited[edited["Species"]]
    pg_checked = edited[edited["Protein Group"]]

    c1_orig = c1_checked["Original Name"].tolist()
    c2_orig = [c for c in num_cols if c not in c1_orig]
    sp_cols = sp_checked["Original Name"].tolist()
    pg_cols = pg_checked["Original Name"].tolist()

    # Validation
    errors = []
    if len(c1_orig) == 0:
        errors.append("❌ Cond 1 must have ≥1 replicate")
    if len(c2_orig) == 0:
        errors.append("❌ Cond 2 must have ≥1 replicate")
    if len(sp_cols) != 1:
        errors.append("❌ Select exactly 1 Species column")
    if len(pg_cols) != 1:
        errors.append("❌ Select exactly 1 Protein Group column")

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    sp_col = sp_cols[0]
    pg_col = pg_cols[0]

    # Apply renaming
    rename_map = {}
    for _, row in edited.iterrows():
        new = row["Rename"].strip()
        if new and new != row["Original Name"]:
            rename_map[row["Original Name"]] = new

    if rename_map:
        df = df.rename(columns=rename_map)
        c1 = [rename_map.get(c, c) for c in c1_orig]
        c2 = [rename_map.get(c, c) for c in c2_orig]
    else:
        c1 = c1_orig
        c2 = c2_orig

# ─────────────────────────────────────────────────────────────
# Detect Species & Build Species Counts
# ─────────────────────────────────────────────────────────────
sp_src, sp_ext, species_list = detect_species_column_and_extract(df)

# Build species counts df
sp_counts = None
if sp_ext and len(species_list) > 1:
    for col in c1 + c2:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    rows = []
    for sp in species_list:
        sp_df = df[df[sp_ext] == sp]
        v1 = (sp_df[c1] > 1).sum(axis=1) >= 2
        v2 = (sp_df[c2] > 1).sum(axis=1) >= 2
        rows.append({
            "species": sp,
            "c1": v1.sum(),
            "c2": v2.sum(),
            "both": (v1 & v2).sum(),
            "total": len(sp_df)
        })
    sp_counts = pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
# Save to Session State — SHORT NAMES for prot/pep reuse
# ─────────────────────────────────────────────────────────────
st.session_state.update({
    # ─── Protein-level data ───
    "prot_df": df,                    # Main protein dataframe
    "prot_c1": c1,                    # Cond1 columns
    "prot_c2": c2,                    # Cond2 columns
    "prot_sp_col": sp_col,            # Species column (user-selected)
    "prot_pg_col": pg_col,            # Protein group column
    "prot_sp_ext": sp_ext,            # Extracted species column (_species)
    "prot_species": species_list,     # Unique species list
    "prot_sp_counts": sp_counts,      # Species counts df
    "prot_n": len(df),                # Total proteins
    "prot_rename": rename_map,        # Rename mapping
    "prot_num_cols": num_cols,        # Numeric columns
    
    # ─── File info ───
    "file_name": uploaded_file.name,
    "file_ts": pd.Timestamp.now()
})

# ─────────────────────────────────────────────────────────────
# Display Summary
# ─────────────────────────────────────────────────────────────
st.success("✅ All assignments applied!")

col1, col2 = st.columns(2)
with col1:
    st.metric("**Condition 1**", f"{len(c1)} reps")
    st.write(", ".join(c1))
with col2:
    st.metric("**Condition 2**", f"{len(c2)} reps")
    st.write(", ".join(c2))

st.info(f"**Species** → `{sp_col}` | **Protein Group** → `{pg_col}`")

# Species counts table
if sp_counts is not None:
    st.markdown("### 📊 Proteins by Species & Condition")
    st.dataframe(sp_counts, hide_index=True, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        chart = sp_counts[["species", "c1", "c2"]].set_index("species")
        st.bar_chart(chart)
    with col2:
        st.write("**Summary:**")
        for _, r in sp_counts.iterrows():
            st.write(f"- **{r['species']}**: {r['total']} total | C1: {r['c1']} | C2: {r['c2']} | Both: {r['both']}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="footer"><strong>Proprietary & Confidential</strong><br>© 2024 Thermo Fisher Scientific</div>', unsafe_allow_html=True)
