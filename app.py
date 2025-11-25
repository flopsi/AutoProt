# app.py — LFQbench Data Import Module (Thermo Fisher Corporate Design)
import streamlit as st
import pandas as pd
import numpy as np
import io

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
# Full Thermo Fisher CSS (pixel-perfect match to your mockup)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --primary-red: #E71316; --dark-red: #A6192E; --gray: #54585A;
        --light-gray: #E2E3E4; --navy: #262262; --green: #B5BD00;
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
# Header + Navigation
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

# ─────────────────────────────────────────────────────────────
# Module Header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="module-header">
    <div class="module-icon">Upload</div>
    <div>
        <h2>Module 1: Data Import & Validation</h2>
        <p>Upload and validate your LFQ intensity matrix</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 1. File Upload
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
# 2. Load & Parse — Fixed low_memory + species extraction
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_and_parse(file):
    content = file.getvalue().decode("utf-8", errors="replace")
    if content.startswith("\ufeff"):
        content = content[1:]
    # Fixed: Use dtype=str to avoid low_memory warnings
    df = pd.read_csv(io.StringIO(content), sep=None, engine="python", dtype=str)
    
    # Convert intensity columns to numeric
    intensity_cols = [c for c in df.columns if c not in ["pg", "name"]]
    for col in intensity_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Extract species from 'name' column
    if "name" in df.columns:
        split = df["name"].str.split(",", n=1, expand=True)
        if split.shape[1] == 2:
            df.insert(1, "Accession", split[0])
            df.insert(2, "Species", split[1])
            df = df.drop(columns=["name"])

    return df

df = load_and_parse(uploaded_file)
st.session_state.df = df

st.success(f"Data imported successfully — {len(df):,} proteins, {len(df.columns)} columns")
st.dataframe(df.head(10), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# 3. Column Renaming (optional)
# ─────────────────────────────────────────────────────────────
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Column Renaming (optional)")

    rename_dict = {}
    cols = df.columns.tolist()
    col1, col2 = st.columns(2)
    for i, col in enumerate(cols):
        with (col1 if i % 2 == 0 else col2):
            new_name = st.text_input(f"`{col}` →", value=col, key=f"rename_{col}")
            if new_name != col and new_name.strip():
                rename_dict[col] = new_name.strip()

    if st.button("Apply Renaming"):
        if rename_dict:
            df = df.rename(columns=rename_dict)
            st.session_state.df = df
            st.success("Columns renamed successfully")
            st.rerun()
        else:
            st.info("No changes made")

    st.markdown("</div>", unsafe_allow_html=True)

# In the condition assignment section (replace the old one):

# ─────────────────────────────────────────────────────────────
# 4. SMART Condition Assignment + Replica Preview Table (with checkboxes)
# ─────────────────────────────────────────────────────────────
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Condition Assignment")

    # All numeric (intensity) columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # ── Auto-detect ratio groups from filename pattern Yxx-Exx ──
    import re
    ratio_groups = {}
    for col in numeric_cols:
        match = re.search(r'_Y(\d{2})-E(\d{2})_', col)
        if match:
            y = int(match.group(1))
            e = int(match.group(2))
            key = f"Y{y:02d}-E{e:02d}"
            ratio_groups.setdefault(key, []).append(col)

    # ── Auto-split into two conditions (lowest yeast % → Cond1, highest → Cond2)
    if len(ratio_groups) >= 2:
        sorted_keys = sorted(ratio_groups.keys(), key=lambda x: int(x.split('-')[0][1:]))
        cond1_key = sorted_keys[0]
        cond2_key = sorted_keys[-1]
        default_cond1 = ratio_groups[cond1_key]
        default_cond2 = ratio_groups[cond2_key]
    else:
        # Fallback: even split
        half = len(numeric_cols) // 2
        default_cond1 = numeric_cols[:half]
        default_cond2 = numeric_cols[half:]

    # ── Preview table with pre-selected checkboxes ──
    st.write("**Select replicates for Condition 1** (all others will automatically go to Condition 2)")

    preview_data = []
    for col in numeric_cols:
        preview_data.append({
            "Include in Cond 1": col in default_cond1,
            "Column": col,
            "Suggested Condition": "Condition 1" if col in default_cond1 else "Condition 2"
        })

    preview_df = pd.DataFrame(preview_data)

    edited_df = st.data_editor(
            preview_df,
            column_config={
                "Include in Cond 1": st.column_config.CheckboxColumn(
                    "Include in Cond 1",          # ← this is the display label
                    help="Check = assign to Condition 1",
                    default_value=True            # ← pre-selects auto-detected replicates
                ),
                "Column": st.column_config.TextColumn(
                    "Column",
                    disabled=True
                ),
                "Suggested Condition": st.column_config.TextColumn(
                    "Suggested Condition",
                    disabled=True
                )
            },
            disabled=["Column", "Suggested Condition"],
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            key="condition_assignment_table"   # ← required for stateful editor
    )
# ────────────────────────────────────────────────────────────────────────

    # Extract final selection
    final_cond1 = edited_df[edited_df["Include in Cond 1"]]["Column"].tolist()
    final_cond2 = [c for c in numeric_cols if c not in final_cond1]

    if len(final_cond1) == 0 or len(final_cond2) == 0:
        st.error("At least one replicate must be assigned to each condition.")
        st.stop()

    st.session_state.cond1_cols = final_cond1
    st.session_state.cond2_cols = final_cond2

    st.success(f"Condition 1: {len(final_cond1)} replicates | Condition 2: {len(final_cond2)} replicates")

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 5. Auto-detect Species and Protein Group columns
# ─────────────────────────────────────────────────────────────
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Metadata Columns")

    # Species column
    species_candidates = [c for c in df.columns if "HUMAN" in df[c].astype(str).str.upper().any()]
    if "Species" in df.columns:
        species_col = "Species"
    elif species_candidates:
        species_col = species_candidates[0]
    else:
        species_col = st.selectbox("Select Species column (contains HUMAN/YEAST/ECOLI)", df.columns)

    # Protein Group / Accession column
    protein_candidates = [c for c in df.columns if "protein" in c.lower()]
    if protein_candidates:
        protein_col = protein_candidates[0]
    else:
        protein_col = df.columns[0]  # fallback

    st.info(f"**Protein IDs** → `{protein_col}`  |  **Species** → `{species_col}`")

    st.session_state.species_col = species_col
    st.session_state.protein_col = protein_col

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 6. Clean Replica Names (auto-trimmed)
# ─────────────────────────────────────────────────────────────
clean_names = {}
for col in numeric_cols:
    # Remove date + everything before last underscore
    clean = col.split("_")[-1].replace(".raw", "")
    clean_names[col] = f"Rep {clean}"

if st.checkbox("Use clean replica names in plots/tables", value=True):
    st.session_state.clean_names = clean_names
else:
    st.session_state.clean_names = {col: col for col in numeric_cols}

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <strong>Proprietary & Confidential | For Internal Use Only</strong><br>
    © 2024 Thermo Fisher Scientific Inc. All rights reserved.<br>
    Contact: proteomics.bioinformatics@thermofisher.com | Version 1.0
</div>
""", unsafe_allow_html=True)
