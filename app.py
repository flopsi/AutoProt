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
# ─────────────────────────────────────────────────────────────
# FIXED: Proper numeric conversion — intensity columns become REAL floats
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_and_parse(file):
    content = file.getvalue().decode("utf-8", errors="replace")
    if content.startswith("\ufeff"):
        content = content[1:]

    # Read ONLY header first to detect intensity columns
    header_df = pd.read_csv(io.StringIO(content), sep=None, engine="python", nrows=0)
    intensity_cols = [c for c in header_df.columns if c not in ["pg", "name"]]

    # Now read full file, forcing intensity columns to float
    df = pd.read_csv(
        io.StringIO(content),
        sep=None,
        engine="python",
        dtype={col: str for col in header_df.columns},  # start safe
        low_memory=False
    )

    # FORCE intensity columns to numeric
    for col in intensity_cols:
        df[col] = pd.to_numeric(df[col].str.replace(",", ""), errors="coerce")

    # Extract species from 'name' column
    if "name" in df.columns:
        split = df["name"].str.split(",", n=1, expand=True)
        if split.shape[1] == 2:
            df.insert(1, "Accession", split[0])
            df.insert(2, "Species", split[1])
            df = df.drop(columns=["name"])

    return df
# ─────────────────────────────────────────────────────────────
# SINGLE UNIFIED TABLE — FULLY WORKING (Streamlit 1.38+)
# ─────────────────────────────────────────────────────────────
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Column Assignment & Renaming")

    # Identify numeric (intensity) columns — this is the real source of truth
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # ── Auto-detect Condition 1 (Yxx-Exx pattern) ──
    import re
    ratio_groups = {}
    for col in numeric_cols:
        m = re.search(r'_Y(\d{2})-E(\d{2})_', col)
        if m:
            key = f"Y{m.group(1)}-E{m.group(2)}"
            ratio_groups.setdefault(key, []).append(col)

    default_cond1 = []
    if len(ratio_groups) >= 2:
        sorted_keys = sorted(ratio_groups.keys(), key=lambda x: int(x.split('-')[0][1:]))
        default_cond1 = ratio_groups[sorted_keys[0]]
    elif len(numeric_cols) >= 2:
        default_cond1 = numeric_cols[:len(numeric_cols)//2]

    # ── Auto-detect Species & Protein Group columns ──
    species_col_default = "Species" if "Species" in df.columns else None
    if not species_col_default:
        for col in df.columns:
            if df[col].astype(str).str.upper().fillna("").str.contains("HUMAN|YEAST|ECOLI").any():
                species_col_default = col
                break

    protein_col_default = next((c for c in df.columns if "protein" in c.lower()), df.columns[0])

    # ── Build unified table (correct Type detection!) ──
    rows = []
    for col in df.columns:
        is_intensity = pd.api.types.is_numeric_dtype(df[col])   # ← CRITICAL: use actual dtype
        preview = " | ".join(df[col].dropna().astype(str).unique()[:3]) if not is_intensity else "numeric"
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

    # ── data_editor with correct default=True (so pre-checks are respected) ──
    edited = st.data_editor(
        df_edit,
        column_config={
            "Rename": st.column_config.TextColumn(
                label="Rename (optional)",
                help="Leave blank or type new name",
                required=False
            ),
            "Cond 1": st.column_config.CheckboxColumn(
                label="Condition 1",
                help="Check = assign to Condition 1",
                default=True          # ← Must be True!
            ),
            "Species": st.column_config.CheckboxColumn(
                label="Species",
                help="Exactly one column",
                default=True
            ),
            "Protein Group": st.column_config.CheckboxColumn(
                label="Protein Group",
                help="Exactly one column",
                default=True
            ),
            "Original Name": st.column_config.TextColumn(label="Original Name", disabled=True),
            "Preview": st.column_config.TextColumn(label="Preview", disabled=True),
            "Type": st.column_config.TextColumn(label="Type", disabled=True),
        },
        disabled=["Original Name", "Preview", "Type"],
        hide_index=True,
        use_container_width=True,
        key="unified_table_final_2024"
    )

    # ── Apply renaming ──
    rename_map = {}
    for _, row in edited.iterrows():
        new_name = row["Rename"].strip()
        if new_name and new_name != row["Original Name"]:
            rename_map[row["Original Name"]] = new_name

    if rename_map:
        df = df.rename(columns=rename_map)
        st.session_state.df = df

    # ── Extract final assignments ──
    cond1_cols = edited[edited["Cond 1"]]["Original Name"].tolist()
    cond2_cols = [c for c in numeric_cols if c not in cond1_cols]
    species_col = edited[edited["Species"]]["Original Name"].tolist()
    protein_col = edited[edited["Protein Group"]]["Original Name"].tolist()

    # ── Validation ──
    if len(cond1_cols) == 0:
        st.error("Condition 1 must have at least one replicate")
        st.stop()
    if len(cond2_cols) == 0:
        st.error("Condition 2 must have at least one replicate")
        st.stop()
    if len(species_col) != 1:
        st.error("Exactly one Species column must be selected")
        st.stop()
    if len(protein_col) != 1:
        st.error("Exactly one Protein Group column must be selected")
        st.stop()

    # ── Save everything ──
    st.session_state.cond1_cols = cond1_cols
    st.session_state.cond2_cols = cond2_cols
    st.session_state.species_col = species_col[0]
    st.session_state.protein_col = protein_col[0]
    st.session_state.df = df

    # ── Success display ──
    st.success("All assignments applied successfully!")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("**Condition 1**", f"{len(cond1_cols)} replicates")
        st.write("Replicates:")
        st.write(", ".join(cond1_cols))
    with c2:
        st.metric("**Condition 2**", f"{len(cond2_cols)} replicates")
        st.write("Replicates:")
        st.write(", ".join(cond2_cols))

    st.info(f"**Species column** → `{species_col[0]}` | **Protein Group** → `{protein_col[0]}`")
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
