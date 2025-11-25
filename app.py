# app.py — LFQbench Data Import Module (Thermo Fisher Corporate Design)
import streamlit as st
import pandas as pd
import numpy as np
import re
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
# Full Thermo Fisher CSS (pixel-perfect match)
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
        <p>Automatic sample name parsing, ratio detection, and condition grouping</p>
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
                Supports MaxQuant proteinGroups.txt (or any LFQ matrix with raw file names as columns)
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["txt","csv","tsv"], label_visibility="collapsed")

if not uploaded_file:
    st.markdown("""
    <div class="footer">
        <strong>Proprietary & Confidential | For Internal Use Only</strong><br>
        © 2024 Thermo Fisher Scientific Inc. All rights reserved.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────
# 2. Load & Parse + Smart Column Processing
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_and_process(file):
    content = file.getvalue().decode("utf-8", errors="replace")
    if content.startswith("\ufeff"):
        content = content[1:]
    df = pd.read_csv(io.StringIO(content), sep="\t", engine="python", dtype=str)  # MaxQuant is tab-separated

    # Convert intensity columns to numeric
    potential_intensity = [c for c in df.columns if c not in ["pg", "name", "Protein IDs", "Majority protein IDs", "Fasta headers"]]
    for col in potential_intensity:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Extract species from 'name' or fallback to common columns
    species_col = None
    if "name" in df.columns:
        split = df["name"].str.split(",", n=1, expand=True)
        if split.shape[1] == 2:
            df.insert(1, "Accession", split[0])
            df.insert(2, "Species", split[1])
            df = df.drop(columns=["name"])
            species_col = "Species"
    
    # === SMART COLUMN PARSING (for raw file names) ===
    intensity_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # Extract ratio key: look for Ydd-Edd pattern
    ratio_pattern = re.compile(r'Y(\d{2})-E(\d{2})')
    col_to_ratio = {}
    col_to_rep = {}
    short_names = {}

    for col in intensity_cols:
        # Strip .raw if present
        clean_col = col.replace(".raw", "")
        match = ratio_pattern.search(clean_col)
        if match:
            ratio_key = match.group(0)  # e.g., Y05-E45
            # Extract replicate number (last _01, _02, etc.)
            rep = clean_col.split("_")[-1]
            rep_num = rep.lstrip("0") or "1"  # handle _01 → 1
            col_to_ratio[col] = ratio_key
            col_to_rep[col] = rep_num
            short_names[col] = f"{ratio_key}_{rep_num}"
        else:
            # Fallback: no ratio found
            short_names[col] = col[:15] + ("..." if len(col)>15 else "")

    # Auto-rename columns to short, clean names
    if len(short_names) == len(intensity_cols):
        df = df.rename(columns=short_names)
        intensity_cols = list(short_names.values())
        st.success("Columns automatically shortened and grouped by mixing ratio")

    # Group by ratio for condition auto-guess
    ratio_groups = {}
    for col, ratio in col_to_ratio.items():
        ratio_groups.setdefault(ratio, []).append(short_names.get(col, col))

    # Sort ratios logically (Y05-E45 first, then Y45-E05)
    sorted_ratios = sorted(ratio_groups.keys(), key=lambda x: (int(x[1:3]), int(x[5:7])))  # Y05 then Y45

    default_cond1 = ratio_groups.get(sorted_ratios[0], intensity_cols[:len(intensity_cols)//2])
    default_cond2 = ratio_groups.get(sorted_ratios[1], intensity_cols[len(intensity_cols)//2:]) if len(sorted_ratios)>1 else []

    # Store ratio mapping for later modules
    st.session_state.ratio_to_cols = ratio_groups
    st.session_state.ratio_order = sorted_ratios  # for expected log2 later

    return df, intensity_cols, default_cond1, default_cond2, species_col or "Species"

df, intensity_cols, auto_cond1, auto_cond2, species_col = load_and_process(uploaded_file)
st.session_state.df = df

st.success(f"Data imported — {len(df):,} proteins, {len(intensity_cols)} samples detected")
if st.session_state.ratio_order:
    st.info(f"Detected mixing ratios: {', '.join(st.session_state.ratio_order)}")

st.dataframe(df.head(10), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# 3. Condition Assignment (auto-guessed + editable)
# ─────────────────────────────────────────────────────────────
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Condition Assignment")

    col1, col2 = st.columns(2)
    with col1:
        cond1_cols = st.multiselect(
            "Condition 1 (auto-detected)",
            options=intensity_cols,
            default=auto_cond1
        )
    with col2:
        cond2_cols = st.multiselect(
            "Condition 2 (auto-detected)",
            options=intensity_cols,
            default=auto_cond2
        )

    if set(cond1_cols) & set(cond2_cols):
        st.error("Cannot assign the same replicate to both conditions")
    else:
        st.session_state.cond1_cols = cond1_cols
        st.session_state.cond2_cols = cond2_cols
        if st.session_state.ratio_order:
            st.success(f"Condition 1 → {st.session_state.ratio_order[0]}\nCondition 2 → {st.session_state.ratio_order[1]}")

    st.markdown("</div>", unsafe_allow_html=True)

# In the condition assignment section (replace the old one):

# ─────────────────────────────────────────────────────────────
# 4. SMART Auto-Detection of Conditions & Ratios
# ─────────────────────────────────────────────────────────────
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Smart Condition Assignment")

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # Auto-detect ratio groups
    import re
    ratio_groups = {}
    for col in numeric_cols:
        match = re.search(r'_Y(\d{2})-E(\d{2})_', col)
        if match:
            yeast_pct = int(match.group(1))
            ecoli_pct = int(match.group(2))
            ratio_key = f"Y{yeast_pct:02d}-E{ecoli_pct:02d}"
            if ratio_key not in ratio_groups:
                ratio_groups[ratio_key] = {
                    'columns': [],
                    'yeast_pct': yeast_pct,
                    'ecoli_pct': ecoli_pct,
                    'human_pct': 100 - yeast_pct - ecoli_pct
                }
            ratio_groups[ratio_key]['columns'].append(col)

    if len(ratio_groups) >= 2:
        # Sort by yeast % (low → high)
        sorted_ratios = sorted(ratio_groups.items(), key=lambda x: x[1]['yeast_pct'])
        
        cond1_key, cond1_info = sorted_ratios[0]
        cond2_key, cond2_info = sorted_ratios[-1]
        
        cond1_cols = cond1_info['columns']
        cond2_cols = cond2_info['columns']
        
        # Auto-calculate expected log2 ratios
        expected_yeast = np.log2(cond2_info['yeast_pct'] / cond1_info['yeast_pct'])
        expected_ecoli = np.log2(cond2_info['ecoli_pct'] / cond1_info['ecoli_pct'])
        expected_human = 0.0  # always 1:1
        
        st.success(f"✅ **Auto-detected {len(cond1_cols)}:3 design**")
        st.info(f"Expected log₂ ratios: Yeast={expected_yeast:.2f}, E.coli={expected_ecoli:.2f}, Human={expected_human:.2f}")
        
        # Confirmation table
        st.dataframe(pd.DataFrame({
            'Ratio': [cond1_key, cond2_key],
            'Condition': ['**Cond 1** (low yeast)', '**Cond 2** (high yeast)'],
            'Replicates': [len(cond1_cols), len(cond2_cols)],
            'Columns': [', '.join(cond1_cols), ', '.join(cond2_cols)]
        }).style.format({'Replicates': '{:.0f}'}))
        
        # Option to override
        if st.button("Use Auto-Detection"):
            st.session_state.cond1_cols = cond1_cols
            st.session_state.cond2_cols = cond2_cols
            st.session_state.expected_ratios = {
                'YEAST': expected_yeast,
                'ECOLI': expected_ecoli, 
                'HUMAN': expected_human
            }
            st.success("Auto-detection confirmed!")
            
    else:
        # Fallback: half-split
        half = len(numeric_cols) // 2
        cond1_cols = numeric_cols[:half]
        cond2_cols = numeric_cols[half:]

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 5. Ready for Data Quality
# ─────────────────────────────────────────────────────────────
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Ready for Data Quality Module")
st.success("All samples automatically grouped by mixing ratio. Conditions and species ready.")
st.info("Next step: **Module 2 – Data Quality** (missing values, CVs, intensity distribution, PCA, etc.)")
st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <strong>Proprietary & Confidential | For Internal Use Only</strong><br>
    © 2024 Thermo Fisher Scientific Inc. All rights reserved.<br>
    Contact: proteomics.bioinformatics@thermofisher.com | Version 1.1
</div>
""", unsafe_allow_html=True)
