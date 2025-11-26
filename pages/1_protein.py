# pages/1_Protein_Import.py
import streamlit as st
import pandas as pd
import re
import io

st.set_page_config(page_title="Data Import Module", layout="wide")

# ─────────────────────────────────────────────────────────────
# EXACT THERMO FISHER BRANDING & LAYOUT
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --primary-red: #E71316;
        --dark-red: #A6192E;
        --gray: #54585A;
        --light-gray: #E2E3E4;
        --navy: #262262;
    }
    .header {
        background: linear-gradient(90deg, var(--primary-red), var(--dark-red));
        padding: 20px 40px;
        color: white;
        margin: -80px -80px 40px -80px;
    }
    .header h1 { margin:0; font-size:28px; font-weight:600; }
    .header p { margin:5px 0 0; font-size:14px; opacity:0.95; }
    .nav {
        background: white;
        border-bottom: 2px solid var(--light-gray);
        padding: 0 40px;
        display: flex;
        gap: 5px;
        margin: -40px -80px 40px -80px;
    }
    .nav-item {
        padding: 15px 25px;
        font-weight: 500;
        font-size: 14px;
        color: var(--gray);
        border-bottom: 3px solid transparent;
    }
    .nav-item.active {
        border-bottom: 3px solid var(--primary-red);
        color: var(--primary-red);
    }
    .module-header {
        background: linear-gradient(90deg, var(--primary-red), var(--dark-red));
        padding: 30px;
        border-radius: 8px;
        color: white;
        display: flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 40px 0;
    }
    .module-icon {
        width: 60px; height: 60px;
        background: rgba(255,255,255,0.2);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 32px;
    }
    .fixed-restart {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 999;
        width: 340px;
    }
    .stButton>button {
        background: var(--primary-red) !important;
        color: white !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }
    .footer {
        text-align: center;
        padding: 40px;
        color: var(--gray);
        font-size: 12px;
        border-top: 1px solid var(--light-gray);
        margin-top: 80px;
    }
</style>
""", unsafe_allow_html=True)

# Header & Nav
st.markdown('<div class="header"><h1>DIA Proteomics Analysis Pipeline</h1><p>Module demonstrations and UI components</p></div>', unsafe_allow_html=True)
st.markdown("""
<div class="nav">
    <div class="nav-item active">Module 1: Data Import</div>
    <div class="nav-item">Module 2: Quality Control</div>
    <div class="nav-item">Module 3: Preprocessing</div>
    <div class="nav-item">Module 4: Analysis</div>
</div>
""", unsafe_allow_html=True)

# Module Header
st.markdown("""
<div class="module-header">
    <div class="module-icon">Upload</div>
    <div>
        <h2 style="margin:0;color:white;">Module 1: Data Import & Validation</h2>
        <p style="margin:5px 0 0;opacity:0.9;">Import mass spectrometry output matrices with automatic format detection and validation</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# RESTORE FROM SESSION
# ─────────────────────────────────────────────────────────────
if "prot_df" in st.session_state and not st.session_state.get("reconfig", False):
    df = st.session_state.prot_df
    c1 = st.session_state.c1_names
    c2 = st.session_state.c2_names
    sp_col = st.session_state.species_col
    sp_counts = st.session_state.sp_counts

    st.success("Data successfully restored from session")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.metric("**Condition A**", f"{len(c1)} replicates")
        st.write(" | ".join(c1))
    with col2:
        st.metric("**Condition B**", f"{len(c2)} replicates")
        st.write(" | ".join(c2))
    with col3:
        if st.button("Reconfigure", type="secondary"):
            st.session_state.reconfig = True
            st.rerun()

    st.info(f"**Species column detected**: `{sp_col}`")
    st.markdown("### Proteins per Species")
    st.dataframe(sp_counts, use_container_width=True, hide_index=True)
    st.bar_chart(sp_counts.set_index("Species")[["A", "B"]], use_container_width=True)

    st.markdown('<div class="fixed-restart">', unsafe_allow_html=True)
    if st.button("Restart Full Analysis", type="primary", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k.startswith(("prot_", "c1_", "c2_", "species_", "sp_", "reconfig")):
                del st.session_state[k]
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Allow re-upload on reconfigure
if st.session_state.get("reconfig", False):
    st.warning("Reconfiguring — please upload the same file again")

# ─────────────────────────────────────────────────────────────
# 1. UPLOAD
# ─────────────────────────────────────────────────────────────
st.markdown("### 1. Upload Your Protein Data")
uploaded = st.file_uploader(
    "Drag and drop your CSV file here",
    type=["csv", "tsv", "txt"],
    help="Supports Spectronaut, DIA-NN, MaxQuant, FragPipe"
)

if not uploaded:
    st.info("Upload a file to begin analysis")
    st.stop()

# Load data
@st.cache_data(show_spinner="Loading file...")
def load_file(f):
    content = f.getvalue().decode("utf-8", errors="replace")
    if content.startswith("\ufeff"): content = content[1:]
    return pd.read_csv(io.StringIO(content), sep=None, engine="python")

df_raw = load_file(uploaded)
st.success(f"Loaded {len(df_raw):,} proteins")

# Detect numeric (intensity) columns
intensity_cols = []
for col in df_raw.columns:
    if pd.to_numeric(df_raw[col].astype(str).str.replace(r"[,\#NUM!]", "", regex=True), errors='coerce').notna().mean() > 0.3:
        intensity_cols.append(col)

# ─────────────────────────────────────────────────────────────
# 2. ASSIGN REPLICATES
# ──────────────────────────────────────
st.markdown("### 2. Assign Replicates to Conditions")

rows = []
for col in intensity_cols:
    preview = df_raw[col].dropna().head(3).astype(str).tolist()
    rows.append({
        "Column": col,
        "Preview": " | ".join(preview),
        "Condition A": True,
        "Condition B": False,
    })

edited = st.data_editor(
    pd.DataFrame(rows),
    column_config={
        "Column": st.column_config.TextColumn(disabled=True),
        "Preview": st.column_config.TextColumn(disabled=True),
        "Condition A": st.column_config.CheckboxColumn("Condition A → A1,A2,A3..."),
        "Condition B": st.column_config.CheckboxColumn("Condition B → B1,B2,B3..."),
    },
    hide_index=True,
    use_container_width=True,
    num_rows="fixed"
)

cond_a = edited[edited["Condition A"]]["Column"].tolist()
cond_b = edited[edited["Condition B"]]["Column"].tolist()

if len(cond_a) == 0 or len(cond_b) == 0:
    st.error("Both conditions must have at least one replicate")
    st.stop()

# ONLY PROCEED IF EQUAL NUMBER OF REPLICATES
if len(cond_a) != len(cond_b):
    st.error(f"Condition A has {len(cond_a)} replicates, Condition B has {len(cond_b)}. "
             "Both must have the same number of replicates to proceed.")
    st.info("Adjust your selection so both conditions have the same number of replicates.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# 3. RENAME TO A1/A2... B1/B2...
# ─────────────────────────────────────────────────────────────
n = len(cond_a)
rename_map = {}
for i, col in enumerate(cond_a):
    rename_map[col] = f"A{i+1}"
for i, col in enumerate(cond_b):
    rename_map[col] = f"B{i+1}"

df = df_raw.rename(columns=rename_map).copy()
c1_names = [f"A{i+1}" for i in range(n)]
c2_names = [f"B{i+1}" for i in range(n)]

st.success(f"Renamed successfully → **A**: {', '.join(c1_names)} | **B**: {', '.join(c2_names)}")

# ─────────────────────────────────────────────────────────────
# 4. AUTO DETECT SPECIES COLUMN
# ─────────────────────────────────────────────────────────────
st.markdown("### 3. Detecting Species Column")

def find_species_col(df):
    keywords = ["HUMAN", "MOUSE", "RAT", "BOVIN", "YEAST", "RABIT", "CANFA", "PANTR", "MACMU","ECOLI"]
    pattern = "|".join(keywords)
    for col in df.columns:
        if col in c1_names + c2_names:
            continue
        if df[col].astype(str).str.upper().str.contains(pattern).any():
            return col
    return None

species_col = find_species_col(df)
if not species_col:
    st.error("No species column found. Looking for columns containing HUMAN, MOUSE, etc.")
    st.stop()

st.success(f"Species column detected: `{species_col}`")

# Extract clean species
def extract_sp(val):
    if pd.isna(val): return "Unknown"
    text = str(val).upper()
    known = ["HUMAN","MOUSE","RAT","BOVIN","YEAST","RABIT","CANFA","PANTR","MACMU","CHICK","PIG","ECOLI"]
    for s in known:
        if s in text:
            return s
    return "Other"

df["Species"] = df[species_col].apply(extract_sp)

# Count proteins
threshold = 1
counts = []
for sp in df["Species"].unique():
    if sp in ["Unknown", "Other"] and df["Species"].nunique() > 2:
        continue
    sub = df[df["Species"] == sp]
    in_a = (sub[c1_names] > threshold).any(axis=1).sum()
    in_b = (sub[c2_names] > threshold).any(axis=1).sum()
    counts.append({"Species": sp, "A": in_a, "B": in_b, "Total": len(sub)})

sp_counts = pd.DataFrame(counts).sort_values("Total", ascending=False)

# ─────────────────────────────────────────────────────────────
# SAVE TO SESSION
# ─────────────────────────────────────────────────────────────
st.session_state.update({
    "prot_df": df,
    "c1_names": c1_names,
    "c2_names": c2_names,
    "species_col": species_col,
    "sp_counts": sp_counts,
    "reconfig": False,
})

# ─────────────────────────────────────────────────────────────
# FINAL SUCCESS DISPLAY
# ─────────────────────────────────────────────────────────────
st.success("All processing complete! Data is ready.")

col1, col2 = st.columns(2)
with col1:
    st.metric("**Condition A**", ", ".join(c1_names))
with col2:
    st.metric("**Condition B**", ", ".join(c2_names))

st.markdown("### Proteins Detected per Species")
st.dataframe(sp_counts, use_container_width=True, hide_index=True)
st.bar_chart(sp_counts.set_index("Species")[["A", "B"]], use_container_width=True)

# Restart button
st.markdown('<div class="fixed-restart">', unsafe_allow_html=True)
if st.button("Restart Full Analysis", type="primary", use_container_width=True):
    for k in list(st.session_state.keys()):
        if k.startswith(("prot_", "c1_", "c2_", "species_", "sp_", "reconfig")):
            del st.session_state[k]
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <strong>Proprietary & Confidential | For Internal Use Only</strong>
    <p>© 2024 Thermo Fisher Scientific Inc. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
