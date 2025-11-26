# pages/1_Protein_Import.py
import streamlit as st
import pandas as pd
import re
import io
from shared import restart_button

# ====================== SAFE SESSION STATE ======================
def ss(key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

st.set_page_config(page_title="Protein Import", layout="wide")

# ====================== BRANDING ======================
st.markdown("""
<style>
    :root {--red:#E71316; --darkred:#A6192E; --gray:#54585A; --light:#E2E3E4;}
    .header {background:linear-gradient(90deg,var(--red),var(--darkred)); padding:20px 40px; color:white; margin:-80px -80px 40px -80px;}
    .header h1,.header p {margin:0;}
    .nav {background:white; border-bottom:2px solid var(--light); padding:0 40px; display:flex; gap:5px; margin:-40px -80px 40px -80px;}
    .nav-item {padding:15px 25px; font-weight:500; color:var(--gray); border-bottom:3px solid transparent;}
    .nav-item.active {border-bottom:3px solid var(--red); color:var(--red);}
    .module-header {background:linear-gradient(90deg,var(--red),var(--darkred)); padding:30px; border-radius:8px; color:white; display:flex; align-items:center; gap:20px;}
    .module-icon {width:60px; height:60px; background:rgba(255,255,255,0.2); border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:32px;}
    .fixed-restart {position:fixed; bottom:20px; left:50%; transform:translateX(-50%); z-index:999; width:340px;}
    .stButton>button {background:var(--red)!important; color:white!important; border-radius:6px!important;}
    .footer {text-align:center; padding:40px; color:var(--gray); font-size:12px; border-top:1px solid var(--light); margin-top:80px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>DIA Proteomics Pipeline</h1><p>Module — Protein-Level Import</p></div>', unsafe_allow_html=True)
st.markdown('<div class="nav"><div class="nav-item">Peptide Import</div><div class="nav-item active">Protein Import</div><div class="nav-item">Analysis</div></div>', unsafe_allow_html=True)
st.markdown('<div class="module-header"><div class="module-icon">Protein</div><div><h2 style="margin:0;color:white;">Protein Data Import</h2><p style="margin:5px 0 0;opacity:0.9;">Auto-detect species • Equal replicates • Set Protein Group as index</p></div></div>', unsafe_allow_html=True)

# ====================== RESTORE FROM CACHE ======================
if ss("prot_df") is not None and not ss("reconfig_prot", False):
    df = ss("prot_df")
    c1 = ss("prot_c1")
    c2 = ss("prot_c2")
    pg_col = ss("prot_pg_col")
    sp_col = ss("prot_sp_col")
    sp_counts = ss("prot_sp_counts")

    st.success("Protein data restored from cache")
    col1, col2, col3 = st.columns([2,2,1])
    with col1:
        st.metric("**Condition A**", f"{len(c1)} reps"); st.write(" | ".join(c1))
    with col2:
        st.metric("**Condition B**", f"{len(c2)} reps"); st.write(" | ".join(c2))
    with col3:
        if st.button("Reconfigure", type="secondary"):
            ss("reconfig_prot", True)
            st.rerun()

    st.info(f"**Protein Group (index)**: `{pg_col}` • **Species**: `{sp_col}`")
    st.markdown("### Proteins per Species")
    st.dataframe(sp_counts, use_container_width=True, hide_index=True)
    st.bar_chart(sp_counts.set_index("Species")[["A", "B"]])
    restart_button()
    st.stop()

if ss("reconfig_prot", False):
    st.warning("Reconfiguring — please upload the same file again")

# ====================== 1. UPLOAD FILE ======================
st.markdown("### 1. Upload Protein-Level File")
uploaded = st.file_uploader(
    "CSV/TSV/TXT from Spectronaut, DIA-NN, MaxQuant, FragPipe",
    type=["csv", "tsv", "txt"],
    key="prot_upload"
)

if not uploaded:
    st.info("Please upload a protein quantification file to begin")
    restart_button()
    st.stop()

# ====================== SAFE FILE LOADING ======================
@st.cache_data(show_spinner="Loading file securely...")
def load_file_safely(file):
    try:
    try:
        content = file.getvalue().decode("utf-8", errors="replace")
        if content.startswith("\ufeff"):
            content = content[1:]
        if not content.strip():
            st.error("File is empty!")
            return None
        df = pd.read_csv(io.StringIO(content), sep=None, engine="python", on_bad_lines='skip')
        if df.empty:
            st.error("File loaded but contains no data.")
            return None
        return df
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None

df_raw = load_file_safely(uploaded)

# ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
# CRITICAL: Stop if loading failed
if df_raw is None or df_raw.empty:
    st.error("Cannot proceed — file loading failed or data is empty.")
    restart_button()
    st.stop()
# ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←

st.success(f"Successfully loaded {len(df_raw):,} protein entries")

# ====================== DETECT INTENSITY COLUMNS (SAFE) ======================
intensity_cols = []
for col in df_raw.columns:
    try:
        # Clean and convert
        cleaned = pd.to_numeric(
            df_raw[col].astype(str).str.replace(r"[,\#NUM!]", "", regex=True),
            errors='coerce'
        )
        if cleaned.notna().mean() > 0.3:  # at least 30% real numbers
            df_raw[col] = cleaned
            intensity_cols.append(col)
    except:
        continue  # skip problematic columns

if not intensity_cols:
    st.error("No quantitative intensity columns detected. Is this really a protein quantification file?")
    st.stop()

st.info(f"Detected {len(intensity_cols)} intensity/replicate columns")

# ====================== 2. ASSIGN REPLICATES ======================
st.markdown("### 2. Assign Replicates (must have equal numbers)")
rows = []
for col in intensity_cols:
    preview = df_raw[col].dropna().head(3).astype(str).tolist()
    rows.append({
        "Column": col,
        "Preview": " | ".join(preview),
        "Condition A → A1,A2...": True,
        "Condition B → B1,B2...": False
    })

edited = st.data_editor(
    pd.DataFrame(rows),
    column_config={
        "Column": st.column_config.TextColumn(disabled=True),
        "Preview": st.column_config.TextColumn(disabled=True),
        "Condition A → A1,A2...": st.column_config.CheckboxColumn("Condition A"),
        "Condition B → B1,B2": st.column_config.CheckboxColumn("Condition B"),
    },
    hide_index=True,
    use_container_width=True,
    num_rows="fixed"
)

a_cols = edited[edited["Condition A → A1,A2..."]]["Column"].tolist()
b_cols = edited[edited["Condition B → B1,B2..."]]["Column"].tolist()

if len(a_cols) == 0 or len(b_cols) == 0:
    st.error("Both conditions must have at least one replicate")
    st.stop()

if len(a_cols) != len(b_cols):
    st.error(f"Must have equal replicates: A={len(a_cols)}, B={len(b_cols)}")
    st.stop()

# Rename
n = len(a_cols)
rename_map = {a_cols[i]: f"A{i+1}" for i in range(n)}
rename_map.update({b_cols[i]: f"B{i+1}" for i in range(n)})
df = df_raw.rename(columns=rename_map).copy()
c1 = [f"A{i+1}" for i in range(n)]
c2 = [f"B{i+1}" for i in range(n)]

st.success(f"Replicates renamed → **A**: {', '.join(c1)} | **B**: {', '.join(c2)}")

# ====================== 3. PROTEIN GROUP COLUMN ======================
st.markdown("### 3. Select Protein Group Column (will be index)")
pg_candidates = [c for c in df.columns if any(kw in c.lower() for kw in ["protein.group","pg","leading","accession","protein ids"])]
pg_col = st.selectbox("Protein Group ID column", pg_candidates or df.columns.tolist())

if st.button("Set Protein Group as Index", type="primary"):
    if pg_col in df.columns:
        df = df.set_index(pg_col)
        st.success(f"Index set to `{pg_col}`")
        st.rerun()

# ====================== 4. SPECIES DETECTION ======================
st.markdown("### 4. Auto-Detect Species")
species_list = ["HUMAN","MOUSE","RAT","ECOLI","BOVIN","YEAST","RABIT","CANFA","MACMU","PANTR"]

def find_species_col(df):
    pattern = "|".join(species_list)
    for c in df.columns:
        if c in c1 + c2: continue
        if df[c].astype(str).str.upper().str.contains(pattern).any():
            return c
    return None

sp_col = find_species_col(df) or "Not found"

if sp_col != "Not found":
    df["Species"] = df[sp_col].astype(str).str.upper().apply(lambda x: next((s for s in species_list if s in x), "Other"))
    counts = []
    for sp in df["Species"].unique():
        if sp == "Other" and df["Species"].nunique() > 2: continue
        sub = df[df["Species"] == sp]
        counts.append({"Species": sp, "A": (sub[c1]>1).any(axis=1).sum(), "B": (sub[c2]>1).any(axis=1).sum(), "Total": len(sub)})
    sp_counts = pd.DataFrame(counts).sort_values("Total", ascending=False)
else:
    sp_counts = pd.DataFrame([{"Species":"All","A":0,"B":0,"Total":len(df)}])

# ====================== SAVE TO CACHE ======================
st.success("Protein data fully processed and cached!")

ss("prot_df", df)
ss("prot_c1", c1)
ss("prot_c2", c2)
ss("prot_pg_col", pg_col)
ss("prot_sp_col", sp_col)
ss("prot_sp_counts", sp_counts)
ss("reconfig_prot", False)

# ====================== FINAL DISPLAY ======================
col1, col2 = st.columns(2)
with col1: st.metric("Condition A", ", ".join(c1))
with col2: st.metric("Condition B", ", ".join(c2))

if sp_col != "Not found":
    st.markdown("### Proteins Detected per Species")
    st.dataframe(sp_counts, use_container_width=True, hide_index=True)
    st.bar_chart(sp_counts.set_index("Species")[["A", "B"]])

restart_button()

st.markdown("""
<div class="footer">
    <strong>Proprietary & Confidential | For Internal Use Only</strong><br>
    © 2024 Thermo Fisher Scientific Inc. All rights reserved.
</div>
""", unsafe_allow_html=True)
