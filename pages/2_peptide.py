# pages/1_Peptide_Import.py
import streamlit as st
import pandas as pd
import re
import io
from shared import restart_button

# ====================== SAFE SESSION STATE HELPER ======================
def ss(key, default=None):
    """Safe access to st.session_state – never raises KeyError"""
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

# ====================== BRANDING & LAYOUT ======================
st.set_page_config(page_title="Peptide Import", layout="wide")
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

st.markdown('<div class="header"><h1>DIA Proteomics Pipeline</h1><p>Module — Peptide-Level Import</p></div>', unsafe_allow_html=True)
st.markdown('<div class="nav"><div class="nav-item active">Peptide Import</div><div class="nav-item">Protein Import</div><div class="nav-item">Analysis</div></div>', unsafe_allow_html=True)
st.markdown('<div class="module-header"><div class="module-icon">Peptide</div><div><h2 style="margin:0;color:white;">Peptide Data Import</h2><p style="margin:5px 0 0;opacity:0.9;">Auto-detect species • Equal replicates • Set peptide sequence as index</p></div></div>', unsafe_allow_html=True)

# ====================== RESTORE FROM CACHE ======================
if ss("pept_df") is not None and not ss("reconfig_pept", False):
    df = ss("pept_df")
    c1 = ss("pept_c1")
    c2 = ss("pept_c2")
    peptide_col = ss("pept_peptide_col")
    sp_col = ss("pept_sp_col")
    sp_counts = ss("pept_sp_counts")

    st.success("Peptide data restored from cache")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.metric("**Condition A**", f"{len(c1)} replicates")
        st.write(" | ".join(c1))
    with col2:
        st.metric("**Condition B**", f"{len(c2)} replicates")
        st.write(" | ".join(c2))
    with col3:
        if st.button("Reconfigure", type="secondary"):
            ss("reconfig_pept", True)
            st.rerun()

    st.info(f"**Peptide Sequence (index)**: `{peptide_col}` • **Species column**: `{sp_col}`")
    st.markdown("### Peptides Detected per Species")
    st.dataframe(sp_counts, use_container_width=True, hide_index=True)
    st.bar_chart(sp_counts.set_index("Species")[["A", "B"]], use_container_width=True)

    restart_button()
    st.stop()

# Reconfigure mode
if ss("reconfig_pept", False):
    st.warning("Reconfiguring peptide data — please re-upload the same file")

# ====================== 1. UPLOAD FILE ======================
st.markdown("### 1. Upload Peptide-Level File")
uploaded = st.file_uploader("CSV/TSV/TXT from Spectronaut, DIA-NN, etc.", type=["csv","tsv","txt"], key="pept_upload")

if not uploaded:
    st.info("Please upload your peptide quantification file to continue")
    restart_button()
    st.stop()

@st.cache_data(show_spinner="Loading and parsing file...")
def load_file(_file):
    content = _file.getvalue().decode("utf-8", errors="replace")
    if content.startswith("\ufeff"):
        content = content[1:]
    return pd.read_csv(io.StringIO(content), sep=None, engine="python")

df_raw = load_file(uploaded)
st.success(f"Successfully loaded {len(df_raw):,} peptide entries")

# ====================== DETECT INTENSITY COLUMNS ======================
intensity_cols = []
for col in df_raw.columns:
    cleaned = pd.to_numeric(df_raw[col].astype(str).str.replace(r"[,\#NUM!]", "", regex=True), errors='coerce')
    if cleaned.notna().mean() > 0.3:
        df_raw[col] = cleaned
        intensity_cols.append(col)

if not intensity_cols:
    st.error("No quantitative (intensity) columns found. Check file format.")
    st.stop()

# ====================== 2. ASSIGN REPLICATES (EQUAL COUNT) ======================
st.markdown("### 2. Assign Replicates to Conditions")
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
        "Condition B → B1,B2...": st.column_config.CheckboxColumn("Condition B"),
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
    st.error(f"Condition A has {len(a_cols)} reps, Condition B has {len(b_cols)} → Must be equal!")
    st.stop()

# Rename replicates
n = len(a_cols)
rename_map = {a_cols[i]: f"A{i+1}" for i in range(n)}
rename_map.update({b_cols[i]: f"B{i+1}" for i in range(n)})
df = df_raw.rename(columns=rename_map).copy()
c1 = [f"A{i+1}" for i in range(n)]
c2 = [f"B{i+1}" for i in range(n)]

st.success(f"Replicates renamed → **A**: {', '.join(c1)} | **B**: {', '.join(c2)}")

# ====================== 3. SELECT PEPTIDE SEQUENCE COLUMN & SET INDEX ======================
st.markdown("### 3. Select Peptide Sequence Column")
peptide_candidates = [c for c in df.columns if any(kw in c.lower() for kw in ["sequence", "peptide", "seq", "stripped", "precursor"])]
peptide_col = st.selectbox("Which column contains the peptide sequence?", peptide_candidates, index=0)

if st.button("Set Peptide Sequence as Index", type="primary"):
    if peptide_col in df.columns:
        df = df.set_index(peptide_col)
    st.success(f"Index successfully set to peptide sequence `{peptide_col}`")
    st.rerun()  # refresh to show new index

# ====================== 4. AUTO-DETECT SPECIES COLUMN ======================
st.markdown("### 4. Auto-Detect Species Column")
species_list = ["HUMAN","MOUSE","RAT","ECOLI","BOVIN","YEAST","RABIT","CANFA","MACMU","PANTR","CHICK"]

def find_species_column(df):
    pattern = "|".join(species_list)
    for col in df.columns:
        if col in c1 + c2:
            continue
        if df[col].astype(str).str.upper().str.contains(pattern).any():
            return col
    return None

sp_col = find_species_column(df) or "Not found"

if sp_col != "Not found":
    def extract_species(val):
        val = str(val).upper()
        for s in species_list:
            if s in val:
                return s
        return "Other"
    df["Species"] = df[sp_col].apply(extract_species)

    # Count per species
    counts = []
    for sp in df["Species"].dropna().unique():
        if sp == "Other" and len(df["Species"].unique()) > 2:
            continue
        sub = df[df["Species"] == sp]
        counts.append({
            "Species": sp,
            "A": (sub[c1] > 1).any(axis=1).sum(),
            "B": (sub[c2] > 1).any(axis=1).sum(),
            "Total": len(sub)
        })
    sp_counts = pd.DataFrame(counts).sort_values("Total", ascending=False)
else:
    sp_counts = pd.DataFrame([{"Species": "All data", "A": 0, "B": 0, "Total": len(df)}])

# ====================== SAVE EVERYTHING TO CACHE ======================
st.success("Peptide processing complete! Data cached and ready for analysis.")

ss("pept_df", df)
ss("pept_c1", c1)
ss("pept_c2", c2)
ss("pept_peptide_col", peptide_col)
ss("pept_sp_col", sp_col)
ss("pept_sp_counts", sp_counts)
ss("reconfig_pept", False)

# ====================== FINAL DISPLAY ======================
col1, col2 = st.columns(2)
with col1:
    st.metric("**Condition A**", ", ".join(c1))
with col2:
    st.metric("**Condition B**", ", ".join(c2))

if sp_col != "Not found":
    st.markdown("### Peptides Detected per Species")
    st.dataframe(sp_counts, use_container_width=True, hide_index=True)
    st.bar_chart(sp_counts.set_index("Species")[["A", "B"]], use_container_width=True)

# ====================== RESTART BUTTON & FOOTER ======================
restart_button()

st.markdown("""
<div class="footer">
    <strong>Proprietary & Confidential | For Internal Use Only</strong><br>
    © 2024 Thermo Fisher Scientific Inc. All rights reserved.
</div>
""", unsafe_allow_html=True)
