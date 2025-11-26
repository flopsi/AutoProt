# pages/1_Protein_Import.py
import streamlit as st
import pandas as pd
import re
import io

st.set_page_config(page_title="Protein Import", layout="wide")

# ====================== BRANDING & LAYOUT ======================
st.markdown("""
<style>
    :root {
        --red: #E71316;
        --darkred: #A6192E;
        --gray: #54585A;
        --light: #E2E3E4;
    }
    .header {background: linear-gradient(90deg, var(--red), var(--darkred)); padding: 20px 40px; color: white; margin: -80px -80px 40px -80px;}
    .header h1 {margin:0; font-size:28px; font-weight:600;}
    .header p {margin:5px 0 0; font-size:14px; opacity:0.95;}
    .nav {background: white; border-bottom: 2px solid var(--light); padding: 0 40px; display: flex; gap: 5px; margin: -40px -80px 40px -80px;}
    .nav-item {padding: 15px 25px; font-weight: 500; color: var(--gray); border-bottom: 3px solid transparent;}
    .nav-item.active {border-bottom: 3px solid var(--red); color: var(--red);}
    .module-header {background: linear-gradient(90deg, var(--red), var(--darkred)); padding: 30px; border-radius: 8px; color: white; display: flex; align-items: center; gap: 20px;}
    .module-icon {width: 60px; height: 60px; background: rgba(255,255,255,0.2); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 32px;}
    .fixed-restart {position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); z-index: 999; width: 340px;}
    .stButton>button {background: var(--red) !important; color: white !important; border-radius: 6px !important; font-weight: 500 !important;}
    .footer {text-align: center; padding: 40px; color: var(--gray); font-size: 12px; border-top: 1px solid var(--light); margin-top: 80px;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header"><h1>DIA Proteomics Analysis Pipeline</h1><p>Module 1 — Protein-Level Import</p></div>', unsafe_allow_html=True)
st.markdown("""
<div class="nav">
    <div class="nav-item active">Module 1: Protein Import</div>
    <div class="nav-item">Module 2: Peptide Import</div>
    <div class="nav-item">Module 3: Analysis</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="module-header">
    <div class="module-icon">Protein</div>
    <div>
        <h2 style="margin:0;color:white;">Protein-Level Data Import</h2>
        <p style="margin:5px 0 0;opacity:0.9;">Upload protein matrix • Auto-detect species & protein groups</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ====================== SESSION RESTORE ======================
if "prot_df" in st.session_state and not st.session_state.get("reconfig_prot", False):
    df = st.session_state.prot_df
    c1 = st.session_state.c1_names
    c2 = st.session_state.c2_names
    sp_col = st.session_state.species_col
    pg_col = st.session_state.pg_col
    sp_counts = st.session_state.sp_counts

    st.success("Protein data restored successfully")

    col1, col2, col3 = st.columns([2,2, 1])
    with col1:
        st.metric("**Condition A**", f"{len(c1)} reps", help="A1, A2, ...")
        st.write(" | ".join(c1))
    with col2:
        st.metric("**Condition B**", f"{len(c2)} reps")
        st.write(" | ".join(c2))
    with col3:
        if st.button("Reconfigure", type="secondary"):
            st.session_state.reconfig_prot = True
            st.rerun()

    st.info(f"**Species column**: `{sp_col}` • **Protein Group column**: `{pg_col}`")
    st.markdown("### Proteins per Species")
    st.dataframe(sp_counts, use_container_width=True, hide_index=True)
    st.bar_chart(sp_counts.set_index("Species")[["A", "B"]], use_container_width=True)

    st.markdown('<div class="fixed-restart">', unsafe_allow_html=True)
    if st.button("Restart Analysis", type="primary", use_container_width=True):
        for k in [k for k in st.session_state.keys() if k.startswith(("prot_", "c1_", "c2_", "reconfig_"))]:
            del st.session_state[k]
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

if st.session_state.get("reconfig_prot", False):
    st.warning("Reconfiguring protein data — please re-upload file")

# ====================== 1. UPLOAD ======================
st.markdown("### 1. Upload Protein Data")
uploaded = st.file_uploader("CSV/TSV/TXT from Spectronaut, DIA-NN, MaxQuant, FragPipe", type=["csv", "tsv", "txt"], key="prot_upload")

if not uploaded:
    st.info("Upload a protein-level quantification file to begin")
    st.stop()

@st.cache_data
def load_df(f):
    s = f.getvalue().decode("utf-8", errors="replace")
    if s.startswith("\ufeff"): s = s[1:]
    return pd.read_csv(io.StringIO(s), sep=None, engine="python")

df = load_df(uploaded)
st.success(f"Loaded {len(df):,} protein groups")

# Detect intensity columns
intensity_cols = []
for c in df.columns:
    if df[c].dtype == "object":
        numeric = pd.to_numeric(df[c].astype(str).str.replace(r"[,\#NUM!]", "", regex=True), errors='coerce')
        if numeric.notna().mean() > 0.3:
            df[c] = numeric
            intensity_cols.append(c)

# ====================== 2. ASSIGN REPLICATES ======================
st.markdown("### 2. Assign Replicates to Conditions (must be equal count)")

rows = []
for c in intensity_cols:
    preview = df[c].dropna().head(3).tolist()
    rows.append({"Column": c, "Preview": " | ".join(map(str, preview)), "A": True, "B": False})

edited = st.data_editor(
    pd.DataFrame(rows),
    column_config={
        "Column": st.column_config.TextColumn(disabled=True),
        "Preview": st.column_config.TextColumn(disabled=True),
        "A": st.column_config.CheckboxColumn("Condition A → A1,A2,..."),
        "B": st.column_config.CheckboxColumn("Condition B → B1,B2,..."),
    },
    hide_index=True,
    use_container_width=True,
    num_rows="fixed"
)

a_cols = edited[edited["A"]]["Column"].tolist()
b_cols = edited[edited["B"]]["Column"].tolist()

if len(a_cols) == 0 or len(b_cols) == 0:
    st.error("Both conditions need at least one replicate")
    st.stop()

if len(a_cols) != len(b_cols):
    st.error(f"Condition A: {len(a_cols)} vs Condition B: {len(b_cols)} — must be equal!")
    st.info("Adjust selection so both sides have the same number of replicates")
    st.stop()

# Rename
n = len(a_cols)
rename_map = {old: f"A{i+1}" for i, old in enumerate(a_cols)}
rename_map.update({old: f"B{i+1}" for i, old in enumerate(b_cols)})
df = df.rename(columns=rename_map)
c1_names = [f"A{i+1}" for i in range(n)]
c2_names = [f"B{i+1}" for i in range(n)]

st.success(f"Renamed → **A**: {', '.join(c1_names)} | **B**: {', '.join(c2_names)}")

# ====================== 3. AUTO DETECT PROTEIN GROUP COLUMN ======================
st.markdown("### 3. Select Protein Group Column (ID)")

pg_candidates = [c for c in df.columns if c.lower() in ["protein.group", "pg", "proteingroup", "leading razor protein", "protein ids", "protein", "accession"]]
if not pg_candidates:
    pg_candidates = [c for c in df.columns if "protein" in c.lower() or "pg" in c.lower()]

pg_col = st.selectbox("Which column contains the Protein Group ID?", pg_candidates, index=0)

if st.button("Set Protein Group as Index"):
    if pg_col in df.columns:
        df = df.set_index(pg_col)
        st.success(f"Index set to `{pg_col}`")
    else:
        st.error("Column not found")

# ====================== 4. AUTO DETECT SPECIES COLUMN ======================
st.markdown("### 4. Auto-Detect Species Column")

species_list = ["HUMAN", "MOUSE", "RAT", "ECOLI", "BOVIN", "YEAST", "RABIT", "CANFA", "MACMU", "PANTR", "CHICK"]

def find_species_col(df):
    pattern = "|".join(species_list)
    for c in df.columns:
        if c in c1_names + c2_names:
            continue
        if df[c].astype(str).str.upper().str.contains(pattern, regex=True).any():
            return c
    return None

species_col = find_species_col(df)

if species_col:
    st.success(f"Species column auto-detected: `{species_col}`")
else:
    st.warning("No species column found. Will skip species breakdown.")
    species_col = None

if species_col:
    def extract_sp(x):
        if pd.isna(x): return "Unknown"
        t = str(x).upper()
        for s in species_list:
            if s in t:
                return s
        return "Other"
    df["Species"] = df[species_col].apply(extract_sp)

    # Count
    counts = []
    for sp in df["Species"].unique():
        if sp in ["Unknown", "Other"] and df["Species"].nunique() > 2: continue
        sub = df[df["Species"] == sp]
        a_count = (sub[c1_names] > 1).any(axis=1).sum()
        b_count = (sub[c2_names] > 1).any(axis=1).sum()
        counts.append({"Species": sp, "A": a_count, "B": b_count, "Total": len(sub)})
    sp_counts = pd.DataFrame(counts).sort_values("Total", ascending=False)
else:
    sp_counts = pd.DataFrame([{"Species": "N/A", "A": 0, "B": 0, "Total": len(df)}])

# ====================== SAVE TO SESSION ======================
st.session_state.update({
    "prot_df": df,
    "c1_names": c1_names,
    "c2_names": c2_names,
    "species_col": species_col or "Not found",
    "pg_col": pg_col,
    "sp_counts": sp_counts,
    "reconfig_prot": False,
})

# ====================== FINAL DISPLAY ======================
st.success("Protein data fully processed and ready!")
col1, col2 = st.columns(2)
with col1: st.metric("Condition A", ", ".join(c1_names))
with col2: st.metric("Condition B", ", ".join(c2_names))

if species_col:
    st.markdown("### Proteins per Species")
    st.dataframe(sp_counts, use_container_width=True, hide_index=True)
    st.bar_chart(sp_counts.set_index("Species")[["A", "B"]], use_container_width=True)

st.markdown('<div class="fixed-restart">', unsafe_allow_html=True)
if st.button("Restart Full Analysis", type="primary", use_container_width=True):
    for k in [k for k in st.session_state.keys() if k.startswith(("prot_", "c1_", "c2_", "reconfig_"))]:
        del st.session_state[k]
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <strong>Proprietary & Confidential | For Internal Use Only</strong>
    <p>© 2024 Thermo Fisher Scientific Inc. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
