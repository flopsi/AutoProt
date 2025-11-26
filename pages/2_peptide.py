# pages/1_Peptide_Import.py
import streamlit as st
import pandas as pd
import re
import io
from shared import restart_button
# ────────────────────── SAFE SESSION STATE HELPER ──────────────────────
def ss(key, default=None):
    """Safe session_state getter/setter – never raises KeyError"""
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]
st.set_page_config(page_title = "Peptide Import", layout="wide")

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

st.markdown('<div class="header"><h1>DIA Proteomics Pipeline</h1><p>Module 1 — Peptide-Level Import</p></div>', unsafe_allow_html=True)
st.markdown('<div class="nav"><div class="nav-item active">Peptide Import</div><div class="nav-item">Peptide Import</div><div class="nav-item">Analysis</div></div>', unsafe_allow_html=True)
st.markdown('<div class="module-header"><div class="module-icon">Peptide</div><div><h2 style="margin:0;color:white;">Peptide Data Import</h2><p style="margin:5px 0 0;opacity:0.9;">Auto-detect species • Equal replicates • Set Peptide Group as index</p></div></div>', unsafe_allow_html=True)

# ====================== RESTORE FROM CACHE ======================
if "prot_df" in st.session_state and not st.session_state.get("reconfig_prot", False):
    df = st.session_state.prot_df
    c1 = st.session_state.prot_c1
    c2 = st.session_state.prot_c2
    peptide_columns = st.session_state.peptide_columns
    sp_col = st.session_state.prot_sp_col
    sp_counts = st.session_state.prot_sp_counts

    st.success("Peptide data loaded from cache")
    col1, col2, col3 = st.columns([2,2,1])
    with col1: st.metric("Condition A", f"{len(c1)} reps", help="A1,A2,..."); st.write(" | ".join(c1))
    with col2: st.metric("Condition B", f"{len(c2)} reps"); st.write(" | ".join(c2))
    with col3:
        if st.button("Reconfigure"): st.session_state.reconfig_prot = True; st.rerun()

    st.info(f"**Peptide Group Column (index)**: `{peptide_columns}` • **Species**: `{sp_col}`")
    st.markdown("### Peptides per Species")
    st.dataframe(sp_counts, use_container_width=True, hide_index=True)
    st.bar_chart(sp_counts.set_index("Species")[["A","B"]])
    restart_button()
    st.stop()

if st.session_state.get("reconfig_prot", False):
    st.warning("Reconfiguring — re-upload the same file")

# ====================== UPLOAD & CACHE ======================
st.markdown("### 1. Upload Peptide-Level File")
uploaded = st.file_uploader("CSV/TSV/TXT", type=["csv","tsv","txt"], key="pep_upload")

if not uploaded:
    st.info("Upload your peptide quantification file")
    st.stop()

@st.cache_data(show_spinner="Parsing file...")
def load_and_parse(_file):
    s = _file.getvalue().decode("utf-8", errors="replace")
    if s.startswith("\ufeff"): s = s[1:]
    df = pd.read_csv(io.StringIO(s), sep=None, engine="python")
    return df

df_raw = load_and_parse(uploaded)
st.success(f"Loaded {len(df_raw):,} Peptide groups")

# Detect intensity columns
intensity_cols = []
for c in df_raw.columns:
    cleaned = pd.to_numeric(df_raw[c].astype(str).str.replace(r"[,\#NUM!]", "", regex=True), errors='coerce')
    if cleaned.notna().mean() > 0.3:
        df_raw[c] = cleaned
        intensity_cols.append(c)

# ====================== REPLICATE ASSIGNMENT ======================
st.markdown("### 2. Assign Replicates (must have equal count)")
rows = [{"Column": c, "Preview": " | ".join(map(str, df_raw[c].dropna().head(3))), "A": True, "B": False} for c in intensity_cols]
edited = st.data_editor(pd.DataFrame(rows), column_config={
    "Column": st.column_config.TextColumn(disabled=True),
    "Preview": st.column_config.TextColumn(disabled=True),
    "A": st.column_config.CheckboxColumn("Condition A → A1,A2,..."),
    "B": st.column_config.CheckboxColumn("Condition B → B1,B2,...")
}, hide_index=True, use_container_width=True, num_rows="fixed")

a_cols = edited[edited["A"]]["Column"].tolist()
b_cols = edited[edited["B"]]["Column"].tolist()

if len(a_cols) != len(b_cols):
    st.error(f"A: {len(a_cols)} ≠ B: {len(b_cols)} → Must be equal!")
    st.stop()

n = len(a_cols)
rename_map = {old: f"A{i+1}" for i, old in enumerate(a_cols)}
rename_map.update({old: f"B{i+1}" for i, old in enumerate(b_cols)})
df = df_raw.rename(columns=rename_map)
c1, c2 = [f"A{i+1}" for i in range(n)], [f"B{i+1}" for i in range(n)]

st.success(f"Renamed → A: {', '.join(c1)} | B: {', '.join(c2)}")

# ====================== Peptide GROUP COLUMN + INDEX ======================
st.markdown("### 3. Select Peptide Group Column (will become index)")
peptide_candidates = [c for c in df.columns if any(k in c.lower() for k in ["sequence","peptide","seq","stripped","Precursor"])]
peptide_col = st.selectbox("Peptide Sequence Column", peptide_candidates)
if st.button("Set as Index"):
    df = df.set_index(peptide_columns)
    st.success(f"Index set to `{peptide_columns}`")

# ====================== SPECIES DETECTION (incl. ECOLI) ======================
st.markdown("### 4. Auto-Detect Species Column")
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

# ====================== SAVE TO GLOBAL CACHE ======================
st.session_state.update({
    "prot_df": df,
    "prot_c1": c1,
    "prot_c2": c2,
    "prot_peptide_columns": peptide_columns,
    "prot_sp_col": sp_col,
    "prot_sp_counts": sp_counts,
    "reconfig_prot": False,
})

st.success("Peptide data cached and ready for downstream modules!")
st.json({k: type(v).__name__ for k, v in st.session_state.items() if k.startswith("prot_")}, expanded=False)

restart_button()
st.markdown('<div class="footer"><strong>Proprietary & Confidential</strong><br>© 2024 Thermo Fisher Scientific</div>', unsafe_allow_html=True)
