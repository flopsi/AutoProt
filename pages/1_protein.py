# pages/1_Protein_Import.py
import streamlit as st
import pandas as pd
import io
from datetime import datetime
from shared import restart_button

# ====================== UNIVERSAL DEBUG LOGGER (ALWAYS VISIBLE) ======================
def log_container = st.empty()  # We'll update this at the very end

def log(msg, data=None):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"<small style='color:#666; font-family:monospace;'>[{timestamp}] {msg}</small>"
    if data is not None:
        with st.expander(f"Details → {msg}", expanded=False):
            st.code(data, language="text")
        return f"{line} (click to expand)"
    return line

# Turn on/off debugging
DEBUG = True  # Set to False before final production if desired

def debug(msg, data=None):
    if DEBUG:
        st.session_state.setdefault("debug_log", []).append(log(msg, data))

# Start logging
st.session_state.setdefault("debug_log", [])
debug("Protein Import page loaded")

# ====================== SAFE SESSION STATE ======================
def ss(key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
        debug(f"Initialized session_state['{key}']", default)
    return st.session_state[key]

# ====================== PAGE SETUP & BRANDING ======================
st.set_page_config(page_title="Protein Import", layout="wide")

st.markdown("""
<style>
    :root {--red:#E71316; --darkred:#A6192E; --gray:#54585A; --light:#E2E3E4;}
    .header {background:linear-gradient(90deg,var(--red),var(--darkred)); padding:20px 40px; color:white; margin:-80px -80px 40px -80px;}
    .header h1,.header p {margin:0;}
    .nav {background:white; border-bottom:2px solid var(--light); padding:0 40px; display:flex; gap:5px; margin:-40px -80px 40px -80px;}
    .nav-item {padding:15px 25px; font-weight:500; color:var(--gray); border-bottom:3px solid transparent;}
    .nav-item.active {border-bottom:3px solid var(--red); color:var(--red);}
    .module-header {background:linear-gradient(90deg,var(--red),var(--darkred)); padding:30px; border-radius:8px; color:white; display:flex; align-items:center; gap:20px;}
    .module-icon {width:60px;height:60px;background:rgba(255,255,255,0.2);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:32px;}
    .fixed-restart {position:fixed; bottom:20px; left:50%; transform:translateX(-50%); z-index:9999; width:340px;}
    .stButton>button {background:var(--red)!important; color:white!important; border-radius:6px!important;}
    .footer {text-align:center; padding:40px; color:var(--gray); font-size:12px; border-top:1px solid var(--light); margin-top:80px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>DIA Proteomics Pipeline</h1><p>Module — Protein-Level Import</p></div>', unsafe_allow_html=True)
st.markdown('<div class="nav"><div class="nav-item">Peptide Import</div><div class="nav-item active">Protein Import</div><div class="nav-item">Analysis</div></div>', unsafe_allow_html=True)
st.markdown('<div class="module-header"><div class="module-icon">Protein</div><div><h2 style="margin:0;color:white;">Protein Data Import</h2><p style="margin:5px 0 0;opacity:0.9;">Auto-detect species • Equal replicates • Set Protein Group as index</p></div></div>', unsafe_allow_html=True)

# ====================== RESTORE FROM CACHE ======================
debug("Checking for cached protein data...")
if ss("prot_df") is not None and not ss("reconfig_prot", False):
    debug("Cache HIT — restoring data")
    df = ss("prot_df")
    c1 = ss("prot_c1")
    c2 = ss("prot_c2")
    pg_col = ss("prot_pg_col")
    sp_col = ss("prot_sp_col")
    sp_counts = ss("prot_sp_counts")

    st.success("Protein data restored from cache")
    debug("Restored successfully", f"Rows: {len(df)}, A: {c1}, B: {c2}")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.metric("**Condition A**", f"{len(c1)} replicates")
        st.write(" | ".join(c1))
    with col2:
        st.metric("**Condition B**", f"{len(c2)} replicates")
        st.write(" | ".join(c2))
    with col3:
        if st.button("Reconfigure", type="secondary"):
            ss("reconfig_prot", True)
            debug("Reconfigure requested")
            st.rerun()

    st.info(f"**Index**: `{pg_col}` • **Species column**: `{sp_col}`")
    st.markdown("### Proteins Detected per Species")
    st.dataframe(sp_counts, use_container_width=True, hide_index=True)
    st.bar_chart(sp_counts.set_index("Species")[["A", "B"]])
    restart_button()
    st.stop()

if ss("reconfig_prot", False):
    st.warning("Reconfiguring — please upload the same file again")
    debug("Reconfigure mode active")

# ====================== 1. FILE UPLOAD ======================
st.markdown("### 1. Upload Protein-Level File")
uploaded_file = st.file_uploader(
    "Drag & drop CSV/TSV/TXT (Spectronaut, DIA-NN, MaxQuant, FragPipe)",
    type=["csv", "tsv", "txt"],
    key="prot_upload"
)

if uploaded_file is None:
    debug("No file uploaded yet")
    st.info("Waiting for file upload...")
    restart_button()
    st.stop()

debug("File uploaded", f"{uploaded_file.name} — {uploaded_file.size:,} bytes")

# ====================== 2. SAFE FILE LOADER ======================
@st.cache_data(show_spinner="Loading file...")
def load_protein_file(file_obj):
    debug("Entered load_protein_file()")
    try:
        content = file_obj.getvalue()
        if len(content) == 0:
            st.error("File is empty")
            return None
        text = content.decode("utf-8", errors="replace")
        if text.startswith("\ufeff"):
            text = text[1:]
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python", on_bad_lines="skip")
        debug("File parsed successfully", f"{df.shape[0]} rows × {df.shape[1]} columns")
        return df
    except Exception as e:
        debug("File loading FAILED", str(e))
        st.error(f"Could not read file: {e}")
        return None

df_raw = load_protein_file(uploaded_file)

if df_raw is None or df_raw.empty:
    debug("df_raw is None or empty — stopping")
    st.stop()

st.success(f"Loaded {len(df_raw):,} protein entries")

# ====================== 3. DETECT INTENSITY COLUMNS ======================
debug("Detecting quantitative columns...")
intensity_cols = []
for col in df_raw.columns:
    try:
        cleaned = pd.to_numeric(
            df_raw[col].astype(str)
            .str.replace(r"[,\#NUM!]", "", regex=True)
            .str.strip(),
            errors='coerce'
        )
        valid_ratio = cleaned.notna().mean()
        if valid_ratio > 0.3:
            df_raw[col] = cleaned
            intensity_cols.append(col)
            debug(f"Column '{col}' → quantitative ({valid_ratio:.0%} valid)")
    except:
        continue

if len(intensity_cols) == 0:
    st.error("No quantitative columns found — check file format")
    debug("No intensity columns detected")
    st.stop()

debug("Intensity columns found", intensity_cols)

# ====================== 4. REPLICATE ASSIGNMENT ======================
st.markdown("### 2. Assign Replicates — Must Have Equal Count")
rows = []
for col in intensity_cols:
    preview = df_raw[col].dropna().head(3).astype(str).tolist()
    rows.append({
        "Column": col,
        "Preview": " | ".join(preview),
        "Condition A": True,
        "Condition B": False
    })

edited = st.data_editor(
    pd.DataFrame(rows),
    column_config={
        "Column": st.column_config.TextColumn(disabled=True),
        "Preview": st.column_config.TextColumn(disabled),
        "Condition A": st.column_config.CheckboxColumn("Condition A → A1,A2,..."),
        "Condition B": st.column_config.CheckboxColumn("Condition B → B1,B2,..."),
    },
    hide_index=True,
    use_container_width=True,
    num_rows="fixed"
)

a_cols = edited[edited["Condition A"]]["Column"].tolist()
b_cols = edited[edited["Condition B"]]["Column"].tolist()

debug("User selection", f"A: {a_cols} | B: {b_cols}")

if len(a_cols) != len(b_cols):
    st.error(f"Replicate count must be equal! A={len(a_cols)}, B={len(b_cols)}")
    st.stop()

n = len(a_cols)
rename_map = {a_cols[i]: f"A{i+1}" for i in range(n)}
rename_map.update({b_cols[i]: f"B{i+1}" for i in range(n)})
df = df_raw.rename(columns=rename_map).copy()
c1 = [f"A{i+1}" for i in range(n)]
c2 = [f"B{i+1}" for i in range(n)]

st.success(f"Replicates renamed → **A**: {', '.join(c1)} | **B**: {', '.join(c2)}")
debug("Renaming complete", {"c1": c1, "c2": c2})

# ====================== 5. PROTEIN GROUP COLUMN ======================
st.markdown("### 3. Select Protein Group Column")
pg_candidates = [c for c in df.columns if any(k in c.lower() for k in ["protein.group","pg","leading","accession","protein"])]
pg_col = st.selectbox("Protein Group ID (will be index)", pg_candidates or df.columns.tolist())

if st.button("Set as Index", type="primary"):
    df = df.set_index(pg_col)
    st.success(f"Index set to `{pg_col}`")
    debug("Index set", pg_col)
    st.rerun()

# ====================== 6. SPECIES DETECTION ======================
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
debug("Species detection", sp_col)

if sp_col != "Not found":
    def get_sp(x):
        x = str(x).upper()
        return next((s for s in species_list if s in x), "Other")
    df["Species"] = df[sp_col].apply(get_sp)
    sp_counts = df.groupby("Species")[c1 + c2].apply(lambda g: pd.Series({
        "A": (g[c1] > 1).any(axis=1).sum(),
        "B": (g[c2] > 1).any(axis=1).sum(),
        "Total": len(g)
    })).reset_index()
else:
    sp_counts = pd.DataFrame([{"Species": "All", "A": 0, "B": 0, "Total": len(df)}])

# ====================== 7. SAVE TO CACHE ======================
debug("Saving all data to session state...")
ss("prot_df", df)
ss("prot_c1", c1)
ss("prot_c2", c2)
ss("prot_pg_col", pg_col)
ss("prot_sp_col", sp_col)
ss("prot_sp_counts", sp_counts)
ss("reconfig_prot", False)

st.success("All processing complete — data cached!")

# ====================== FINAL DISPLAY ======================
col1, col2 = st.columns(2)
with col1: st.metric("Condition A", ", ".join(c1))
with col2: st.metric("Condition B", ", ".join(c2))

if sp_col != "Not found":
    st.markdown("### Proteins Detected per Species")
    st.dataframe(sp_counts, use_container_width=True, hide_index=True)
    st.bar_chart(sp_counts.set_index("Species")[["A", "B"]])

# ====================== DEBUG LOG DISPLAY (ALWAYS VISIBLE) ======================
if DEBUG and st.session_state.debug_log:
    with log_container:
        st.markdown("### Debug Log")
        for line in st.session_state.debug_log:
            st.markdown(line, unsafe_allow_html=True)
        if st.button("Clear Debug Log"):
            st.session_state.debug_log = []
            st.rerun()

# ====================== RESTART & FOOTER ======================
restart_button()

st.markdown("""
<div class="footer">
    <strong>Proprietary & Confidential | For Internal Use Only</strong><br>
    © 2024 Thermo Fisher Scientific Inc.
</div>
""", unsafe_allow_html=True)
