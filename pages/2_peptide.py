# pages/1_Protein_Import.py
import streamlit as st
import pandas as pd
import io
from shared import restart_button
# Add this at the very top of your file (after imports)

from datetime import datetime

# ====================== UNIVERSAL DEBUG LOGGER ======================
def log(msg, data=None):
    """Shows debug info in the main app — visible on ALL deployments"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.write(f"<small style='color:#666;'>[{timestamp}] {msg}</small>", unsafe_allow_html=True)
    if data is not None:
        with st.expander(f"Details → {msg}", expanded=False):
            st.code(data, language="python")

# Optional: turn off in production
DEBUG = True  # ← Set to False when going live


# ====================== SAFE SESSION STATE ======================
def ss(key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
        debug(f"Initialized ss('{key}') = {default}")
    else:
        debug(f"Retrieved ss('{key}') = {type(st.session_state[key]).__name__}")
    return st.session_state[key]

# ====================== BRANDING ======================
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
    .fixed-restart {position:fixed; bottom:20px; left:50%; transform:translateX(-50%); z-index:999; width:340px;}
    .stButton>button {background:var(--red)!important; color:white!important; border-radius:6px!important;}
    .footer {text-align:center; padding:40px; color:var(--gray); font-size:12px; border-top:1px solid var(--light); margin-top:80px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>DIA Proteomics Pipeline</h1><p>Module — Protein-Level Import</p></div>', unsafe_allow_html=True)
st.markdown('<div class="nav"><div class="nav-item">Peptide Import</div><div class="nav-item active">Protein Import</div><div class="nav-item">Analysis</div></div>', unsafe_allow_html=True)
st.markdown('<div class="module-header"><div class="module-icon">Protein</div><div><h2 style="margin:0;color:white;">Protein Data Import</h2><p style="margin:5px 0 0;opacity:0.9;">Auto-detect species • Equal replicates • Set Protein Group as index</p></div></div>', unsafe_allow_html=True)

# ====================== RESTORE FROM CACHE ======================
debug("Checking for cached data...")
if ss("prot_df") is not None and not ss("reconfig_prot", False):
    debug("Cache hit! Restoring data...")
    df = ss("prot_df")
    c1 = ss("prot_c1")
    c2 = ss("prot_c2")
    pg_col = ss("prot_pg_col")
    sp_col = ss("prot_sp_col")
    sp_counts = ss("prot_sp_counts")

    st.success("Protein data restored from cache")
    debug(f"Restored: {len(df)} rows, index={df.index.name}, A={c1}, B={c2}")

    col1, col2, col3 = st.columns([2,2,1])
    with col1: st.metric("**Condition A**", f"{len(c1)} reps"); st.write(" | ".join(c1))
    with col2: st.metric("**Condition B**", f"{len(c2)} reps"); st.write(" | ".join(c2))
    with col3:
        if st.button("Reconfigure"):
            ss("reconfig_prot", True)
            debug("Reconfigure flag set")
            st.rerun()

    st.info(f"**Index**: `{pg_col}` • **Species column**: `{sp_col}`")
    st.markdown("### Proteins Detected per Species")
    st.dataframe(sp_counts, use_container_width=True, hide_index=True)
    st.bar_chart(sp_counts.set_index("Species")[["A", "B"]])
    restart_button()
    st.stop()

if ss("reconfig_prot", False):
    st.warning("Reconfiguring — please re-upload the same file")
    debug("Reconfigure mode active")

# ====================== UPLOAD FILE ======================
st.markdown("### 1. Upload Protein-Level File")
uploaded_file = st.file_uploader("CSV/TSV/TXT", type=["csv","tsv","txt"], key="prot_upload")

if uploaded_file is None:
    debug("No file uploaded yet")
    st.info("Please upload a file to begin")
    restart_button()
    st.stop()

debug(f"File uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")

# ====================== LOAD FILE SAFELY ======================
@st.cache_data(show_spinner="Reading file...")
def load_file(file_obj):
    debug("Inside load_file() function")
    try:
        raw = file_obj.read()
        debug(f"Raw bytes read: {len(raw)}")
        if len(raw) == 0:
            st.error("File is empty")
            return None
        file_obj.seek(0)
        text = raw.decode("utf-8", errors="replace")
        if text.startswith("\ufeff"):
            text = text[1:]
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python", on_bad_lines="skip")
        debug(f"Loaded DataFrame: {df.shape[0]} rows × {df.shape[1]} cols")
        return df
    except Exception as e:
        debug(f"ERROR in load_file: {e}")
        st.error(f"Failed to read file: {e}")
        return None

df_raw = load_file(uploaded_file)

if df_raw is None:
    debug("df_raw is None — stopping execution")
    st.stop()

debug("df_raw loaded successfully")
st.success(f"Loaded {len(df_raw):,} rows, {len(df_raw.columns)} columns")

# ====================== DETECT INTENSITY COLUMNS ======================
debug("Starting intensity column detection")
intensity_cols = []
for col in df_raw.columns:
    try:
        cleaned = pd.to_numeric(
            df_raw[col].astype(str).str.replace(r"[,\#NUM!]", "", regex=True),
            errors='coerce'
        )
        valid_ratio = cleaned.notna().mean()
        debug(f"Column '{col}': {valid_ratio:.1%} numeric")
        if valid_ratio > 0.3:
            df_raw[col] = cleaned
            intensity_cols.append(col)
    except Exception as e:
        debug(f"Failed on column '{col}': {e}")

debug(f"Found {len(intensity_cols)} intensity columns: {intensity_cols}")

if len(intensity_cols) == 0:
    st.error("No quantitative columns found — is this a protein quantification file?")
    st.stop()

# ====================== REPLICATE ASSIGNMENT ======================
st.markdown("### 2. Assign Replicates (must be equal)")
rows = []
for col in intensity_cols:
    preview = df_raw[col].dropna().head(3).astype(str).tolist()
    rows.append({"Column": col, "Preview": " | ".join(preview), "A": True, "B": False})

edited = st.data_editor(pd.DataFrame(rows), column_config={
    "Column": st.column_config.TextColumn(disabled=True),
    "Preview": st.column_config.TextColumn(disabled=True),
    "A": st.column_config.CheckboxColumn("Condition A → A1,A2,..."),
    "B": st.column_config.CheckboxColumn("Condition B → B1,B2,..."),
}, hide_index=True, use_container_width=True, num_rows="fixed")

a_cols = edited[edited["A"]]["Column"].tolist()
b_cols = edited[edited["B"]]["Column"].tolist()
debug(f"User selected — A: {a_cols}, B: {b_cols}")

if len(a_cols) != len(b_cols) or len(a_cols) == 0:
    st.error(f"Replicate count mismatch: A={len(a_cols)}, B={len(b_cols)}")
    st.stop()

n = len(a_cols)
rename_map = {a_cols[i]: f"A{i+1}" for i in range(n)}
rename_map.update({b_cols[i]: f"B{i+1}" for i in range(n)})
df = df_raw.rename(columns=rename_map).copy()
c1 = [f"A{i+1}" for i in range(n)]
c2 = [f"B{i+1}" for i in range(n)]
debug(f"Renamed columns: {list(rename_map.values())}")

st.success(f"Renamed → A: {', '.join(c1)} | B: {', '.join(c2)}")

# ====================== PROTEIN GROUP & SPECIES ======================
pg_candidates = [c for c in df.columns if any(k in c.lower() for k in ["protein.group","pg","leading","accession"])]
pg_col = st.selectbox("Protein Group column", pg_candidates or df.columns.tolist())

if st.button("Set as Index"):
    df = df.set_index(pg_col)
    debug(f"Set index to {pg_col}")
    st.rerun()

# Species detection
species_list = ["HUMAN","MOUSE","RAT","ECOLI","BOVIN","YEAST"]
sp_col = None
for c in df.columns:
    if c in c1 + c2: continue
    if df[c].astype(str).str.upper().str.contains("|".join(species_list)).any():
        sp_col = c
        break
sp_col = sp_col or "Not found"
debug(f"Species column: {sp_col}")

# ====================== SAVE TO CACHE ======================
debug("Saving all data to session state...")
ss("prot_df", df)
ss("prot_c1", c1)
ss("prot_c2", c2)
ss("prot_pg_col", pg_col)
ss("prot_sp_col", sp_col)
ss("reconfig_prot", False)

st.success("All done! Data cached and ready for next steps")

# Final display & restart
col1, col2 = st.columns(2)
with col1: st.metric("Condition A", ", ".join(c1))
with col2: st.metric("Condition B", ", ".join(c2))

restart_button()

st.markdown("""
<div class="footer">
    <strong>Proprietary & Confidential</strong><br>
    © 2024 Thermo Fisher Scientific
</div>
""", unsafe_allow_html=True)
