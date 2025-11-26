# pages/1_Protein_Import.py
import streamlit as st
import pandas as pd
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
    .module-icon {width:60px;height:60px;background:rgba(255,255,255,0.2);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:32px;}
    .fixed-restart {position:fixed; bottom:20px; left:50%; transform:translateX(-50%); z-index:999; width:340px;}
    .stButton>button {background:var(--red)!important; color:white!important; border-radius:6px!important;}
    .footer {text-align:center; padding:40px; color:var(--gray); font-size:12px; border-top:1px solid var(--light); margin-top:80px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>DIA Proteomics Pipeline</h1><p>Module — Protein-Level Import</p></div>', unsafe_allow_html=True)
st.markdown('<div class="nav"><div class="nav-item">Peptide Import</div><div class="nav-item active">Protein Import</div><div class="nav-item">Analysis</div></div>', unsafe_allow_html=True)
st.markdown('<div class="module-header"><div class="module-icon">Protein</div><div><h2 style="margin:0;color:white;">Protein Data Import</h2><p style="margin:5px 0 0;opacity:0.9;">Auto-detect species • Equal replicates • Set Protein Group as index</p></div></div>', unsafe_allow_html=True)

# ====================== RESTORE FROM CACHE =====================
if ss("prot_df") is not None and not ss("reconfig_prot", False):
    df = ss("prot_df")
    c1 = ss("prot_c1")
    c2 = ss("prot_c2")
    pg_col = ss("prot_pg_col")
    sp_col = ss("prot_sp_col")
    sp_counts = ss("prot_sp_counts")

    st.success("Protein data restored from session cache")

    col1, col2, col3 = st.columns([2,2,1])
    with col1:
        st.metric("**Condition A**", f"{len(c1)} replicates")
        st.write(" | ".join(c1))
    with col2:
        st.metric("**Condition B**", f"{len(c2)} replicates")
        st.write(" | ".join(c2))
    with col3:
        if st.button("Reconfigure", type="secondary"):
            ss("reconfig_prot", True)
            st.rerun()

    st.info(f"**Protein Group (index)**: `{pg_col}` • **Species column**: `{sp_col}`")
    st.markdown("### Proteins Detected per Species")
    st.dataframe(sp_counts, use_container_width=True, hide_index=True)
    st.bar_chart(sp_counts.set_index("Species")[["A", "B"]])

    restart_button()
    st.stop()

# Reconfigure mode
if ss("reconfig_prot", False):
    st.warning("Reconfiguring — please upload the same file again")

# ===================== 1. UPLOAD FILE =====================
st.markdown("### 1. Upload Protein-Level File")
uploaded_file = st.file_uploader(
    "Drag & drop or browse (CSV/TSV/TXT)",
    type=["csv", "tsv", "txt"],
    key="prot_upload"
)

# THIS IS THE KEY FIX — DO NOT PROCEED IF NO FILE
if uploaded_file is None:
    st.info("Waiting for you to upload a protein quantification file...")
    restart_button()
    st.stop()

# ===================== 2. LOAD FILE SAFELY =====================
@st.cache_data(show_spinner="Loading your file...")
def load_protein_file(file_obj):
    try:
        raw_bytes = file_obj.read()
        if not raw_bytes.strip():
            st.error("Uploaded file is empty!")
            return None
        
        # Reset pointer and decode
        file_obj.seek(0)
        text = raw_bytes.decode("utf-8", errors="replace")
        if text.startswith("\ufeff"):
            text = text[1:]

        df = pd.read_csv(io.StringIO(text), sep=None, engine="python", on_bad_lines="skip")
        
        if df.empty:
            st.error("File was read but contains no data rows.")
            return None
            
        st.success(f"File loaded: {len(df):,} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        st.error(f"Could not read file: {str(e)}")
        return None

df_raw = load_protein_file(uploaded_file)

# FINAL SAFETY NET — STOP HERE IF LOADING FAILED
if df_raw is None:
    st.stop()

# ===================== 3. DETECT INTENSITY COLUMNS =====================
st.markdown("### Detecting quantitative columns...")
intensity_cols = []

for col in df_raw.columns:
    try:
        # Clean common junk and try to convert to numeric
        cleaned = pd.to_numeric(
            df_raw[col]
            .astype(str)
            .str.replace(r"[,\#NUM!]", "", regex=True)
            .str.strip(),
            errors='coerce'
        )
        if cleaned.notna().sum() > len(df_raw) * 0.3:  # >30% valid numbers
            df_raw[col] = cleaned
            intensity_cols.append(col)
    except:
        continue

if len(intensity_cols) == 0:
    st.error("No intensity/replicate columns found. This doesn't look like a quantification file.")
    st.stop()

st.success(f"Found {len(intensity_cols)} quantitative columns")

# ===================== 4. REPLICATE ASSIGNMENT =====================
st.markdown("### 2. Assign Replicates — Must Be Equal Count")
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
        "Preview": st.column_config.TextColumn(disabled=True),
        "Condition A": st.column_config.CheckboxColumn("Condition A → A1,A2,..."),
        "Condition B": st.column_config.CheckboxColumn("Condition B → B1,B2,..."),
    },
    hide_index=True,
    use_container_width=True,
    num_rows="fixed"
)

a_cols = edited[edited["Condition A"]]["Column"].tolist()
b_cols = edited[edited["Condition B"]]["Column"].tolist()

if len(a_cols) != len(b_cols) or len(a_cols) == 0:
    st.error(f"Both conditions must have the same number of replicates. Currently A: {len(a_cols)}, B: {len(b_cols)}")
    st.stop()

# Rename to A1/A2... B1/B2...
n = len(a_cols)
rename_map = {a_cols[i]: f"A{i+1}" for i in range(n)}
rename_map.update({b_cols[i]: f"B{i+1}" for i in range(n)})
df = df_raw.rename(columns=rename_map).copy()
c1 = [f"A{i+1}" for i in range(n)]
c2 = [f"B{i+1}" for i in range(n)]

st.success(f"Renamed → **A**: {', '.join(c1)} | **B**: {', '.join(c2)}")

# ===================== 5. PROTEIN GROUP COLUMN =====================
st.markdown("### 3. Select Protein Group Column")
pg_candidates = [c for c in df.columns if any(kw in c.lower() for kw in ["protein.group","pg","leading","accession","protein"])]
pg_col = st.selectbox("Protein Group ID column (will become index)", pg_candidates or df.columns.tolist())

if st.button("Set as Index", type="primary"):
    df = df.set_index(pg_col)
    st.success(f"Index set to `{pg_col}`")
    st.rerun()

# ===================== 6. SPECIES DETECTION =====================
st.markdown("### 4. Auto-Detect Species Column")
species_list = ["HUMAN","MOUSE","RAT","ECOLI","BOVIN","YEAST","RABIT","CANFA","MACMU","PANTR"]

def detect_species_col(df):
    pattern = "|".join(species_list)
    for c in df.columns:
        if c in c1 + c2: continue
        if df[c].astype(str).str.upper().str.contains(pattern).any():
            return c
    return None

sp_col = detect_species_col(df) or "Not found"

if sp_col != "Not found":
    def get_species(x):
        x = str(x).upper()
        return next((s for s in species_list if s in x), "Other")
    df["Species"] = df[sp_col].apply(get_species)
    sp_counts = (
        df.assign(
            A=(df[c1] > 1).any(axis=1),
            B=(df[c2] > 1).any(axis=1)
        )
        .groupby("Species")[["A","B"]].sum()
        .assign(Total=lambda x: x.A + x.B - (x.A & x.B).sum())
        .reset_index()
    )
else:
    sp_counts = pd.DataFrame([{"Species":"All data","A":0,"B":0,"Total":len(df)}])

# ===================== 7. SAVE TO CACHE =====================
st.success("All steps completed — data is now cached!")

ss("prot_df", df)
ss("prot_c1", c1)
ss("prot_c2", c2)
ss("prot_pg_col", pg_col)
ss("prot_sp_col", sp_col)
ss("prot_sp_counts", sp_counts)
ss("reconfig_prot", False)

# ===================== 8. FINAL SUMMARY =====================
col1, col2 = st.columns(2)
with col1: st.metric("Condition A", ", ".join(c1))
with col2: st.metric("Condition B", ", ".join(c2))

if sp_col != "Not found":
    st.markdown("### Proteins Detected per Species")
    st.dataframe(sp_counts, use_container_width=True, hide_index=True)
    st.bar_chart(sp_counts.set_index("Species")[["A", "B"]])

# ===================== FOOTER =====================
restart_button()

st.markdown("""
<div class="footer">
    <strong>Proprietary & Confidential | For Internal Use Only</strong><br>
    © 2024 Thermo Fisher Scientific Inc. All rights reserved.
</div>
""", unsafe_allow_html=True)
