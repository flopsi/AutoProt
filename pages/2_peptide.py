# pages/2_Peptide_Import.py
import streamlit as st
import pandas as pd
import io
from shared import restart_button, debug
import plotly.express as px
# ====================== SAFE SESSION STATE ======================
def ss(key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

st.set_page_config(page_title="Peptide Import", layout="wide")
debug("Peptide Import page loaded")

# ====================== BRANDING (same as protein page) ======================
st.markdown("""
<style>
    :root {--red:#E71316; --darkred:#A6192E; --gray:#54585A; --light:#E2E3E4;}
    .header {background:linear-gradient(90deg,var(--red),var(--darkred)); padding:20px 40px; color:white; margin:-80px -80px 40px -80px;}
    .header h1,.header p {margin:0;}
    .module-header {background:linear-gradient(90deg,var(--red),var(--darkred)); padding:30px; border-radius:12px; color:white; margin-bottom:30px;}
    .module-icon {width:60px;height:60px;background:rgba(255,255,255,0.2);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:32px;}
    .stButton>button {background:var(--red)!important; color:white!important; border-radius:8px!important;}
    .footer {text-align:center; padding:40px; color:var(--gray); font-size:12px; border-top:1px solid var(--light); margin-top:80px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>DIA Proteomics Pipeline</h1><p>Module — Peptide-Level Import</p></div>', unsafe_allow_html=True)
st.markdown("""
<div class="module-header">
    <div class="module-icon">Peptide</div>
    <div>
        <h2 style="margin:0;color:white;">Peptide Data Import</h2>
        <p style="margin:5px 0 0;opacity:0.9;">Auto-detect peptide sequence • Equal replicates • Species detection (incl. ECOLI)</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ====================== RESTORE FROM CACHE ======================
if ss("pept_df") is not None and not ss("reconfig_pept", False):
    df = ss("pept_df")
    c1 = ss("pept_c1")
    c2 = ss("pept_c2")
    seq_col = ss("pept_seq_col")
    sp_col = ss("pept_sp_col")

    st.success("Peptide data restored from cache")
    col1, col2, col3 = st.columns([2,2,1])
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

    st.info(f"**Peptide Sequence (index)**: `{seq_col}` • **Species column**: `{sp_col}`")
    restart_button()
    st.stop()

if ss("reconfig_pept", False):
    st.warning("Reconfiguring — please upload the same peptide file again")

# ====================== 1. UPLOAD PEPTIDE FILE ======================
st.markdown("### 1. Upload Peptide-Level File")
uploaded_file = st.file_uploader(
    "CSV/TSV/TXT from Spectronaut, DIA-NN, MaxQuant, FragPipe, etc.",
    type=["csv", "tsv", "txt"],
    key="pept_upload"
)

if uploaded_file is None:
    st.info("Please upload your peptide quantification file")
    restart_button()
    st.stop()

debug("Peptide file uploaded", f"{uploaded_file.name} ({uploaded_file.size:,} bytes)")

# ====================== 2. LOAD FILE SAFELY ======================
@st.cache_data(show_spinner="Loading peptide file...")
def load_peptide_file(file_obj):
    try:
        content = file_obj.getvalue()
        if len(content) == 0:
            st.error("Uploaded file is empty")
            return None
        text = content.decode("utf-8", errors="replace")
        if text.startswith("\ufeff"):
            text = text[1:]
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python", on_bad_lines="skip")
        debug("Peptide file loaded", f"{df.shape[0]} rows × {df.shape[1]} columns")
        return df
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        debug("Peptide load error", str(e))
        return None

df_raw = load_peptide_file(uploaded_file)
if df_raw is None:
    st.stop()

st.success(f"Loaded {len(df_raw):,} peptide entries")

# ====================== 3. DETECT INTENSITY COLUMNS ======================
intensity_cols = []
for col in df_raw.columns:
    try:
        cleaned = pd.to_numeric(
            df_raw[col].astype(str).str.replace(r"[,\#NUM!]", "", regex=True),
            errors='coerce'
        )
        if cleaned.notna().mean() > 0.3:
            df_raw[col] = cleaned
            intensity_cols.append(col)
    except:
        continue

if len(intensity_cols) == 0:
    st.error("No quantitative columns found")
    st.stop()

debug("Intensity columns detected", intensity_cols)

# ====================== 4. REPLICATE ASSIGNMENT (EQUAL COUNT REQUIRED) ======================
st.markdown("### 2. Assign Replicates — Must Have Equal Number")
rows = [{"Column": c, "Preview": " | ".join(map(str, df_raw[c].dropna().head(3))), "A": True, "B": False} for c in intensity_cols]

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

if len(a_cols) != len(b_cols) or len(a_cols) == 0:
    st.error(f"Both conditions must have the same number of replicates (A={len(a_cols)}, B={len(b_cols)})")
    st.stop()

n = len(a_cols)
rename_map = {a_cols[i]: f"A{i+1}" for i in range(n)}
rename_map.update({b_cols[i]: f"B{i+1}" for i in range(n)})
df = df_raw.rename(columns=rename_map).copy()
c1 = [f"A{i+1}" for i in range(n)]
c2 = [f"B{i+1}" for i in range(n)]

st.success(f"Replicates renamed → **A**: {', '.join(c1)} | **B**: {', '.join(c2)}")

# ====================== 5. SELECT PEPTIDE SEQUENCE COLUMN ======================
st.markdown("### 3. Select Peptide Sequence Column")
seq_candidates = [c for c in df.columns if any(k in c.lower() for k in ["sequence","peptide","seq","stripped","precursor"])]
seq_col = st.selectbox("Which column contains the peptide sequence?", seq_candidates, index=0)

if st.button("Set Peptide Sequence as Index", type="primary"):
    if seq_col in df.columns:
        df = df.set_index(seq_col)
        st.success(f"Index set to `{seq_col}`")
        st.rerun()

# ====================== 6. SPECIES DETECTION (incl. ECOLI) ======================
st.markdown("### 4. Auto-Detect Species Column")
species_list = ["HUMAN","MOUSE","RAT","ECOLI","BOVIN","YEAST","RABIT","CANFA","MACMU","PANTR"]

def find_species_col(df):
    pattern = "|".join(species_list)
    for c in df.columns:
        if c in c1 + c2: continue
        if df[c].astype(str).str.upper().str.contains(pattern).any():
            return c
    return "Not found"

sp_col = find_species_col(df)

if sp_col != "Not found":
    def extract_species(x):
        x = str(x).upper()
        return next((s for s in species_list if s in x), "Other")
    df["Species"] = df[sp_col].apply(extract_species)
else:
    sp_col = "Not found"




# ====================== FINAL: SAVE EVERYTHING PERMANENTLY ======================
st.success("Peptide processing complete — data is now permanently saved!")

# These lines make the peptide analysis page work forever
ss("peptide_data_ready", True)
ss("pept_final_df", df)
ss("pept_final_c1", c1)
ss("pept_final_c2", c2)
ss("pept_final_seq", df.index.name if not isinstance(df.index, pd.RangeIndex) else "None")


# GO TO ANALYSIS BUTTON
st.markdown("---")
if st.button("Go to Peptide Exploratory Analysis", type="primary", use_container_width=True):
    st.switch_page("pages/4_Peptide_Analysis.py")

restart_button()

restart_button()

st.markdown("""
<div class="footer">
    <strong>Proprietary & Confidential | For Internal Use Only</strong><br>
    © 2024 Thermo Fisher Scientific Inc.
</div>
""", unsafe_allow_html=True)
