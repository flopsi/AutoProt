# pages/1_Protein_Import.py
import streamlit as st
import pandas as pd
import io
import numpy as np
from shared import restart_button

st.set_page_config(page_title="Protein Import", layout="wide")
st.markdown("""
<style>
    .header {background:linear-gradient(90deg,#E71316,#A6192E); padding:20px 40px; color:white; margin:-80px -80px 40px;}
    .header h1,.header p {margin:0;}
    .stButton>button {background:#E71316 !important; color:white !important;}
    .matched-col {background-color: #d4edda !important; color: #155724;}
    .preview-table th {background-color: #f8f9fa;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="header"><h1>DIA Proteomics Pipeline</h1><p>Protein Import + Metadata</p></div>', unsafe_allow_html=True)

# === UPLOAD FILES ===
col1, col2 = st.columns(2)
with col1:
    if "protein_bytes" not in st.session_state:
        uploaded_prot = st.file_uploader("Upload Wide-Format Protein File", type=["csv","tsv","txt"])
        if uploaded_prot:
            st.session_state.protein_bytes = uploaded_prot.getvalue()
            st.session_state.protein_name = uploaded_prot.name
            st.rerun()
    else:
        st.success(f"Protein: **{st.session_state.protein_name}**")

with col2:
    if "metadata_bytes" not in st.session_state:
        uploaded_meta = st.file_uploader("Upload Metadata File (metadata.tsv)", type=["tsv","csv","txt"])
        if uploaded_meta:
            st.session_state.metadata_bytes = uploaded_meta.getvalue792()
            st.session_state.metadata_name = uploaded_meta.name
            st.rerun()
    else:
        st.success(f"Metadata: **{st.session_state.metadata_name}**")

if "protein_bytes" not in st.session_state or "metadata_bytes" not in st.session_state:
    st.info("Please upload both files to continue.")
    restart_button()
    st.stop()

# === LOAD DATA ===
@st.cache_data(show_spinner="Loading files...")
def load_df(b):
    text = b.decode("utf-8", errors="replace")
    if text.startswith("\ufeff"): text = text[1:]
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")

df_raw = load_df(st.session_state.protein_bytes)
df_meta = load_df(st.session_state.metadata_bytes)

st.write(f"**Raw data:** {df_raw.shape[0]:,} proteins × {df_raw.shape[1]} columns")
st.write(f"**Metadata:** {len(df_meta)} runs defined")

# === METADATA-BASED RENAMING (Run Label substring match) ===
rename_dict = {}
correction_dict = {}
used_columns = set()

for idx, row in df_meta.iterrows():
    run_label = str(row["Run Label"]).strip()
    condition = str(row["Condition"]).strip()
    replicate = str(row["Replicate"]).strip()
    new_name = f"{condition}{replicate}"
    factor = float(row.get("Quantity Correction Factor", 1.0))

    matches = [c for c in df_raw.columns if run_label in str(c)]
    
    if len(matches) == 0:
        st.warning(f"Run Label not found → `{run_label}`")
        continue
    if len(matches) > 1:
        st.error(f"Multiple matches for `{run_label}`: {matches}")
        st.stop()
    
    col = matches[0]
    if col in used_columns:
        st.error(f"Column `{col}` used twice!")
        st.stop()
    
    rename_dict[col] = new_name
    correction_dict[new_name] = factor
    used_columns.add(col)

if not rename_dict:
    st.error("No Run Labels matched any columns!")
    st.stop()

# Apply renaming
df = df_raw.rename(columns=rename_dict).copy()

# Extract conditions
c1 = sorted([c for c in rename_dict.values() if c.startswith("A")])
c2 = sorted([c for c in rename_dict.values() if c.startswith("B")])

# === FORCE FLOAT + CLEAN ===
for col in c1 + c2:
    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
df[c1 + c2] = df[c1 + c2].replace([0, np.nan], 1.0)

# Apply correction factors
if st.checkbox("Apply Quantity Correction Factors", value=True):
    for col, f in correction_dict.items():
        if f != 1.0 and col in df.columns:
            df[col] = df[col] * f

# === SPECIES DETECTION ===
candidate_cols = [c for c in df.columns if c not in c1 + c2]
species_col = st.selectbox("Column for species detection", candidate_cols, index=0)

def get_species(x):
    if pd.isna(x): return "Other"
    t = str(x).upper()
    if any(k in t for k in ["HUMAN", "HOMO", "HSA"]): return "HUMAN"
    if any(k in t for k in ["MOUSE", "MUS", "MMU"]): return "MOUSE"
    if any(k in t for k in ["YEAST", "SACCHA"]): return "YEAST"
    return "Other"

df["Species"] = df[
