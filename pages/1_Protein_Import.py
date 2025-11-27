# pages/1_Protein_Import.py
import streamlit as st
import pandas as pd
import io
import numpy as np
from shared import restart_button

st.set_page_config(page_title="Protein Import", layout="wide")

# === STYLING ===
st.markdown("""
<style>
    .header {background:linear-gradient(90deg,#E71316,#A6192E); padding:20px 40px; color:white; margin:-80px -80px 40px;}
    .header h1,.header p {margin:0;}
    .stButton>button {background:#E71316 !important; color:white !important;}
    .intensity-col {background-color: #d4edda !important; color: #155724 !important; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>DIA Proteomics Pipeline</h1><p>Protein Import + Metadata</p></div>', unsafe_allow_html=True)

# === FILE UPLOAD ===
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
            st.session_state.metadata_bytes = uploaded_meta.getvalue()
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
st.write(f"**Metadata:** {len(df_meta)} runs")

# === METADATA MATCHING (Run Label substring in column name) ===
rename_dict = {}
used_cols = set()

for _, row in df_meta.iterrows():
    run_label = str(row["Run Label"]).strip()
    condition = str(row["Condition"]).strip()
    replicate = str(row["Replicate"]).strip()
    new_name = f"{condition}{replicate}"

    matches = [c for c in df_raw.columns if run_label in str(c)]
    
    if len(matches) == 0:
        st.warning(f"Not found: `{run_label}`")
        continue
    if len(matches) > 1:
        st.error(f"Multiple matches for `{run_label}`: {matches}")
        st.stop()
    
    col = matches[0]
    if col in used_cols:
        st.error(f"Duplicate match: `{col}` used twice!")
        st.stop()
    
    rename_dict[col] = new_name
    used_cols.add(col)

if not rename_dict:
    st.error("No columns matched from metadata!")
    st.stop()

# Apply rename
df = df_raw.rename(columns=rename_dict).copy()

# Extract A and B
c1 = sorted([v for v in rename_dict.values() if v.startswith("A")])
c2 = sorted([v for v in rename_dict.values() if v.startswith("B")])

if not c1 or not c2:
    st.error(f"Missing condition: A={len(c1)}, B={len(c2)}")
    st.stop()

# Force float64
for col in c1 + c2:
    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
df[c1 + c2] = df[c1 + c2].replace([0, np.nan], 1.0)

# === SPECIES DETECTION ===
candidate_cols = [c for c in df.columns if c not in c1 + c2]
species_col = st.selectbox("Column for species detection", candidate_cols, index=0)

def detect_species(x):
    if pd.isna(x): return "Other"
    t = str(x).upper()
    if any(k in t for k in ["HUMAN", "HOMO", "HSA"]): return "HUMAN"
    if any(k in t for k in ["MOUSE", "MUS", "MMU"]): return "MOUSE"
    if any(k in t for k in ["YEAST", "SACCHA"]): return "YEAST"
    return "Other"

df["Species"] = df[species_col].apply(detect_species)

# ===================================
# === BEAUTIFUL FINAL PREVIEW ===
# ===================================
st.success(f"Successfully matched and renamed **{len(rename_dict)}** intensity columns!")

colA, colB = st.columns(2)
with colA:
    st.subheader("Condition A")
    st.code(" | ".join(c1), language=None)
with colB:
    st.subheader("Condition B")
    st.code(" | ".join(c2), language=None)

st.write("**Species distribution:**", dict(df["Species"].value_counts()))

# Mapping table
mapping_df = pd.DataFrame([
    {"Original Column": old, "Renamed To": new}
    for old, new in rename_dict.items()
])
st.subheader("Column Mapping from Metadata")
st.dataframe(mapping_df, use_container_width=True)

# FINAL PREVIEW - THIS WORKS 100%
st.subheader("Data Preview (first 10 rows)")

meta_cols = [c for c in df.columns if c not in c1 + c2][:5]
intensity_cols_to_show = (c1 + c2)[:12]
display_cols = meta_cols + intensity_cols_to_show

preview = df[display_cols].head(10).copy()

def highlight_intensity(val):
    # val is a column of values; we use the column name from the outer scope
    return ['background-color: #d4edda; color: #155724; font-weight: bold' 
            if col_name in (c1 + c2) else '' 
            for col_name in preview.columns]

styled = preview.style.apply(highlight_intensity, axis=0)
st.dataframe(preview, use_container_width=True)

# === SAVE TO SESSION ===
st.session_state.prot_df = df
st.session_state.prot_c1 = c1
st.session_state.prot_c2 = c2
st.session_state.species_column_name = species_col

st.success("Protein data ready for analysis!")

if st.button("Go to Protein Analysis", type="primary", use_container_width=True):
    st.switch_page("pages/3_Protein_Analysis.py")

restart_button()
