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
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="header"><h1>DIA Proteomics Pipeline</h1><p>Protein Import</p></div>', unsafe_allow_html=True)

# === UPLOAD BOTH FILES ===
col1, col2 = st.columns(2)
with col1:
    if "protein_bytes" not in st.session_state:
        uploaded_prot = st.file_uploader("Upload Protein File (wide format)", type=["csv","tsv","txt"])
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
@st.cache_data
def load_df(bytes_data):
    text = bytes_data.decode("utf-8", errors="replace")
    if text.startswith("\ufeff"): text = text[1:]
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")

df_prot = load_df(st.session_state.protein_bytes)
meta = load_df(st.session_state.metadata_bytes)

st.write(f"Protein data: **{df_prot.shape[0]:,}** proteins × **{df_prot.shape[1]}** columns")
st.write(f"Metadata: **{len(meta)}** runs")

# === AUTO-MATCH RUNS USING METADATA ===
# Find columns in protein data that match "File Name" or "Run Label" in metadata
run_col_in_prot = None
for col in df_prot.columns:
    if col in meta["File Name"].astype(str).values or col in meta["Run Label"].astype(str).values:
        run_col_in_prot = col
        break

if not run_col_in_prot:
    st.error("Could not match any column in protein file to metadata 'File Name' or 'Run Label'")
    st.stop()

# Map protein columns → condition via metadata
col_to_condition = {}
col_to_replicate = {}
col_to_correction = {}

for _, row in meta.iterrows():
    file_key = str(row["File Name"])
    label_key = str(row["Run Label"])
    condition = row["Condition"]
    replicate = row["Replicate"]
    factor = row.get("Quantity Correction Factor", 1.0)

    for col in df_prot.columns:
        if str(col) == file_key or str(col) == label_key:
            col_to_condition[col] = condition
            col_to_replicate[col] = f"{condition}{replicate}"
            col_to_correction[col] = float(factor)
            break

matched_cols = list(col_to_condition.keys())
if len(matched_cols) < 2:
    st.error("Less than 2 runs matched. Check file names.")
    st.stop()

st.success(f"Matched **{len(matched_cols)}** runs from metadata")

# === RENAME COLUMNS AUTOMATICALLY ===
rename_map = {col: col_to_replicate[col] for col in matched_cols}
df = df_prot.rename(columns=rename_map).copy()

# Extract condition A and B
conditions = meta["Condition"].unique()
if len(conditions) != 2:
    st.error("Metadata must have exactly 2 conditions (A and B)")
    st.stop()

cond_a, cond_b = conditions
c1 = sorted([c for c in df.columns if c.startswith(cond_a)])
c2 = sorted([c for c in df.columns if c.startswith(cond_b)])

st.success(f"Auto-assigned → **{cond_a}**: {', '.join(c1)} | **{cond_b}**: {', '.join(c2)}")

# === APPLY CORRECTION FACTORS (optional) ===
if st.checkbox("Apply Quantity Correction Factors from metadata", value=True):
    for old_col, factor in col_to_correction.items():
        new_col = rename_map.get(old_col)
        if new_col and factor != 1.0:
            df[new_col] = df[new_col] * factor
    st.info("Correction factors applied")

# === REPLACE 0 / NaN → 1.0 ===
intensity_cols = c1 + c2
df[intensity_cols] = df[intensity_cols].replace([0, np.nan], 1.0)

# === SPECIES DETECTION ===
st.markdown("### Select Column for Species Detection")
candidate_cols = [c for c in df.columns if c not in intensity_cols]
species_col = st.selectbox("Column with protein description/accession", candidate_cols, index=0)

species_map = {
    "HUMAN": ["HUMAN", "HOMO", "HSA"],
    "MOUSE": ["MOUSE", "MUS", "MMU"],
    "YEAST": ["YEAST", "SACCHA"],
    "ECOLI": ["ECOLI", "ESCHERICHIA"],
    "Other": []
}
def detect_species(x):
    if pd.isna(x): return "Other"
    x = str(x).upper()
    for sp, keywords in species_map.items():
        if any(k in x for k in keywords):
            return sp
    return "Other"

df["Species"] = df[species_col].apply(detect_species)
st.write("Species distribution:", df["Species"].value_counts().to_dict())

# === SAVE TO SESSION ===
st.session_state.prot_df = df
st.session_state.prot_c1 = c1
st.session_state.prot_c2 = c2
st.session_state.species_column_name = species_col
st.session_state.condition_colors = {
    cond_a: meta[meta["Condition"] == cond_a]["Color"].iloc[0],
    cond_b: meta[meta["Condition"] == cond_b]["Color"].iloc[0],
}

st.success("Protein data ready with metadata!")

if st.button("Go to Protein Analysis", type="primary", use_container_width=True):
    st.switch_page("pages/3_Protein_Analysis.py")

restart_button()
