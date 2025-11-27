# pages/1_Protein_Import.py
import streamlit as st
import pandas as pd
import io
from shared import restart_button
import numpy as np


st.set_page_config(page_title="Protein Import", layout="wide")
st.markdown("""
<style>
    .header {background:linear-gradient(90deg,#E71316,#A6192E); padding:20px 40px; color:white; margin:-80px -80px 40px;}
    .header h1,.header p {margin:0;}
    .stButton>button {background:#E71316 !important; color:white !important;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="header"><h1>DIA Proteomics Pipeline</h1><p>Protein Import + Metadata</p></div>', unsafe_allow_html=True)

# === UPLOAD ===
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
    st.info("Please upload both files.")
    restart_button()
    st.stop()

# === LOAD DATA ===
@st.cache_data(show_spinner="Loading data...")
def load_df(b):
    text = b.decode("utf-8", errors="replace")
    if text.startswith("\ufeff"): text = text[1:]
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")

df_raw = load_df(st.session_state.protein_bytes)
df_meta = load_df(st.session_state.metadata_bytes)

# === METADATA MATCHING ===
rename_dict = {}
used = set()

for _, row in df_meta.iterrows():
    label = str(row["Run Label"]).strip()
    cond = str(row["Condition"]).strip()
    rep = str(row["Replicate"]).strip()
    new_name = f"{cond}{rep}"

    matches = [c for c in df_raw.columns if label in str(c)]
    if not matches:
        st.warning(f"Run Label not found: `{label}`")
        continue
    if len(matches) > 1:
        st.error(f"Multiple matches: `{label}` → {matches}")
        st.stop()
    col = matches[0]
    if col in used:
        st.error(f"Duplicate column: `{col}`")
        st.stop()
    rename_dict[col] = new_name
    used.add(col)

if not rename_dict:
    st.error("No columns matched from metadata!")
    st.stop()

df = df_raw.rename(columns=rename_dict).copy()

c1 = sorted([v for v in rename_dict.values() if v.startswith("A")])
c2 = sorted([v for v in rename_dict.values() if v.startswith("B")])
all_intensity = c1 + c2

# Force float
for col in all_intensity:
    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
df[all_intensity] = df[all_intensity].replace([0, np.nan], 1.0)

# === CLEAN PROTEIN INFO ===
pg_col = [c for c in df.columns if "Protein" in c and "Group" in c or c.startswith("PG.")][0]
df["PG"] = df[pg_col].astype(str).str.split(";").str[0]

name_col = [c for c in df.columns if "ProteinNames" in c or "Gene" in c]
df["Name"] = df[name_col[0]].astype(str).str.split(";").str[0] if name_col else "Unknown"

# === SPECIES FROM PROTEIN NAME ===
st.subheader("Species Assignment")
species_options = {
    "HUMAN": ["HUMAN", "HOMO", "HSA"],
    "MOUSE": ["MOUSE", "MUS", "MMU"],
    "RAT":   ["RAT", "RATTUS"],
    "ECOLI": ["ECOLI", "ESCHERICHIA"],
    "YEAST": ["YEAST", "SACCHA"],
    "BOVIN": ["BOVIN", "BOVINE"],
    "Other": []
}

selected = st.multiselect("Which species are in your sample?", list(species_options.keys()), default=["HUMAN", "ECOLI"])

lookup = {}
for sp in selected:
    for kw in species_options[sp]:
        lookup[kw] = sp

def assign_species(name):
    if pd.isna(name): return "Other"
    up = str(name).upper()
    for kw, sp in lookup.items():
        if kw in up:
            return sp
    return "Other"

df["Species"] = df["Name"].apply(assign_species)

# === FILTER OUT ALL-IMPUTED PROTEINS ===
before = len(df)
df = df[df[all_intensity].ne(1.0).any(axis=1)]
after = len(df)
st.info(f"Removed {before - after:,} proteins with no real quantification (all values = 1.0)")

# === FINAL DATAFRAME ===
final_cols = ["PG", "Name", "Species"] + all_intensity
df_final = df[final_cols].copy()

# === FINAL RESULT ===
st.success(f"Final dataset: **{len(df_final):,} proteins**")

colA, colB = st.columns(2)
with colA:
    st.subheader("Condition A")
    st.write(" | ".join(c1))
with colB:
    st.subheader("Condition B")
    st.write(" | ".join(c2))

# === SPECIES COUNT (CLEAN) ===
st.subheader("Proteins per Species")
species_table = df_final["Species"].value_counts().reset_index()
species_table.columns = ["Species", "Protein Count"]
st.table(species_table)

# === SIMPLE CLEAN PREVIEW ===
st.subheader("Data Preview")
st.dataframe(df_final.head(5), use_container_width=True)

# === SAVE ===
st.session_state.prot_df = df_final
st.session_state.prot_c1 = c1
st.session_state.prot_c2 = c2

st.success("Protein data ready!")

if st.button("Go to Protein Analysis", type="primary", use_container_width=True):
    st.switch_page("pages/3_Protein_Analysis.py")

restart_button()
