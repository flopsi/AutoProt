# pages/1_Protein_Import.py
import streamlit as st
import pandas as pd
import io
from shared import restart_button

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
        uploaded_meta = st.file_uploader("Upload Metadata File", type=["tsv","csv","txt"])
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
    if len(matches) == 0:
        st.warning(f"Not found: `{label}`")
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
    st.error("No columns matched!")
    st.stop()

# Rename intensity columns
df = df_raw.rename(columns=rename_dict).copy()

# Extract A/B columns
c1 = sorted([v for v in rename_dict.values() if v.startswith("A")])
c2 = sorted([v for v in rename_dict.values() if v.startswith("B")])
all_intensity = c1 + c2

# Force float
for col in all_intensity:
    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
df[all_intensity] = df[all_intensity].replace([0, np.nan], 1.0)

# === CLEAN UP COLUMNS ===
# 1. Keep only first Protein Group (PG)
pg_candidates = [c for c in df.columns if "Protein" in c and "Group" in c or c.startswith("PG.")]
if not pg_candidates:
    st.error("No PG.ProteinGroups column found!")
    st.stop()
pg_col = pg_candidates[0]
df["PG"] = df[pg_col].astype(str).str.split(";").str[0]

# 2. Keep only first Protein Name
name_candidates = [c for c in df.columns if "ProteinNames" in c or "Gene" in c]
if name_candidates:
    name_col = name_candidates[0]
    df["ProteinName"] = df[name_col].astype(str).str.split(";").str[0]
else:
    df["ProteinName"] = "Unknown"

# 3. Species
species_candidates = [c for c in df.columns if any(x in str(c) for x in ["Fasta", "Description", "Protein"])]
species_col = st.selectbox("Column for species detection", species_candidates, index=0)
def get_species(x):
    if pd.isna(x): return "Other"
    t = str(x).upper()
    if any(k in t for k in ["HUMAN", "HOMO", "HSA"]): return "HUMAN"
    if any(k in t for k in ["MOUSE", "MUS", "MMU"]): return "MOUSE"
    if any(k in t for k in ["YEAST"]): return "YEAST"
    return "Other"
df["Species"] = df[species_col].apply(get_species)

# === FILTER: Drop rows with all 6 replicates = 1.0 ===
before = len(df)
df = df[df[all_intensity].ne(1.0).any(axis=1)]  # Keep if at least one > 1.0
after = len(df)
st.info(f"Removed {before - after:,} proteins with only imputed values (all 1.0)")

# === FINAL DATAFRAME: Only desired columns ===
final_cols = ["PG", "ProteinName", "Species"] + all_intensity
df_final = df[final_cols].copy()

# Rename ProteinName → more readable
df_final = df_final.rename(columns={"ProteinName": "Name"})

# ===================================
# === PREVIEW ===
# ===================================
st.success(f"Final dataset: **{len(df_final):,} proteins** × **{len(df_final.columns)} columns**")

colA, colB = st.columns(2)
with colA:
    st.subheader("Condition A")
    st.code(" | ".join(c1))
with colB:
    st.subheader("Condition B")
    st.code(" | ".join(c2))

st.write("**Species:**", dict(df_final["Species"].value_counts()))

# Preview with highlighted intensity
preview = df_final.head(12)
def highlight_intensity(val):
    return ['background-color: #d4edda; color: #155724; font-weight: bold' 
            if col in all_intensity else '' for col in preview.columns]
st.dataframe(preview.style.apply(highlight_intensity, axis=0), use_container_width=True)

# === SAVE ===
st.session_state.prot_df = df_final
st.session_state.prot_c1 = c1
st.session_state.prot_c2 = c2

st.success("Protein data ready → Go to Analysis!")

if st.button("Go to Protein Analysis", type="primary", use_container_width=True):
    st.switch_page("pages/3_Protein_Analysis.py")

restart_button()
