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

# === LOAD DATA ===
@st.cache_data
def load_df(b):
    text = b.decode("utf-8", errors="replace")
    if text.startswith("\ufeff"): text = text[1:]
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")

df_data = load_df(st.session_state.protein_bytes)   # or peptide_bytes
df_meta = load_df(st.session_state.metadata_bytes)

# === AUTO-RENAME USING METADATA (EXACT MATCH ON "Run Label") ===
rename_dict = {}
correction_dict = {}

for _, row in df_meta.iterrows():
    run_label = str(row["Run Label"]).strip()
    condition = str(row["Condition"]).strip()
    replicate = str(row["Replicate"]).strip()
    new_col_name = f"{condition}{replicate}"  # e.g., A1, B2
    
    factor = float(row.get("Quantity Correction Factor", 1.0))
    
    if run_label in df_data.columns:
        rename_dict[run_label] = new_col_name
        correction_dict[new_col_name] = factor
    else:
        st.warning(f"Run Label not found in data: {run_label}")

if not rename_dict:
    st.error("No columns matched! Check that 'Run Label' values exactly match column names in your data file.")
    st.stop()

# Apply renaming
df_data = df_data.rename(columns=rename_dict)

# Extract condition groups
all_new_cols = list(rename_dict.values())
cond_a = sorted([c for c in all_new_cols if c.startswith("A")])
cond_b = sorted([c for c in all_new_cols if c.startswith("B")])

if not cond_a or not cond_b:
    st.error("Could not find both Condition A and B replicates.")
    st.stop()

st.success(f"Automatically renamed {len(rename_dict)} columns using metadata")
st.write("→ Condition A:", ", ".join(cond_a))
st.write("→ Condition B:", ", ".join(cond_b))

# === OPTIONAL: Apply correction factors ===
if st.checkbox("Apply Quantity Correction Factors", value=True):
    for col, factor in correction_dict.items():
        if factor != 1.0 and col in df_data.columns:
            df_data[col] = df_data[col] * factor
    st.info("Correction factors applied")

# === Replace 0 and NaN with 1.0 ===
df_data[cond_a + cond_b] = df_data[cond_a + cond_b].replace([0, np.nan], 1.0)

# === Save to session ===
st.session_state.prot_df = df_data
st.session_state.prot_c1 = cond_a
st.session_state.prot_c2 = cond_b

st.success("Protein data ready with metadata!")

if st.button("Go to Protein Analysis", type="primary", use_container_width=True):
    st.switch_page("pages/3_Protein_Analysis.py")

restart_button()
