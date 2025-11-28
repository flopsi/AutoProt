# pages/1_Import.py
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_tags import st_tags, st_tags_sidebar
import re

st.set_page_config(page_title="Import", layout="wide")
st.title("Proteomics Data Import — Professional Mode (Schessner et al., 2022)")

st.markdown("""
**Two import modes — your choice:**
- **Metadata-driven** — for complex, multi-condition experiments
- **Direct column selection** — fast, simple, no metadata needed
**Exactly as used at Max Planck Institute**
""")

# === UPLOAD ===
uploaded_prot = st.file_uploader("Upload Intensity Table (wide format)", type=["csv", "tsv", "txt"])
if not uploaded_prot:
    st.info("Please upload either protein or peptide file to continue.")
    st.stop()

if not uploaded_prot:
    st.info("Please upload your intensity table.")
    st.stop()

@st.cache_data  # 👈 Add the caching decorator
def load(file):
    df = pd.read_csv(file, sep=None, engine="python", header=0,index_col=0)
    return df

df_prot=load(uploaded_prot)
st.success(f"Loaded: {len(df_prot):,} features × {len(df_prot.columns):,} columns")
st.dataframe(df_prot.head(5))

proteomes = st_tags(label="#Type the proteomes in your sample:",
                    text = "Press enter to add more",
                    value = ["HUMAN","YEAST","ECOLI","RAT"],
                    suggestions = ["TEST"],
                    maxtags=5,
                    key="proteomes")


if proteomes:
    pattern = "|".join(map(re.escape, proteomes))
    mask = df_prot["PG.ProteinNames"].astype(str).str.contains(proteomes, case=False, na=False)
    df_filtered = df[mask]
else:
    df_filtered = df_prot
                    
st.write("### Results:")
st.write(proteomes)
