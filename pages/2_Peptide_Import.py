# pages/2_Peptide_Import.py
import streamlit as st
import pandas as pd
import io
from shared import restart_button

def ss(key, default=None):
    if key not in st.session_state:
        return st.session_state[key]
    return default

st.set_page_config(page_title="Peptide Import", layout="wide")

st.markdown("""
<style>
    .header {background:linear-gradient(90deg,#E71316,#A6192E); padding:20px 40px; color:white; margin:-80px -80px 40px;}
    .header h1,.header p {margin:0;}
    .stButton>button {background:#E71316 !important; color:white !important;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>DIA Proteomics Pipeline</h1><p>Peptide Import</p></div>', unsafe_allow_html=True)

# === UPLOAD ONCE, KEEP FOREVER ===
if "uploaded_peptide_bytes" not in st.session_state:
    st.markdown("### Upload Peptide File")
    uploaded = st.file_uploader("CSV / TSV / TXT", type=["csv","tsv","txt"])
    if uploaded:
        st.session_state.uploaded_peptide_bytes = uploaded.getvalue()
        st.session_state.uploaded_peptide_name = uploaded.name
        st.rerun()
    else:
        restart_button()
        st.stop()
else:
    st.success(f"Peptide file ready: **{st.session_state.uploaded_peptide_name}**")

# === LOAD FROM BYTES ===
@st.cache_data(show_spinner="Loading peptide data...")
def load_peptide_data(_bytes):
    text = _bytes.decode("utf-8", errors="replace")
    if text.startswith("\ufeff"):
        text = text[1:]
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")

df_raw = load_peptide_data(st.session_state.uploaded_peptide_bytes)

st.write(f"**{len(df_raw):,}** peptides × **{len(df_raw.columns)}** columns")

# === INTENSITY + REPLICATES (same as protein) ===
# ... copy the exact same block from protein page ...

# === SAVE FINAL DATA ===
st.session_state.pept_df = df
st.session_state.pept_c1 = c1
st.session_state.pept_c2 = c2

st.success("Peptide data saved — ready for analysis")

if st.button("Go to Peptide Analysis", type="primary", use_container_width=True):
    st.switch_page("pages/4_Peptide_Analysis.py")

restart_button()
