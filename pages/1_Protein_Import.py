# pages/1_Protein_Import.py
import streamlit as st
import pandas as pd
import io
from shared import restart_button

def ss(key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

st.set_page_config(page_title="Protein Import", layout="wide")

st.markdown("""
<style>
    .header {background:linear-gradient(90deg,#E71316,#A6192E); padding:20px 40px; color:white; margin:-80px -80px 40px;}
    .header h1,.header p {margin:0;}
    .stButton>button {background:#E71316 !important; color:white !important;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>DIA Proteomics Pipeline</h1><p>Protein Import</p></div>', unsafe_allow_html=True)

# === UPLOAD ONCE, KEEP FOREVER ===
if "uploaded_protein_bytes" not in st.session_state:
    st.markdown("### Upload Protein File")
    uploaded = st.file_uploader("CSV / TSV / TXT", type=["csv","tsv","txt"])
    if uploaded:
        st.session_state.uploaded_protein_bytes = uploaded.getvalue()
        st.session_state.uploaded_protein_name = uploaded.name
        st.rerun()
    else:
        restart_button()
        st.stop()
else:
    st.success(f"Protein file ready: **{st.session_state.uploaded_protein_name}**")

# === LOAD FROM BYTES (safe ===
@st.cache_data(show_spinner="Loading protein data...")
def load_protein_data(_bytes):
    text = _bytes.decode("utf-8", errors="replace")
    if text.startswith("\ufeff"):
        text = text[1:]
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")

df_raw = load_protein_data(st.session_state.uploaded_protein_bytes)

st.write(f"**{len(df_raw):,}** proteins × **{len(df_raw.columns)}** columns")

# === INTENSITY COLUMNS ===
intensity_cols = []
for col in df_raw.columns:
    cleaned = pd.to_numeric(df_raw[col].astype(str).str.replace(r"[,\#NUM!]", "", regex=True), errors='coerce')
    if cleaned.notna().mean() > 0.3:
        df_raw[col] = cleaned
        intensity_cols.append(col)

if not intensity_cols:
    st.error("No quantitative columns found")
    st.stop()

# === REPLICATES ===
st.markdown("### Assign Replicates (must be equal)")
rows = [{"Column": c, "A": True, "B": False} for c in intensity_cols]
edited = st.data_editor(
    pd.DataFrame(rows),
    column_config={
        "Column": st.column_config.TextColumn(disabled=True),
        "A": st.column_config.CheckboxColumn("Condition A"),
        "B": st.column_config.CheckboxColumn("Condition B"),
    },
    hide_index=True, use_container_width=True, num_rows="fixed"
)

a_cols = edited[edited["A"]]["Column"].tolist()
b_cols = edited[edited["B"]]["Column"].tolist()

if len(a_cols) != len(b_cols) or len(a_cols) == 0:
    st.error("Must have equal replicates")
    st.stop()

n = len(a_cols)
df = df_raw.rename(columns={a_cols[i]: f"A{i+1}" for i in range(n)} | {b_cols[i]: f"B{i+1}" for i in range(n)}).copy()
c1 = [f"A{i+1}" for i in range(n)]
c2 = [f"B{i+1}" for i in range(n)]

st.success(f"Renamed → A: {', '.join(c1)} | B: {', '.join(c2)}")

# === SAVE FINAL DATA ===
st.session_state.prot_df = df
st.session_state.prot_c1 = c1
st.session_state.prot_c2 = c2

st.success("Protein data saved — ready for analysis")

# === GO TO ANALYSIS ===
if st.button("Go to Protein Analysis", type="primary", use_container_width=True):
    st.switch_page("pages/3_Protein_Analysis.py")

restart_button()
