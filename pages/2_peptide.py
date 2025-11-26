# pages/2_Peptide_Import.py
import streamlit as st
import pandas as pd
import io
from shared import restart_button

def ss(key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

st.set_page_config(page_title="Peptide Import", layout="wide")

st.markdown("""
<style>
    .header {background:linear-gradient(90deg,#E71316,#A6192E); padding:20px 40px; color:white; margin:-80px -80px 40px;}
    .header h1,.header p {margin:0;}
    .stButton>button {background:#E71316 !important; color:white !important;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>DIA Proteomics Pipeline</h1><p><p>Peptide Import</p></div>', unsafe_allow_html=True)

# === FILE UPLOAD THAT NEVER DIES ===
uploaded_file = ss("uploaded_peptide_file")

if uploaded_file is None:
    st.markdown("### Upload Peptide File")
    uploaded_file = st.file_uploader("CSV / TSV / TXT", type=["csv","tsv","txt"], key="pept_uploader")
    if uploaded_file:
        ss("uploaded_peptide_file", uploaded_file)
        st.rerun()
else:
    st.success(f"Peptide file ready: **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")

# === LOAD DATA ===
@st.cache_data(show_spinner="Loading peptide data...")
def load_peptide_data(file_obj):
    bytes_data = file_obj.getvalue()
    text = bytes_data.decode("utf-8", errors="replace")
    if text.startswith("\ufeff"):
        text = text[1:]
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")

df_raw = load_peptide_data(uploaded_file)

st.write(f"**{len(df_raw):,}** peptides × **{len(df_raw.columns)}** columns")

# === DETECT INTENSITY COLUMNS ===
intensity_cols = []
for col in df_raw.columns:
    cleaned = pd.to_numeric(df_raw[col].astype(str).str.replace(r"[,\#NUM!]", "", regex=True), errors='coerce')
    if cleaned.notna().mean() > 0.3:
        df_raw[col] = cleaned
        intensity_cols.append(col)

if not intensity_cols:
    st.error("No quantitative columns found")
    st.stop()

# === REPLICATE ASSIGNMENT ===
st.markdown("### Assign Replicates (must be equal count)")
rows = [{"Column": c, "A": True, "B": False} for c in intensity_cols]
edited = st.data_editor(
    pd.DataFrame(rows),
    column_config={
        "Column": st.column_config.TextColumn(disabled=True),
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
    st.error(f"Must have equal number of replicates (A={len(a_cols)}, B={len(b_cols)})")
    st.stop()

# === RENAME REPLICATES ===
n = len(a_cols)
rename_map = {a_cols[i]: f"A{i+1}" for i in range(n)}
rename_map.update({b_cols[i]: f"B{i+1}" for i in range(n)})
df = df_raw.rename(columns=rename_map).copy()

c1 = [f"A{i+1}" for i in range(n)]
c2 = [f"B{i+1}" for i in range(n)]

st.success(f"Renamed → **A**: {', '.join(c1)} | **B**: {', '.join(c2)}")

# === SAVE FINAL DATA ===
ss("pept_df", df)
ss("pept_c1", c1)
ss("pept_c2", c2)

st.success("Peptide data fully processed and permanently saved")

# === GO TO ANALYSIS ===
st.markdown("---")
if st.button("Go to Peptide Analysis", type="primary", use_container_width=True):
    st.switch_page("pages/4_Peptide_Analysis.py")

restart_button()
