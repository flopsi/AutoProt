# pages/2_Peptide_Import.py

            
import streamlit as st
import pandas as pd
import io
from shared import restart_button, debug
# At the very bottom of both pages — replace your debug log section with this:
if st.session_state.get("debug_log"):
    with st.expander("Debug Log", expanded=False):
        for entry in st.session_state.debug_log:
            if isinstance(entry, tuple):
                line, data = entry
                st.markdown(line, unsafe_allow_html=True)
                if data is not None:
                    st.code(data)
            else:
                # Old format (just string)
                st.markdown(entry, unsafe_allow_html=True)
        if st.button("Clear Debug Log"):
            st.session_state.debug_log = []
            st.rerun()
def ss(key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

st.set_page_config(page_title="Peptide Import", layout="wide")
debug("Peptide Import page started")

st.markdown("""
<style>
    :root {--red:#E71316; --darkred:#A6192E;}
    .header {background:linear-gradient(90deg,var(--red),var(--darkred)); padding:20px 40px; color:white; margin:-80px -80px 40px;}
    .header h1,.header p {margin:0;}
    .module-header {background:linear-gradient(90deg,var(--red),var(--darkred)); padding:30px; border-radius:12px; color:white;}
    .stButton>button {background:var(--red)!important; color:white!important;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>DIA Proteomics Pipeline</h1><p>Peptide-Level Import</p></div>', unsafe_allow_html=True)
st.markdown('<div class="module-header"><h2>Peptide Data Import</h2><p>Auto-detect peptide sequence • Equal replicates • Species detection</p></div>', unsafe_allow_html=True)

# Restore
if ss("pept_df") is not None and not ss("reconfig_pept", False):
    df = ss("pept_df")
    c1, c2 = ss("pept_c1"), ss("pept_c2")
    seq_col = ss("pept_seq_col")
    st.success("Peptide data restored")
    st.write(f"**Sequence column**: `{seq_col}`")
    st.metric("A", ", ".join(c1))
    st.metric("Condition B", ", ".join(c2))
    if st.button("Reconfigure"):
        ss("reconfig_pept", True)
        st.rerun()
    restart_button()
    st.stop()

if ss("reconfig_pept", False):
    st.warning("Reconfiguring...")

uploaded = st.file_uploader("Upload Peptide File", type=["csv","tsv","txt"], key="pept_up")
if not uploaded: st.stop()

@st.cache_data
def load(f):
    text = f.getvalue().decode("utf-8", errors="replace")
    if text.startswith("\ufeff"): text = text[1:]
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")

df_raw = load(uploaded)

# Same intensity detection...
intensity = []
for c in df_raw.columns:
    cleaned = pd.to_numeric(df_raw[c].astype(str).str.replace(r"[,\#NUM!]", "", regex=True), errors='coerce')
    if cleaned.notna().mean() > 0.3:
        df_raw[c] = cleaned
        intensity.append(c)

edited = st.data_editor(
    pd.DataFrame([{"Column": c, "A": True, "B": False} for c in intensity]),
    column_config={"Column": st.column_config.TextColumn(disabled=True),
                   "A": st.column_config.CheckboxColumn("Condition A"),
                   "B": st.column_config.CheckboxColumn("Condition B")},
    hide_index=True, use_container_width=True, num_rows="fixed"
)

a = edited[edited["A"]]["Column"].tolist()
b = edited[edited["B"]]["Column"].tolist()
if len(a) != len(b):
    st.error("Equal replicates required!")
    st.stop()

n = len(a)
df = df_raw.rename(columns={a[i]: f"A{i+1}" for i in range(n)} | {b[i]: f"B{i+1}" for i in range(n)}).copy()
c1, c2 = [f"A{i+1}" for i in range(n)], [f"B{i+1}" for i in range(n)]

# Peptide sequence column
seq_candidates = [c for c in df.columns if any(k in c.lower() for k in ["sequence","peptide","seq","stripped"])]
seq_col = st.selectbox("Peptide Sequence Column", seq_candidates)
if st.button("Set as Index"):
    df = df.set_index(seq_col)
    st.rerun()

# Species (same as protein)
species_list = ["HUMAN","MOUSE","RAT","ECOLI","BOVIN","YEAST"]
sp_col = next((c for c in df.columns if c not in c1+c2 and df[c].astype(str).str.upper().str.contains("|".join(species_list)).any()), "Not found")

# Save
ss("pept_df", df)
ss("pept_c1", c1)
ss("pept_c2", c2)
ss("pept_seq_col", seq_col)
ss("reconfig_pept", False)

st.success("Peptide data ready!")
restart_button()

# Debug log
if st.session_state.get("debug_log"):
    with st.expander("Debug Log"):
        for line, data in st.session_state.debug_log:
            st.markdown(line, unsafe_allow_html=True)
            if data: st.code(data)
