# pages/1_Protein_Import.py
import streamlit as st
import pandas as pd
import io
from shared import restart_button, debug

def ss(key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

st.set_page_config(page_title="Protein Import", layout="wide")
debug("Protein Import page started")

st.markdown("""
<style>
    :root {--red:#E71316; --darkred:#A6192E;}
    .header {background:linear-gradient(90deg,var(--red),var(--darkred)); padding:20px 40px; color:white; margin:-80px -80px 40px;}
    .header h1,.header p {margin:0;}
    .module-header {background:linear-gradient(90deg,var(--red),var(--darkred)); padding:30px; border-radius:12px; color:white;}
    .stButton>button {background:var(--red)!important; color:white!important;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>DIA Proteomics Pipeline</h1><p>Protein-Level Import</p></div>', unsafe_allow_html=True)
st.markdown('<div class="module-header"><h2>Protein Data Import</h2><p>Auto-detect species • Equal replicates • Set Protein Group as index</p></div>', unsafe_allow_html=True)

# Restore
if ss("prot_df") is not None and not ss("reconfig_prot", False):
    df = ss("prot_df")
    c1, c2 = ss("prot_c1"), ss("prot_c2")
    st.success("Protein data restored")
    st.metric("Condition A", f"{len(c1)} reps → {', '.join(c1)}")
    st.metric("Condition B", f"{len(c2)} reps → {', '.join(c2)}")
    if st.button("Reconfigure"):
        ss("reconfig_prot", True)
        st.rerun()
    restart_button()
    st.stop()

if ss("reconfig_prot", False):
    st.warning("Reconfiguring — upload the same file")

# Upload
uploaded = st.file_uploader("Upload Protein File", type=["csv","tsv","txt"], key="prot_up")
if not uploaded:
    st.info("Upload file to continue")
    restart_button()
    st.stop()

debug("File uploaded", uploaded.name)

@st.cache_data
def load(f):
    text = f.getvalue().decode("utf-8", errors="replace")
    if text.startswith("\ufeff"): text = text[1:]
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")

df_raw = load(uploaded)
debug("Loaded", f"{df_raw.shape}")

# Intensity columns
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
    st.error("Must have equal replicates!")
    st.stop()

n = len(a)
df = df_raw.rename(columns={a[i]: f"A{i+1}" for i in range(n)} | {b[i]: f"B{i+1}" for i in range(n)}).copy()
c1, c2 = [f"A{i+1}" for i in range(n)], [f"B{i+1}" for i in range(n)]

# Protein group
pg = st.selectbox("Protein Group column", [c for c in df.columns if "protein" in c.lower() or "pg" in c.lower()])
if st.button("Set as Index"):
    df = df.set_index(pg)
    st.rerun()

# Species
species_list = ["HUMAN","MOUSE","RAT","ECOLI","BOVIN","YEAST"]
sp_col = next((c for c in df.columns if c not in c1+c2 and df[c].astype(str).str.upper().str.contains("|".join(species_list)).any()), "Not found")
if sp_col != "Not found":
    df["Species"] = df[sp_col].astype(str).str.upper().apply(lambda x: next((s for s in species_list if s in x), "Other"))

# Save
ss("prot_df", df); ss("prot_c1", c1); ss("prot_c2", c2); ss("reconfig_prot", False)

st.success("Protein data ready!")
st.metric("Condition A", ", ".join(c1))
st.metric("Condition B", ", ".join(c2))

if "Species" in df.columns:
    st.bar_chart(df["Species"].value_counts())

restart_button()

if st.session_state.get("debug_log"):
    with st.expander("Debug Log"):
        for line, data in st.session_state.debug_log:
            st.markdown(line, unsafe_allow_html=True)
            if data: st.code(data)
