# pages/1_Protein_Import.py
import streamlit as st
import pandas as pd
import io
from shared import restart_button, debug
import plotly.express as px

def ss(key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

# Set defaults on first load
ss("prot_pg_col", ss("prot_pg_col", "Not selected yet"))
ss("prot_sp_col", ss("prot_sp_col", "Not found"))


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

# ====================== FINAL SAFE SAVE ======================
debug("Saving final data to session state...")

# Always save these — they are guaranteed to exist
ss("prot_df", df)
ss("prot_c1", c1)
ss("prot_c2", c2)
ss("reconfig_prot", False)

# Safe save for optional columns
ss("prot_pg_col", pg_col if 'pg_col' in locals() and pg_col else "Not selected")
ss("prot_sp_col", sp_col if 'sp_col' in locals() and sp_col != "Not found" else "Not found")

st.success("All data cached successfully!")

# Only save pg_col if user has selected one
if 'pg_col' in locals() and pg_col is not None:
    ss("prot_pg_col", pg_col)
    debug("Saved Protein Group column", pg_col)
else:
    ss("prot_pg_col", "Not selected yet")
    debug("Protein Group column not selected yet")

# Same for species
if 'sp_col' in locals() and sp_col != "Not found":
    ss("prot_sp_col", sp_col)
    debug("Saved species column", sp_col)
else:
    ss("prot_sp_col", "Not found")

# ====================== FINAL: SAVE EVERYTHING PERMANENTLY ======================
st.success("Protein processing complete — data is now permanently saved!")

# These lines make the analysis page work forever
ss("protein_data_ready", True)
ss("prot_final_df", df)                    # Final processed dataframe
ss("prot_final_c1", c1)                    # ["A1", "A2", ...]
ss("prot_final_c2", c2)                    # ["B1", "B2", ...]
ss("prot_final_pg", df.index.name if not isinstance(df.index, pd.RangeIndex) else "None")

# Beautiful plots
col1, col2 = st.columns(2)
with col1:
    st.subheader("Intensity Distribution (log scale)")
    fig1 = px.box(df[c1 + c2].melt(), x="variable", y="value", color="variable", log_y=True, height=500)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Proteins per Species")
    if "Species" in df.columns:
        species_counts = df["Species"].value_counts().reset_index()
        species_counts.columns = ["Species", "Count"]
        fig2 = px.bar(species_counts, x="Species", y="Count", color="Species", height=500)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No species column detected")

# GO TO ANALYSIS BUTTON
st.markdown("---")
if st.button("Go to Protein Exploratory Analysis", type="primary", use_container_width=True):
    st.switch_page("pages/3_Protein_Analysis.py")

restart_button()

st.markdown("""
<div class="footer">
    <strong>Proprietary & Confidential</strong><br>© 2024 Thermo Fisher Scientific
</div>
""", unsafe_allow_html=True)
