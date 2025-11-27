# pages/1_peptide_Import.py
import streamlit as st
import pandas as pd
import io
import numpy as np
from shared import restart_button

def clear_all_session():
    keys = ["peptide_bytes", "metadata_bytes", "peptide_name", "metadata_name",
            "pep_df", "pep_c1", "pep_c2", "pep_seq_col", "pep_pg_col"]
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]

st.set_page_config(page_title="Peptide Import", layout="wide")

# === STYLING ===
st.markdown("""
<style>
    .header {background:linear-gradient(90deg,#E71316,#A6192E); padding:20px 40px; color:white; margin:-80px -80px 40px;}
    .header h1,.header p {margin:0;}
    .stButton>button {background:#E71316 !important; color:white !important;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="header"><h1>DIA Proteomics Pipeline</h1><p>Peptide Import + Metadata</p></div>', unsafe_allow_html=True)

# === FILE UPLOAD ===
col1, col2 = st.columns(2)
with col1:
    if "peptide_bytes" not in st.session_state:
        uploaded_pep = st.file_uploader("Upload Wide-Format Peptide File","", type=["csv", "tsv", "txt"])
        if uploaded_pep:
            st.session_state.peptide_bytes = uploaded_pep.getvalue()
            st.session_state.peptide_name = uploaded_pep.name
            st.rerun()
    else:
        st.success(f"Peptide: **{st.session_state.peptide_name}**")
with col2:
    if "metadata_bytes" not in st.session_state:
        uploaded_meta = st.file_uploader("Upload Metadata File (metadata.tsv)", type=["tsv", "csv", "txt"])
        if uploaded_meta:
            st.session_state.metadata_bytes = uploaded_meta.getvalue()
            st.session_state.metadata_name = uploaded_meta.name
            st.rerun()
    else:
        st.success(f"Metadata: **{st.session_state.metadata_name}**")

if "peptide_bytes" not in st.session_state or "metadata_bytes" not in st.session_state:
    st.info("Please upload both files.")
    if st.button("Restart / Clear All"):
        clear_all_session()
        st.rerun()
    st.stop()

# === LOAD DATA ===
@st.cache_data(show_spinner="Loading files...")
def load_dataframe(bytes_data):
    text = bytes_data.decode("utf-8", errors="replace")
    if text.startswith("\ufeff"): text = text[1:]
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")

df_raw = load_dataframe(st.session_state.peptide_bytes)
df_meta = load_dataframe(st.session_state.metadata_bytes)

# === METADATA MATCHING ===
rename_dict = {}
used_columns = set()
for _, row in df_meta.iterrows():
    run_label = str(row["Run Label"]).strip()
    condition = str(row["Condition"]).strip()
    replicate = str(row["Replicate"]).strip()
    new_name = f"{condition}{replicate}"
    matches = [c for c in df_raw.columns if run_label in str(c)]
    if not matches:
        st.warning(f"Run Label not found: `{run_label}`")
        continue
    if len(matches) > 1:
        st.error(f"Multiple matches for `{run_label}`: {matches}")
        st.stop()
    col = matches[0]
    if col in used_columns:
        st.error(f"Column `{col}` matched twice!")
        st.stop()
    rename_dict[col] = new_name
    used_columns.add(col)

if not rename_dict:
    st.error("No intensity columns matched!")
    st.stop()

df = df_raw.rename(columns=rename_dict).copy()
c1 = sorted([name for name in rename_dict.values() if name.startswith("A")])
c2 = sorted([name for name in rename_dict.values() if name.startswith("B")])
all_intensity_cols = c1 + c2

# Convert to numeric
for col in all_intensity_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
df[all_intensity_cols] = df[all_intensity_cols].replace([0, np.nan], 1.0)

# === AUTO-DETECT PEPTIDE SEQUENCE COLUMN (>90% end with K or R before _) ===
def detect_peptide_sequence_column(df):
    candidates = []
    for col in df.columns:
        if df[col].dtype != "object": continue
        sample = df[col].dropna().astype(str).head(1000)
        if sample.empty: continue
        # Check pattern: ends with K or R before optional modification like _
        pattern = r'[KR](?=[_\.])|[KR]$'
        matches = sample.str.contains(pattern, regex=True)
        if matches.mean() > 0.90:
            candidates.append((col, matches.mean()))
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    return None

auto_seq_col = detect_peptide_sequence_column(df)

# === USER COLUMN ASSIGNMENT ===
st.subheader("Column Assignment")

# Default selections
default_seq = auto_seq_col or df.columns[0]
pg_cols = [c for c in df.columns if any(x in c.lower() for x in ["protein", "pg.", "leading", "fasta"])]
default_pg = pg_cols[0] if pg_cols else df.columns[1]

rows = []
for col in df.columns:
    preview = " | ".join(df[col].dropna().astype(str).unique()[:3])
    rows.append({
        "Rename": col,
        "Peptide Sequence": col == default_seq,
        "Protein Group": col == default_pg,
        "Original Name": col,
        "Preview": preview,
        "Type": "Intensity" if col in all_intensity_cols else "Metadata"
    })

edited = st.data_editor(
    pd.DataFrame(rows),
    column_config={
        "Rename": st.column_config.TextColumn("Rename"),
        "Peptide Sequence": st.column_config.CheckboxColumn("Peptide Sequence"),
        "Protein Group": st.column_config.CheckboxColumn("Protein Group"),
        "Original Name": st.column_config.TextColumn("Original", disabled=True),
        "Preview": st.column_config.TextColumn("Preview", disabled=True),
        "Type": st.column_config.TextColumn("Type", disabled=True),
    },
    disabled=["Original Name", "Preview", "Type"],
    hide_index=True,
    use_container_width=True,
    key="pep_col_table"
)

# Extract selections
seq_checked = edited[edited["Peptide Sequence"]]
pg_checked = edited[edited["Protein Group"]]

seq_cols = seq_checked["Original Name"].tolist()
pg_cols = pg_checked["Original Name"].tolist()

errors = []
if len(seq_cols) != 1: errors.append("Select exactly 1 Peptide Sequence column")
if len(pg_cols) != 1: errors.append("Select exactly 1 Protein Group column")
if errors:
    for e in errors: st.error(e)
    st.stop()

pep_seq_col = seq_cols[0]
pep_pg_col = pg_cols[0]

# Rename
rename_map = {}
for _, row in edited.iterrows():
    new = row["Rename"].strip()
    if new and new != row["Original Name"]:
        rename_map[row["Original Name"]] = new

df_final = df.rename(columns=rename_map).copy()

# === FINAL CLEANUP ===
df_final["Sequence"] = df_final[pep_seq_col]
df_final["PG"] = df_final[pep_pg_col].astype(str).str.split(";").str[0]

# Species from protein name
species_keywords = {
    "HUMAN": ["HUMAN", "HOMO"], "ECOLI": ["ECOLI"], "YEAST": ["YEAST", "SACCHA"]
}
def get_species(pg):
    if pd.isna(pg): return "Other"
    pg_up = str(pg).upper()
    for sp, kws in species_keywords.items():
        if any(kw in pg_up for kw in kws):
            return sp
    return "Other"
df_final["Species"] = df_final["PG"].apply(get_species)

final_cols = ["Sequence", "PG", "Species"] + all_intensity_cols
df_final = df_final[final_cols].copy()

# === SAVE TO SESSION ===
st.session_state.pep_df = df_final
st.session_state.pep_c1 = c1
st.session_state.pep_c2 = c2
st.session_state.pep_seq_col = "Sequence"
st.session_state.pep_pg_col = "PG"

# === DISPLAY ===
st.success(f"Final dataset: **{len(df_final):,} peptides**")
colA, colB = st.columns(2)
with colA: st.subheader("Condition A"); st.code(" | ".join(c1))
with colB: st.subheader("Condition B"); st.code(" | ".join(c2))

st.write("**Peptides per species:**")
for sp, count in df_final["Species"].value_counts().items():
    st.write(f"• **{sp}**: {count:,}")

st.subheader("Data Preview")
st.dataframe(df_final.head(12), use_container_width=True)

if st.button("Go to Peptide Analysis", type="primary", use_container_width=True):
    st.switch_page("pages/3_Peptide_Analysis.py")

if st.button("Restart Everything"):
    clear_all_session()
    st.rerun()
