# pages/1_Protein_Import.py
import streamlit as st
import pandas as pd
import io
import numpy as np  # Required for .ne(), .any(), etc.

# Import your shared restart function (make sure this file exists!)
from shared import restart_button

# === FULLY RESET SESSION STATE ON RESTART ===
def clear_all_session():
    keys_to_remove = [
        "protein_bytes", "metadata_bytes", "protein_name", "metadata_name",
        "prot_df", "prot_c1", "prot_c2"
    ]
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]

st.set_page_config(page_title="Protein Import", layout="wide")

# === STYLING ===
st.markdown("""
<style>
    .header {background:linear-gradient(90deg,#E71316,#A6192E); padding:20px 40px; color:white; margin:-80px -80px 40px;}
    .header h1,.header p {margin:0;}
    .stButton>button {background:#E71316 !important; color:white !important;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>DIA Proteomics Pipeline</h1><p>Protein Import + Metadata</p></div>', unsafe_allow_html=True)

# === FILE UPLOAD ===
col1, col2 = st.columns(2)

with col1:
    if "protein_bytes" not in st.session_state:
        uploaded_prot = st.file_uploader("Upload Wide-Format Protein File", type=["csv", "tsv", "txt"])
        if uploaded_prot:
            st.session_state.protein_bytes = uploaded_prot.getvalue()
            st.session_state.protein_name = uploaded_prot.name
            st.rerun()
    else:
        st.success(f"Protein: **{st.session_state.protein_name}**")

with col2:
    if "metadata_bytes" not in st.session_state:
        uploaded_meta = st.file_uploader("Upload Metadata File (metadata.tsv)", type=["tsv", "csv", "txt"])
        if uploaded_meta:
            st.session_state.metadata_bytes = uploaded_meta.getvalue()
            st.session_state.metadata_name = uploaded_meta.name
            st.rerun()
    else:
        st.success(f"Metadata: **{st.session_state.metadata_name}**")

# Wait for both files
if "protein_bytes" not in st.session_state or "metadata_bytes" not in st.session_state:
    st.info("Please upload both protein and metadata files.")
    if st.button("Restart / Clear All"):
        clear_all_session()
        st.rerun()
    st.stop()

# === LOAD DATA ===
@st.cache_data(show_spinner="Loading files...")
def load_dataframe(bytes_data):
    text = bytes_data.decode("utf-8", errors="replace")
    if text.startswith("\ufeff"):
        text = text[1:]
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")

df_raw = load_dataframe(st.session_state.protein_bytes)
df_meta = load_dataframe(st.session_state.metadata_bytes)

# === METADATA MATCHING (substring in column name) ===
rename_dict = {}
used_columns = set()

for _, row in df_meta.iterrows():
    run_label = str(row["Run Label"]).strip()
    condition = str(row["Condition"]).strip()
    replicate = str(row["Replicate"]).strip()
    new_name = f"{condition}{replicate}"

    matches = [c for c in df_raw.columns if run_label in str(c)]
    
    if len(matches) == 0:
        st.warning(f"Run Label not found in headers: `{run_label}`")
        continue
    if len(matches) > 1:
        st.error(f"Multiple columns contain `{run_label}`: {matches}")
        st.stop()
    
    col = matches[0]
    if col in used_columns:
        st.error(f"Column `{col}` matched more than once!")
        st.stop()
    
    rename_dict[col] = new_name
    used_columns.add(col)

if not rename_dict:
    st.error("No intensity columns were matched using metadata!")
    st.stop()

# Apply renaming
df = df_raw.rename(columns=rename_dict).copy()

# Extract replicate columns
c1 = sorted([name for name in rename_dict.values() if name.startswith("A")])
c2 = sorted([name for name in rename_dict.values() if name.startswith("B")])
all_intensity_cols = c1 + c2

# Convert to float and replace missing/imputed with 1.0
for col in all_intensity_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
df[all_intensity_cols] = df[all_intensity_cols].replace([0, np.nan], 1.0)

# === EXTRACT FIRST PROTEIN ACCESSION & NAME ===
pg_cols = [c for c in df.columns if "Protein" in c and ("Group" in c or "Groups" in c) or c.startswith("PG.")]
if not pg_cols:
    st.error("Could not find ProteinGroups column (PG.ProteinGroups)")
    st.stop()
df["PG"] = df[pg_cols[0]].astype(str).str.split(";").str[0]

name_cols = [c for c in df.columns if "ProteinNames" in c or "Gene" in c]
df["Name"] = df[name_cols[0]].astype(str).str.split(";").str[0] if name_cols else "Unknown"

# === SPECIES DETECTION FROM PROTEIN NAME ===
st.subheader("Species Assignment")
species_keywords = {
    "HUMAN": ["HUMAN", "HOMO", "HSA"],
    "MOUSE": ["MOUSE", "MUS", "MMU"],
    "RAT":   ["RAT", "RATTUS", "RNO"],
    "ECOLI": ["ECOLI", "ESCHERICHIA"],
    "YEAST": ["YEAST", "SACCHA", "CEREVISIAE"],
    "BOVIN": ["BOVIN", "BOVINE", "BOS"],
}

selected = st.multiselect(
    "Which species are present?",
    options=list(species_keywords.keys()),
    default=["HUMAN", "ECOLI","YEAST"]
)

species_lookup = {}
for sp in selected:
    for kw in species_keywords[sp]:
        species_lookup[kw] = sp

def get_species(name):
    if pd.isna(name):
        return "Other"
    name_up = str(name).upper()
    for kw, sp in species_lookup.items():
        if kw in name_up:
            return sp
    return "Other"

df["Species"] = df["Name"].apply(get_species)

# === REMOVE FULLY IMPUTED PROTEINS ===
before = len(df)
df = df[df[all_intensity_cols].ne(1.0).any(axis=1)]
after = len(df)
st.info(f"Removed {before - after:,} proteins with only imputed values (all = 1.0)")

# === FINAL CLEAN TABLE ===
final_columns = ["PG", "Name", "Species"] + all_intensity_cols
df_final = df[final_columns].copy()

# === DISPLAY RESULTS ===
st.success(f"Final dataset: **{len(df_final):,} proteins** × **{len(df_final.columns)} columns**")

colA, colB = st.columns(2)
with colA:
    st.subheader("Condition A")
    st.code(" | ".join(c1), language=None)
with colB:
    st.subheader("Condition B")
    st.code(" | ".join(c2), language=None)

st.write("**Proteins per species:**")
for species, count in df_final["Species"].value_counts().items():
    st.write(f"• **{species}**: {count:,}")

st.subheader("Data Preview")
st.dataframe(df_final.head(12), use_container_width=True)



# === SAVE TO SESSION STATE (GUARANTEED TO BE USED IN ANALYSIS) ===
st.session_state.prot_df = df_final
st.session_state.prot_c1 = c1
st.session_state.prot_c2 = c2

st.success("Protein data successfully processed and ready!")

if st.button("Go to Protein Analysis", type="primary", use_container_width=True):
    st.switch_page("pages/3_Protein_Analysis.py")

# Add this at the end of any page
st.markdown("""
<style>
    .restart-fixed {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 999;
        background: #E71316;
        color: white;
        padding: 15px 30px;
        border-radius: 10px;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        text-align: center;
    }
</style>
<div class="restart-fixed">
    🔄 Restart Entire Analysis
</div>
""", unsafe_allow_html=True)


#Full Restart Button
if st.button("RESTART", key="restart_fixed", help="Clear all data and start over"):
    st.cache_data.clear()
    st.cache_resource.clear()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
