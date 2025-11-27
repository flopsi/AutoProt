# pages/2_Peptide_Import.py
import streamlit as st
import pandas as pd
import io
import re
from pandas.api.types import is_numeric_dtype

st.set_page_config(page_title="Peptide Import", layout="wide")

# === CSS & HEADER ===
st.markdown("""
<style>
    .header {background: linear-gradient(90deg, #E71316 0%, #A6192E 100%); padding: 20px 40px; color: white; margin: -60px -60px 40px -60px;}
    .header h1 {margin:0; font-size:28px; font-weight:600;}
    .stButton>button {background:#E71316 !important; color:white !important; border:none !important; padding:12px 24px !important; border-radius:6px !important; font-weight:500 !important;}
    .footer {text-align:center; padding:30px; color:#54585A; font-size:12px; border-top:1px solid #E2E3E4; margin-top:60px;}
    .fixed-restart {position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); z-index: 999; width: 300px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>Peptide Level Data Import</h1></div>', unsafe_allow_html=True)

# === CHECK IF DATA EXISTS IN SESSION ===
if "pep_df" in st.session_state and not st.session_state.get("allow_reconfigure_pep", False):
    st.info("Data restored from session state")

    df = st.session_state.pep_df
    c1 = st.session_state.pep_c1
    c2 = st.session_state.pep_c2
    sp_col = st.session_state.pep_sp_col
    pg_col = st.session_state.pep_pg_col

    st.success("Peptide data loaded!")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric("**Condition 1**", f"{len(c1)} reps")
        st.write(", ".join(c1))
    with col2:
        st.metric("**Condition 2**", f"{len(c2)} reps")
        st.write(", ".join(c2))
    with col3:
        if st.button("Reconfigure", key="reconfig_pep"):
            st.session_state["allow_reconfigure_pep"] = True
            st.rerun()

    st.info(f"**Species** → `{sp_col}` | **Protein Group** → `{pg_col}`")

    st.markdown("---")
    st.markdown('<div class="footer"><strong>Proprietary & Confidential</strong><br>© 2024 Thermo Fisher Scientific</div>', unsafe_allow_html=True)

    st.markdown("""<style>.fixed-restart {position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); z-index: 999; width: 300px;}</style><div class="fixed-restart">""", unsafe_allow_html=True)
    if st.button("Restart Analysis", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# === UPLOAD FLOW ===
st.markdown("### Upload Peptide Data")
uploaded_file = st.file_uploader("", type=["csv","tsv","txt"], label_visibility="collapsed")

if not uploaded_file:
    st.markdown('<div class="footer"><strong>Proprietary & Confidential</strong><br>© 2024 Thermo Fisher Scientific</div>', unsafe_allow_html=True)
    st.stop()

# === LOAD & PARSE ===
@st.cache_data
def load_peptides(file):
    content = file.getvalue().decode("utf-8", errors="replace")
    if content.startswith("\ufeff"):
        content = content[1:]
    df = pd.read_csv(io.StringIO(content), sep=None, engine="python", dtype=str)
    
    # Find intensity columns
    intensity_cols = [c for c in df.columns if any(x in c.lower() for x in ["intensity", "lfq", "ibaq"])]
    for col in intensity_cols:
        df[col] = pd.to_numeric(df[col].str.replace(",", ""), errors="coerce")
    
    # Try to extract Species from Protein IDs or separate column
    if "Protein IDs" in df.columns:
        df["Species"] = df["Protein IDs"].str.extract(r";([A-Z]{5})", expand=False)
    if "Species" not in df.columns:
        df["Species"] = "UNKNOWN"
    
    return df, intensity_cols

df_raw, intensity_cols = load_peptides(uploaded_file)
st.success(f"Peptide data imported — {len(df_raw):,} peptides")

# === COLUMN ASSIGNMENT ===
st.subheader("Column Assignment & Renaming")

num_cols = [c for c in df_raw.columns if is_numeric_dtype(df_raw[c])]
default_c1 = num_cols[:len(num_cols)//2]

rows = []
for col in df_raw.columns:
    preview = " | ".join(df_raw[col].dropna().astype(str).unique()[:3])
    rows.append({
        "Rename": col,
        "Cond 1": col in default_c1 and col in intensity_cols,
        "Species": col == "Species",
        "Protein Group": col == "Protein IDs" or col == "Leading razor protein",
        "Original Name": col,
        "Preview": preview,
        "Type": "Intensity" if col in intensity_cols else "Metadata"
    })

df_edit = pd.DataFrame(rows)
edited = st.data_editor(
    df_edit,
    column_config={
        "Rename": st.column_config.TextColumn("Rename"),
        "Cond 1": st.column_config.CheckboxColumn("Cond 1"),
        "Species": st.column_config.CheckboxColumn("Species"),
        "Protein Group": st.column_config.CheckboxColumn("PG"),
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
c1_checked = edited[edited["Cond 1"]]
sp_checked = edited[edited["Species"]]
pg_checked = edited[edited["Protein Group"]]

c1_orig = c1_checked["Original Name"].tolist()
c2_orig = [c for c in intensity_cols if c not in c1_orig]
sp_cols = sp_checked["Original Name"].tolist()
pg_cols = pg_checked["Original Name"].tolist()

errors = []
if len(c1_orig) == 0: errors.append("Cond 1 must have ≥1 replicate")
if len(c2_orig) == 0: errors.append("Cond 2 must have ≥1 replicate")
if len(sp_cols) != 1: errors.append("Select exactly 1 Species column")
if len(pg_cols) != 1: errors.append("Select exactly 1 Protein Group column")
if errors:
    for e in errors: st.error(e)
    st.stop()

sp_col = sp_cols[0]
pg_col = pg_cols[0]

# Rename
rename_map = {}
for _, row in edited.iterrows():
    new = row["Rename"].strip()
    if new and new != row["Original Name"]:
        rename_map[row["Original Name"]] = new

df_final = df_raw.rename(columns=rename_map)
c1 = [rename_map.get(c, c) for c in c1_orig]
c2 = [rename_map.get(c, c) for c in c2_orig]

# === SAVE TO SESSION ===
st.session_state.update({
    "pep_df": df_final,
    "pep_c1": c1,
    "pep_c2": c2,
    "pep_sp_col": sp_col,
    "pep_pg_col": pg_col,
})

# === DISPLAY ===
st.success("All assignments applied!")
col1, col2 = st.columns(2)
with col1:
    st.metric("**Condition 1**", f"{len(c1)} reps")
    st.write(", ".join(c1))
with col2:
    st.metric("**Condition 2**", f"{len(c2)} reps")
    st.write(", ".join(c2))

st.info(f"**Species** → `{sp_col}` | **Protein Group** → `{pg_col}`")

st.markdown("---")
st.markdown('<div class="footer"><strong>Proprietary & Confidential</strong><br>© 2024 Thermo Fisher Scientific</div>', unsafe_allow_html=True)

# Restart button
st.markdown("""<style>.fixed-restart {position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); z-index: 999; width: 300px;}</style><div class="fixed-restart">""", unsafe_allow_html=True)
if st.button("Restart Analysis", use_container_width=True):
    st.session_state.clear()
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)
