# pages/1_protein.py
import streamlit as st
import pandas as pd
import io
import re
from pandas.api.types import is_numeric_dtype

st.set_page_config(page_title="Protein Import", layout="wide")

# ─────────────────────────────────────────────────────────────
# CSS & HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .header {background: linear-gradient(90deg, #E71316 0%, #A6192E 100%); padding: 20px 40px; color: white; margin: -60px -60px 40px -60px;}
    .header h1 {margin:0; font-size:28px; font-weight:600;}
    .card {background:white; border:1px solid #E2E3E4; border-radius:8px; padding:25px; box-shadow:0 2px 4px rgba(0,0,0,0.05); margin-bottom:25px;}
    .upload-area {border:2px dashed #E2E3E4; border-radius:8px; padding:60px 30px; text-align:center; background:#fafafa; cursor:pointer;}
    .upload-area:hover {border-color:#E71316; background:rgba(231,19,22,0.02);}
    .stButton>button {background:#E71316 !important; color:white !important; border:none !important; padding:12px 24px !important; border-radius:6px !important; font-weight:500 !important;}
    .stButton>button:hover {background:#A6192 19 46 !important;}
    .footer {text-align:center; padding:30px; color:#54585A; font-size:12px; border-top:1px solid #E2E3E4; margin-top:60px;}
    .fixed-restart {position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); z-index: 999; width: 300px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>Protein Level Data Import</h1></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HELPER: Display restored data + Reconfigure/Restart buttons
# ─────────────────────────────────────────────────────────────
def show_restored_data():
    df = st.session_state.prot_df
    c1 = st.session_state.prot_c1
    c2 = st.session_state.prot_c2
    sp_col = st.session_state.prot_sp_col
    pg_col = st.session_state.prot_pg_col
    sp_counts = st.session_state.get("prot_sp_counts")

    st.success("Data restored from session state")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric("**Condition 1**", f"{len(c1)} reps")
        st.write(", ".join(c1))
    with col2:
        st.metric("**Condition 2**", f"{len(c2)} reps")
        st.write(", ".join(c2))
    with col3:
        if st.button("Reconfigure", key="reconfig_btn"):
            st.session_state.allow_reconfigure_prot = True
            st.rerun()

    st.info(f"**Species** → `{sp_col}` | **Protein Group** → `{pg_col}`")

    if sp_counts is not None and len(sp_counts) > 0:
        st.markdown("### Proteins by Species & Condition")
        st.dataframe(sp_counts, hide_index=True, use_container_width=True)
        c1c, c2c = st.columns(2)
        with c1c:
            chart = sp_counts[["species", "c1", "c2"]].set_index("species")
            st.bar_chart(chart)
        with c2c:
            st.write("**Summary:**")
            for _, r in sp_counts.iterrows():
                st.write(f"- **{r['species']}**: {r['total']} total | C1: {r['c1']} | C2: {r['c2']} | Both: {r['both']}")

    st.markdown("---")
    st.markdown('<div class="footer"><strong>Proprietary & Confidential</strong><br>© 2024 Thermo Fisher Scientific</div>', unsafe_allow_html=True)

    # Fixed Restart Button
    st.markdown('<div class="fixed-restart">', unsafe_allow_html=True)
    if st.button("Restart Analysis", use_container_width=True, type="primary"):
        for key in list(st.session_state.keys()):
            if key.startswith("prot_") or key == "allow_reconfigure_prot":
                del st.session_state[key]
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────
# MAIN LOGIC: Restore vs Configure
# ─────────────────────────────────────────────────────────────
if "prot_df" in st.session_state and not st.session_state.get("allow_reconfigure_prot", False):
    show_restored_data()

# If user wants to reconfigure → fall through to upload/config flow
if st.session_state.get("allow_reconfigure_prot", False):
    st.warning("Reconfiguring protein data... Upload the same file again or modify settings.")

# ─────────────────────────────────────────────────────────────
# UPLOAD FLOW
# ─────────────────────────────────────────────────────────────
st.markdown("### Upload Protein Data")

uploaded_file = st.file_uploader(
    "Drag and drop your file here",
    type=["csv", "tsv", "txt"],
    key="prot_upload_new",  # unique key
    label_visibility="collapsed"
)

col_skip, _ = st.columns([1, 4])
if col_skip.button("Skip Protein Upload"):
    st.session_state.skip_prot = True
    st.info("Protein upload skipped")
    st.stop()

if not uploaded_file:
    st.info("Please upload a file to continue")
    st.stop()

# ─────────────────────────────────────────────────────────────
# LOAD DATA (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_and_parse(_file):
    content = _.getvalue().decode("utf-8", errors="replace")
    if content.startswith("\ufeff"):
        content = content[1:]
    df = pd.read_csv(io.StringIO(content), sep=None, engine="python", dtype=str)
    
    # Clean intensity columns
    exclude = ["pg", "name", "Protein.Group", "PG.ProteinNames", "Accession", "Species"]
    intensity_cols = [c for c in df.columns if c not in exclude]
    for col in intensity_cols:
        df[col] = pd.to_numeric(df[col].str.replace(",", "").replace("#NUM!", ""), errors="coerce")
    
    # Split name → Accession + Species if needed
    if "name" in df.columns:
        split = df["name"].str.split(",", n=1, expand=True)
        if split.shape[1] == 2:
            df.insert(1, "Accession", split[0])
            df.insert(2, "Species", split[1])
            df = df.drop(columns=["name"])
    return df

df = load_and_parse(uploaded_file)
st.success(f"Data imported — {len(df):,} proteins")

# ─────────────────────────────────────────────────────────────
# COLUMN ASSIGNMENT & RENAMING
# ─────────────────────────────────────────────────────────────
st.subheader("Column Assignment & Renaming")

num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]

# Default selections
default_c1 = num_cols[:len(num_cols)//2] if num_cols else []
sp_col_default = "Species" if "Species" in df.columns else None
pg_col_default = "Protein.Group" if "Protein.Group" in df.columns else df.columns[0]

# Build editor dataframe
rows = []
for col in df.columns:
    preview = " | ".join(df[col].dropna().astype(str).unique()[:3])
    rows.append({
        "Rename": col,
        "Cond 1": col in default_c1,
        "Species": col == sp_col_default,
        "Protein Group": col == pg_col_default,
        "Original": col,
        "Preview": preview[:100],
        "Type": "Intensity" if col in num_cols else "Metadata"
    })

edited = st.data_editor(
    pd.DataFrame(rows),
    column_config={
        "Rename": st.column_config.TextColumn("Rename Column", required=False),
        "Cond 1": st.column_config.CheckboxColumn("Condition 1", help="Select replicates for condition 1"),
        "Species": st.column_config.CheckboxColumn("Species Column", help="Exactly one"),
        "Protein Group": st.column_config.CheckboxColumn("Protein Group ID", help="Exactly one"),
        "Original": st.column_config.TextColumn(disabled=True),
        "Preview": st.column_config.TextColumn(disabled=True),
        "Type": st.column_config.TextColumn(disabled=True),
    },
    disabled=["Original", "Preview", "Type"],
    hide_index=True,
    use_container_width=True,
    key="prot_editor"
)

# Extract selections
c1_orig = edited[edited["Cond 1"]]["Original"].tolist()
c2_orig = [c for c in num_cols if c not in c1_orig]
sp_cols = edited[edited["Species"]]["Original"].tolist()
pg_cols = edited[edited["Protein Group"]]["Original"].tolist()

# Validation
errors = []
if len(c1_orig) == 0: errors.append("Select at least one column for Condition 1")
if len(c2_orig) == 0: errors.append("Condition 2 has no columns (select fewer for C1)")
if len(sp_cols) != 1: errors.append("Select exactly one Species column")
if len(pg_cols) != 1: errors.append("Select exactly one Protein Group column")

if errors:
    for e in errors:
        st.error(e)
    st.stop()

sp_col = sp_cols[0]
pg_col = pg_cols[0]

# Apply renaming
rename_map = {row["Original"]: row["Rename"].strip() for _, row in edited.iterrows() if row["Rename"].strip() and row["Rename"].strip() != row["Original"]}
if rename_map:
    df = df.rename(columns=rename_map)
    c1 = [rename_map.get(old, old) for old in c1_orig]
    c2 = [rename_map.get(old, old) for old in c2_orig]
else:
    c1 = c1_orig
    c2 = c2_orig

# ─────────────────────────────────────────────────────────────
# SPECIES DETECTION & COUNTS
# ─────────────────────────────────────────────────────────────
def detect_species_column_and_extract(df):
    # ... (keep your existing function unchanged) ...
    # Just return sp_col_found, extracted_col_name, unique_species
    pass  # ← replace with your full function

sp_src, sp_ext, species_list = detect_species_column_and_extract(df)
sp_counts = None
if sp_ext and len(species_list) > 1:
    # ... your existing counting logic ...
    pass

# ─────────────────────────────────────────────────────────────
# SAVE TO SESSION STATE
# ─────────────────────────────────────────────────────────────
st.session_state.update({
    "prot_df": df,
    "prot_c1": c1,
    "prot_c2": c2,
    "prot_sp_col": sp_col,
    "prot_pg_col": pg_col,
    "prot_sp_ext": sp_ext,
    "prot_species": species_list,
    "prot_sp_counts": sp_counts,
    "prot_rename": rename_map,
    "allow_reconfigure_prot": False,  # Reset flag after successful config
})

# ─────────────────────────────────────────────────────────────
# SUCCESS DISPLAY
# ─────────────────────────────────────────────────────────────
st.success("All assignments applied successfully!")

col1, col2 = st.columns(2)
with col1:
    st.metric("**Condition 1**", f"{len(c1)} reps")
    st.write(", ".join(c1))
with col2:
    st.metric("**Condition 2**", f"{len(c2)} reps")
    st.write(", ".join(c2))

st.info(f"**Species** → `{sp_col}` | **Protein Group** → `{pg_col}`")

if sp_counts is not None and len(sp_counts) > 0:
    st.markdown("### Proteins by Species & Condition")
    st.dataframe(sp_counts, hide_index=True, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.bar_chart(sp_counts[["species", "c1", "c2"]].set_index("species"))
    with c2:
        for _, r in sp_counts.iterrows():
            st.write(f"- **{r['species']}**: {r['total']} total | C1: {r['c1']} | C2: {r['c2']} | Both: {r['both']}")

# Final restart button
st.markdown('<div class="fixed-restart">', unsafe_allow_html=True)
if st.button("Restart Analysis", use_container_width=True, type="primary"):
    for k in list(st.session_state.keys()):
        if k.startswith("prot_") or k == "allow_reconfigure_prot":
            del st.session_state[k]
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer"><strong>Proprietary & Confidential</strong><br>© 2024 Thermo Fisher Scientific</div>', unsafe_allow_html=True)
