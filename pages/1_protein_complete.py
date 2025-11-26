# pages/1_protein.py
import streamlit as st
import pandas as pd
import io
import re
from pandas.api.types import is_numeric_dtype

st.set_page_config(page_title="Protein Import", layout="wide")

# ─────────────────────────────────────────────────────────────
# Detect Species Function
# ─────────────────────────────────────────────────────────────
def detect_species_column_and_extract(df):
    metadata_cols = []
    for col in df.columns:
        sample = df[col].dropna().head(50)
        try:
            pd.to_numeric(sample.replace('#NUM!', pd.NA), errors='raise')
        except:
            metadata_cols.append(col)
    
    species_col = None
    prefix_delim = None
    
    for col in metadata_cols:
        sample_values = df[col].dropna().astype(str).head(100)
        for val in sample_values:
            if 'HUMAN' in val.upper():
                species_col = col
                match = re.search(r'([^A-Za-z0-9])HUMAN', val, re.IGNORECASE)
                prefix_delim = match.group(1) if match else ''
                break
        if species_col:
            break
    
    if not species_col:
        return None, None, []
    
    if prefix_delim:
        prefix_esc = re.escape(prefix_delim)
        pattern = f'{prefix_esc}([A-Z]+)(?:[^A-Za-z]|$)'
    else:
        pattern = r'^([A-Z]+)$'
    
    def extract_first_species(val):
        if pd.isna(val):
            return None
        val_str = str(val).upper()
        matches = re.findall(pattern, val_str)
        return matches[0] if matches else None
    
    df['_species'] = df[species_col].apply(extract_first_species)
    unique_species = sorted(df['_species'].dropna().unique().tolist())
    return species_col, '_species', unique_species

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .header {background: linear-gradient(90deg, #E71316 0%, #A6192E 100%); padding: 20px 40px; color: white; margin: -60px -60px 40px -60px;}
    .header h1 {margin:0; font-size:28px; font-weight:600;}
    .card {background:white; border:1px solid #E2E3E4; border-radius:8px; padding:25px; box-shadow:0 2px 4px rgba(0,0,0,0.05); margin-bottom:25px;}
    .upload-area {border:2px dashed #E2E3E4; border-radius:8px; padding:60px 30px; text-align:center; background:#fafafa; cursor:pointer;}
    .upload-area:hover {border-color:#E71316; background:rgba(231,19,22,0.02);}
    .stButton>button {background:#E71316 !important; color:white !important; border:none !important; padding:12px 24px !important; border-radius:6px !important; font-weight:500 !important;}
    .stButton>button:hover {background:#A6192E !important;}
    .footer {text-align:center; padding:30px; color:#54585A; font-size:12px; border-top:1px solid #E2E3E4; margin-top:60px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>🔬 Protein Level Data Import</h1></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CHECK IF DATA EXISTS IN SESSION
# ─────────────────────────────────────────────────────────────
if "prot_df" in st.session_state:
    st.info("📂 **Data restored from session state**")
    
    df = st.session_state.prot_df
    c1 = st.session_state.prot_c1
    c2 = st.session_state.prot_c2
    sp_col = st.session_state.prot_sp_col
    pg_col = st.session_state.prot_pg_col
    sp_counts = st.session_state.prot_sp_counts
    
    st.success("✅ Protein data loaded!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("**Condition 1**", f"{len(c1)} reps")
        st.write(", ".join(c1))
    with col2:
        st.metric("**Condition 2**", f"{len(c2)} reps")
        st.write(", ".join(c2))
    
    st.info(f"**Species** → `{sp_col}` | **Protein Group** → `{pg_col}`")
    
    if sp_counts is not None and len(sp_counts) > 0:
        st.markdown("### 📊 Proteins by Species & Condition")
        st.dataframe(sp_counts, hide_index=True, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            chart = sp_counts[["species", "c1", "c2"]].set_index("species")
            st.bar_chart(chart)
        with col2:
            st.write("**Summary:**")
            for _, r in sp_counts.iterrows():
                st.write(f"- **{r['species']}**: {r['total']} total | C1: {r['c1']} | C2: {r['c2']} | Both: {r['both']}")
    
    st.markdown("---")
    st.markdown('<div class="footer"><strong>Proprietary & Confidential</strong><br>© 2024 Thermo Fisher Scientific</div>', unsafe_allow_html=True)
    
    # Restart button on restored page
    st.markdown("""<style>.fixed-restart {position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); z-index: 999; width: 300px;}</style><div class="fixed-restart">""", unsafe_allow_html=True)
    if st.button("🔄 Restart Analysis", use_container_width=True, key="restart_prot"):
        st.session_state.clear()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────
# UPLOAD FLOW
# ─────────────────────────────────────────────────────────────
st.markdown("### 📤 Upload Protein Data")

with st.container():
    st.markdown('<div class="card"><div class="upload-area"><div style="font-size:64px; opacity:0.5; margin-bottom:20px;">📤</div><div style="font-size:16px; color:#54585A; margin-bottom:10px;"><strong>Drag and drop your file</strong></div><div style="font-size:13px; color:#54585A; opacity:0.7;">Supports .csv, .tsv, .txt</div></div></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader("", type=["csv","tsv","txt"], label_visibility="collapsed", key="prot_upload")
    with col2:
        skip_upload = st.button("⏭️ Skip", key="skip_prot_btn")

if skip_upload:
    st.session_state["skip_prot"] = True
    st.info("⏭️ Protein upload skipped")
    st.markdown('<div class="footer"><strong>Proprietary & Confidential</strong><br>© 2024 Thermo Fisher Scientific</div>', unsafe_allow_html=True)
    st.stop()

if not uploaded_file:
    st.markdown('<div class="footer"><strong>Proprietary & Confidential</strong><br>© 2024 Thermo Fisher Scientific</div>', unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────
# Load & Parse
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_and_parse(file):
    content = file.getvalue().decode("utf-8", errors="replace")
    if content.startswith("\ufeff"): 
        content = content[1:]
    df = pd.read_csv(io.StringIO(content), sep=None, engine="python", dtype=str)
    intensity_cols = [c for c in df.columns if c not in ["pg", "name", "Protein.Group", "PG.ProteinNames", "Accession", "Species"]]
    for col in intensity_cols:
        df[col] = pd.to_numeric(df[col].str.replace(",", "").replace("#NUM!", ""), errors="coerce")
    if "name" in df.columns:
        split = df["name"].str.split(",", n=1, expand=True)
        if split.shape[1] == 2:
            df.insert(1, "Accession", split[0])
            df.insert(2, "Species", split[1])
            df = df.drop(columns=["name"])
    return df

df = load_and_parse(uploaded_file)
st.success(f"✅ Data imported — {len(df):,} proteins")

# ─────────────────────────────────────────────────────────────
# Column Assignment
# ─────────────────────────────────────────────────────────────
with st.container():
    st.subheader("⚙️ Column Assignment & Renaming")

    intensity_cols = [c for c in df.columns if c not in ["pg", "name", "Protein.Group", "PG.ProteinNames", "Accession", "Species"]]
    for col in intensity_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    meta_cols_list = ["Protein.Group", "PG.ProteinNames", "Accession", "Species"]
    for col in meta_cols_list:
        if col in df.columns:
            df[col] = df[col].astype("string")

    num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]

    ratio_groups = {}
    for col in num_cols:
        m = re.search(r'_Y(\d{2})-E(\d{2})_', col)
        if m:
            key = f"Y{m.group(1)}-E{m.group(2)}"
            ratio_groups.setdefault(key, []).append(col)

    default_c1 = ratio_groups.get(
        sorted(ratio_groups.keys(), key=lambda x: int(x.split('-')[0][1:]))[0],
        num_cols[:len(num_cols)//2]
    ) if ratio_groups else num_cols[:len(num_cols)//2]

    sp_col_default = "Species" if "Species" in df.columns else "PG.ProteinNames"
    pg_col_default = "Protein.Group" if "Protein.Group" in df.columns else df.columns[0]

    rows = []
    for col in df.columns:
        is_num = col in num_cols
        preview = " | ".join(df[col].dropna().astype(str).unique()[:3])
        rows.append({
            "Rename": col,
            "Cond 1": col in default_c1 and is_num,
            "Species": col == sp_col_default,
            "Protein Group": col == pg_col_default,
            "Original Name": col,
            "Preview": preview,
            "Type": "Intensity" if is_num else "Metadata"
        })

    df_edit = pd.DataFrame(rows)

    edited = st.data_editor(
        df_edit,
        column_config={
            "Rename": st.column_config.TextColumn("Rename", required=False),
            "Cond 1": st.column_config.CheckboxColumn("Cond 1", default=True),
            "Species": st.column_config.CheckboxColumn("Species", default=True),
            "Protein Group": st.column_config.CheckboxColumn("PG", default=True),
            "Original Name": st.column_config.TextColumn("Original", disabled=True),
            "Preview": st.column_config.TextColumn("Preview", disabled=True),
            "Type": st.column_config.TextColumn("Type", disabled=True),
        },
        disabled=["Original Name", "Preview", "Type"],
        hide_index=True,
        use_container_width=True,
        key="prot_col_table"
    )

    c1_checked = edited[edited["Cond 1"]]
    sp_checked = edited[edited["Species"]]
    pg_checked = edited[edited["Protein Group"]]

    c1_orig = c1_checked["Original Name"].tolist()
    c2_orig = [c for c in num_cols if c not in c1_orig]
    sp_cols = sp_checked["Original Name"].tolist()
    pg_cols = pg_checked["Original Name"].tolist()

    errors = []
    if len(c1_orig) == 0:
        errors.append("❌ Cond 1 must have ≥1 replicate")
    if len(c2_orig) == 0:
        errors.append("❌ Cond 2 must have ≥1 replicate")
    if len(sp_cols) != 1:
        errors.append("❌ Select exactly 1 Species column")
    if len(pg_cols) != 1:
        errors.append("❌ Select exactly 1 Protein Group column")

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    sp_col = sp_cols[0]
    pg_col = pg_cols[0]

    rename_map = {}
    for _, row in edited.iterrows():
        new = row["Rename"].strip()
        if new and new != row["Original Name"]:
            rename_map[row["Original Name"]] = new

    if rename_map:
        df = df.rename(columns=rename_map)
        c1 = [rename_map.get(c, c) for c in c1_orig]
        c2 = [rename_map.get(c, c) for c in c2_orig]
    else:
        c1 = c1_orig
        c2 = c2_orig

# ─────────────────────────────────────────────────────────────
# Species Detection
# ─────────────────────────────────────────────────────────────
sp_src, sp_ext, species_list = detect_species_column_and_extract(df)

sp_counts = None
if sp_ext and len(species_list) > 1:
    for col in c1 + c2:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    rows = []
    for sp in species_list:
        sp_df = df[df[sp_ext] == sp]
        v1 = (sp_df[c1] > 1).sum(axis=1) >= 2
        v2 = (sp_df[c2] > 1).sum(axis=1) >= 2
        rows.append({
            "species": sp,
            "c1": v1.sum(),
            "c2": v2.sum(),
            "both": (v1 & v2).sum(),
            "total": len(sp_df)
        })
    sp_counts = pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
# Save to Session
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
    "prot_n": len(df),
    "prot_rename": rename_map,
    "prot_num_cols": num_cols,
})

# ─────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────
st.success("✅ All assignments applied!")

col1, col2 = st.columns(2)
with col1:
    st.metric("**Condition 1**", f"{len(c1)} reps")
    st.write(", ".join(c1))
with col2:
    st.metric("**Condition 2**", f"{len(c2)} reps")
    st.write(", ".join(c2))

st.info(f"**Species** → `{sp_col}` | **Protein Group** → `{pg_col}`")

if sp_counts is not None:
    st.markdown("### 📊 Proteins by Species & Condition")
    st.dataframe(sp_counts, hide_index=True, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        chart = sp_counts[["species", "c1", "c2"]].set_index("species")
        st.bar_chart(chart)
    with col2:
        st.write("**Summary:**")
        for _, r in sp_counts.iterrows():
            st.write(f"- **{r['species']}**: {r['total']} total | C1: {r['c1']} | C2: {r['c2']} | Both: {r['both']}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# Restart Button
# ─────────────────────────────────────────────────────────────
st.markdown("""<style>.fixed-restart {position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); z-index: 999; width: 300px;}</style><div class="fixed-restart">""", unsafe_allow_html=True)
if st.button("🔄 Restart Analysis", use_container_width=True, key="restart_prot_final"):
    st.session_state.clear()
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer"><strong>Proprietary & Confidential</strong><br>© 2024 Thermo Fisher Scientific</div>', unsafe_allow_html=True)
