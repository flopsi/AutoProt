# pages/2_Peptide_Import.py
import streamlit as st
import pandas as pd
import io
from shared import restart_button
import numpy as np

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
st.markdown('<div class="header"><h1>DIA Proteomics Pipeline</h1><p>Peptide Import</p></div>', unsafe_allow_html=True)

# === UPLOAD ONCE, KEEP FOREVER ===
if "uploaded_peptide_bytes" not in st.session_state:
    st.markdown("### Upload Peptide File")
    uploaded = st.file_uploader("CSV / TSV / TXT", type=["csv","tsv","txt"], key="peptide_uploader")
    if uploaded:
        st.session_state.uploaded_peptide_bytes = uploaded.getvalue()
        st.session_state.uploaded_peptide_name = uploaded.name
        st.rerun()
    else:
        restart_button()
        st.stop()
else:
    st.success(f"Peptide file ready: **{st.session_state.uploaded_peptide_name}**")

# === LOAD FROM BYTES ===
@st.cache_data(show_spinner="Loading peptide data...")
def load_peptide_data(_bytes):
    text = _bytes.decode("utf-8", errors="replace")
    if text.startswith("\ufeff"):
        text = text[1:]
    return pd.read_csv(io.StringIO(text), sep=None, engine="python", low_memory=False)

df_raw = load_peptide_data(st.session_state.uploaded_peptide_bytes)
st.write(f"**{len(df_raw):,}** rows × **{len(df_raw.columns)}** columns (raw)")

# === DETECT LONG FORMAT (e.g., Spectronaut, DIA-NN long export) ===
sample_col_candidates = [c for c in df_raw.columns if any(x in c.lower() for x in ["file", "run", "sample", "raw"])]
quantity_col_candidates = [c for c in df_raw.columns if any(x in c.lower() for x in ["quantity", "intensity", "area", "fg.", "pg."])]
peptide_col_candidates = [c for c in df_raw.columns if any(x in c.lower() for x in ["peptide", "sequence", "modified", "stripped", "eg.", "pep"])]

is_long_format = (
    len(sample_col_candidates) > 0 and
    len(quantity_col_candidates) >= 1 and
    len(peptide_col_candidates) > 0
)

if is_long_format:
    sample_col = sample_col_candidates[0]
    quantity_col = quantity_col_candidates[0]
    peptide_col = peptide_col_candidates[0]

    st.info(f"Detected long-format peptide file!\n"
            f"→ Sample column: `{sample_col}`\n"
            f"→ Quantity column: `{quantity_col}`\n"
            f"→ Peptide column: `{peptide_col}`")

    # Optional: Let user confirm or correct column choices
    col1, col2, col3 = st.columns(3)
    with col1:
        sample_col = st.selectbox("Sample/Run column", df_raw.columns, index=df_raw.columns.get_loc(sample_col))
    with col2:
        quantity_col = st.selectbox("Quantity/Intensity column", df_raw.columns, index=df_raw.columns.get_loc(quantity_col))
    with col3:
        peptide_col = st.selectbox("Peptide sequence column", df_raw.columns, index=df_raw.columns.get_loc(peptide_col))

    if st.button("Pivot Long → Wide Format", type="secondary"):
        with st.spinner("Pivoting peptide data..."):
            # Keep metadata columns (everything except sample/quantity)
            meta_cols = [c for c in df_raw.columns if c not in [sample_col, quantity_col]]

            # Drop duplicates per peptide+sample if any (safety)
            df_clean = df_raw.drop_duplicates(subset=[peptide_col, sample_col])

            # Pivot: peptides → rows, samples → columns
            df_pivot = df_clean.pivot_table(
                values=quantity_col,
                index=[peptide_col] + [c for c in meta_cols if c != sample_col],
                columns=sample_col,
                aggfunc='mean'  # or 'sum' depending on your data
            ).reset_index()

            # Flatten multi-level columns
            if isinstance(df_pivot.columns, pd.MultiIndex):
                df_pivot.columns = [str(c[1]) if c[0] == quantity_col.split('.')[-1] or c[0] == '' else c[0] for c in df_pivot.columns.values]
                df_pivot.columns = [c if c != peptide_col else "Peptide" for c in df_pivot.columns]

            df_raw = df_pivot
            st.success(f"Pivoted successfully! Now {len(df_raw):,} peptides × {len(df_raw.columns)} columns")
            st.write("First few columns after pivot:", list(df_raw.columns[:10]))

# === FIND INTENSITY COLUMNS (after possible pivot) ===
intensity_cols = []
for col in df_raw.columns:
    # Skip obvious non-intensity columns
    if col in ["Peptide", "Sequence", "Modified Sequence", "Protein", "Gene", peptide_col]:
        continue
    cleaned = pd.to_numeric(df_raw[col].astype(str).str.replace(r"[,\#NUM!]", "", regex=True), errors='coerce')
    if cleaned.notna().mean() > 0.3:  # at least 30% real values
        df_raw[col] = cleaned
        intensity_cols.append(col)

if not intensity_cols:
    st.error("No quantitative (intensity) columns found. Check your file format.")
    st.stop()

st.write(f"Found **{len(intensity_cols)}** quantitative columns for replicate assignment.")

# === REPLICATES ASSIGNMENT ===
st.markdown("### Assign Replicates (must be equal number in A and B)")
rows = [{"Column": c, "A": False, "B": False} for c in intensity_cols]
edited = st.data_editor(
    pd.DataFrame(rows),
   
