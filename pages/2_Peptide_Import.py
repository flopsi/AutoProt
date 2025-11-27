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

@st.cache_data(show_spinner="Loading peptide data...")
def load_peptide_data(_bytes):
    text = _bytes.decode("utf-8", errors="replace")
    if text.startswith("\ufeff"):
        text = text[1:]
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")  # ← Removed low_memory=False

df_raw = load_peptide_data(st.session_state.uploaded_peptide_bytes)
st.write(f"**{len(df_raw):,}** rows × **{len(df_raw.columns)}** columns (raw)")

# === DETECT LONG FORMAT ===
sample_col_candidates = [c for c in df_raw.columns if any(x in c.lower() for x in ["file", "run", "sample", "raw"])]
quantity_col_candidates = [c for c in df_raw.columns if any(x in c.lower() for x in ["quantity", "intensity", "area", "fg.", "pg."])]
peptide_col_candidates = [c for c in df_raw.columns if any(x in c.lower() for x in ["peptide", "sequence", "modified", "stripped", "eg.", "pep"])]

is_long_format = len(sample_col_candidates) > 0 and len(quantity_col_candidates) >= 1 and len(peptide_col_candidates) > 0

if is_long_format:
    # Pre-select the most likely columns
    sample_col   = sample_col_candidates[0]
    quantity_col = quantity_col_candidates[0]
    peptide_col  = peptide_col_candidates[0]

    st.info(f"Long-format peptide file detected!\n"
            f"Sample column: `{sample_col}` | Quantity: `{quantity_col}` | Peptide: `{peptide_col}`")

    col1, col2, col3 = st.columns(3)
    with col1:
        sample_col   = st.selectbox("Sample / Run column",   df_raw.columns, index=df_raw.columns.get_loc(sample_col))
    with col2:
        quantity_col = st.selectbox("Quantity / Intensity column", df_raw.columns, index=df_raw.columns.get_loc(quantity_col))
    with col3:
        peptide_col  = st.selectbox("Peptide sequence column", df_raw.columns, index=df_raw.columns.get_loc(peptide_col))

    if st.button("Pivot Long to Wide Format", type="primary"):
        with st.spinner("Pivoting..."):
            df_clean = df_raw.drop_duplicates(subset=[peptide_col, sample_col])
            meta_cols = [c for c in df_raw.columns if c not in [sample_col, quantity_col]]

            pivoted = df_clean.pivot_table(
                values=quantity_col,
                index=[peptide_col] + meta_cols,
                columns=sample_col,
                aggfunc="mean",
                dropna=False
            )

            # Safe column flattening
            pivoted.columns = [str(col) if col != quantity_col else str(sample) for col, sample in pivoted.columns]

            # Resolve any name clashes
            run_cols = [c for c in pivoted.columns if c not in meta_cols + [peptide_col]]
            rename_dict = {}
            for rc in run_cols:
                if rc in pivoted.columns.tolist() + meta_rc:
                    rename_dict[rc] = f"{rc}_intensity"
            if rename_dict:
                pivoted = pivoted.rename(columns=rename_dict)

            df_raw = pivoted.reset_index().rename(columns={peptide_col: "Peptide"})
            st.success(f"Pivoted → {df_raw.shape[0]:,} peptides × {df_raw.shape[1]} columns")
else:
    st.info("Wide-format file detected – no pivoting needed.")
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
    column_config={
        "Column": st.column_config.TextColumn("Intensity Column", disabled=True),
        "A": st.column_config.CheckboxColumn("Condition A", default=False),
        "B": st.column_config.CheckboxColumn("Condition B", default=False),
    },
    hide_index=True,
    use_container_width=True,
    num_rows="fixed"
)

a_cols = edited[edited["A"]]["Column"].tolist()
b_cols = edited[edited["B"]]["Column"].tolist()

if len(a_cols) != len(b_cols) or len(a_cols) == 0:
    st.error(f"Must assign equal number of replicates! Currently: A={len(a_cols)}, B={len(b_cols)}")
    st.stop()

# === RENAME TO A1, A2, ..., B1, B2... ===
n = len(a_cols)
rename_map = {a_cols[i]: f"A{i+1}" for i in range(n)}
rename_map.update({b_cols[i]: f"B{i+1}" for i in range(n)})

df = df_raw.rename(columns=rename_map).copy()
c1 = [f"A{i+1}" for i in range(n)]
c2 = [f"B{i+1}" for i in range(n)]

st.success(f"Renamed replicates → **A**: {', '.join(c1)} | **B**: {', '.join(c2)}")

# Optional: Replace 0 and NaN with small value (common in peptide data)
df[c1 + c2] = df[c1 + c2].replace([0, np.nan], 1.0)

# === SAVE TO SESSION STATE ===
st.session_state.pept_df = df
st.session_state.pept_c1 = c1
st.session_state.pept_c2 = c2

st.success("Peptide data successfully loaded and formatted!")

if st.button("Go to Peptide Analysis", type="primary", use_container_width=True):
    st.switch_page("pages/4_Peptide_Analysis.py")

restart_button()
