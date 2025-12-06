"""
pages/1_Data_Upload.py

Data upload, configuration, and initial validation
Leverages existing helper functions for maximum code reuse
"""

import streamlit as st
import pandas as pd
import numpy as np
import re

from helpers.io import (
    read_file, 
    detect_numeric_columns,
    detect_protein_id_column,
    detect_species_column,
    clean_species_name,
    drop_proteins_with_invalid_intensities
)
from helpers.core import ProteinData
from helpers.audit import log_event

# ============================================================================
# HELPER FUNCTIONS (only those not in helpers)
# ============================================================================

def longest_common_prefix(strings: list) -> str:
    """Find longest common prefix from list of strings."""
    if not strings:
        return ""
    s1, s2 = min(strings), max(strings)
    for i, (c1, c2) in enumerate(zip(s1, s2)):
        if c1 != c2:
            return s1[:i]
    return s1

def generate_default_column_names(n_cols: int, replicates_per_condition: int = 3) -> list:
    """
    Generate default names: A1, A2, A3, B1, B2, B3, etc.
    
    Args:
        n_cols: Total number of columns
        replicates_per_condition: Samples per condition
    
    Returns:
        List of generated names
    """
    names = []
    for i in range(n_cols):
        condition_idx = i // replicates_per_condition
        replicate_num = (i % replicates_per_condition) + 1
        condition_letter = chr(ord('A') + condition_idx)
        names.append(f"{condition_letter}{replicate_num}")
    return names

def infer_species_from_protein_name(name: str) -> str:
    """Extract species from protein name (e.g., 'PROT_HUMAN' → 'HUMAN')."""
    if pd.isna(name):
        return None
    s = str(name).upper()
    
    # Check common patterns
    if "_HUMAN" in s:
        return "HUMAN"
    if "_MOUSE" in s:
        return "MOUSE"
    if "_YEAST" in s:
        return "YEAST"
    if "_ECOLI" in s or "_ECOL" in s:
        return "ECOLI"
    
    # Fallback: last token after underscore
    if "_" in s:
        tail = s.split("_")[-1]
        if len(tail) >= 3:
            return tail
    
    return None

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(page_title="Data Upload", layout="wide")

with st.sidebar:
    st.title("📊 Upload Settings")
    st.info("""
        **Supported Formats:**
        - CSV (.csv)
        - TSV (.tsv, .txt)
        - Excel (.xlsx)
    """)

# ============================================================================
# MAIN PAGE
# ============================================================================

st.title("📊 Data Upload & Configuration")

# Step 1: File Upload
st.subheader("1️⃣ Upload File")

uploaded_file = st.file_uploader(
    "Choose a proteomics data file",
    type=["csv", "tsv", "txt", "xlsx"],
    help="Supports CSV, TSV, and Excel formats"
)

if uploaded_file is None:
    st.warning("⚠️ Please upload a data file to continue")
    st.stop()

# Step 2: Read File (using helper)
st.subheader("2️⃣ Loading Data...")

try:
    filename = uploaded_file.name.lower()
    df = read_file(uploaded_file)  # Helper function with caching!
    
    # Determine format
    if filename.endswith(".xlsx"):
        file_format = "Excel"
    elif filename.endswith((".tsv", ".txt")):
        file_format = "TSV"
    else:
        file_format = "CSV"
    
    st.success(f"✅ Loaded {file_format}: {len(df):,} rows × {len(df.columns)} columns")
    
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.stop()

# Clean column names
df.columns = [re.sub(r"[\(\)\[\]\{\}]", "", c).replace(" ", "_").strip("_") for c in df.columns]

# Auto-trim shared prefix from long columns
long_cols = [c for c in df.columns if len(c) > 40]
prefix = longest_common_prefix(long_cols)

if prefix and len(prefix) >= 20:
    st.info(f"🧹 Removing shared prefix: `{prefix}`")
    df.columns = [c[len(prefix):].lstrip("_.") if c.startswith(prefix) else c for c in df.columns]

# Step 3: Data Cleaning
st.subheader("3️⃣ Data Cleaning")

n_nan_before = df.isna().sum().sum()
n_zero_before = (df == 0).sum().sum()

st.info("Replacing NaN and 0 with 1.0 for log transformation...")

numeric_cols_detected = detect_numeric_columns(df)  # Helper!

for col in numeric_cols_detected:
    df[col] = df[col].fillna(1.0)
    df.loc[df[col] == 0, col] = 1.0

c1, c2 = st.columns(2)
c1.metric("NaN Replaced", n_nan_before, delta=f"-{n_nan_before}")
c2.metric("Zeros Replaced", n_zero_before, delta=f"-{n_zero_before}")

st.success("✅ Cleaning complete")

# Step 4: Select Quantitative Columns
st.subheader("4️⃣ Select Quantitative Columns")

df_cols = pd.DataFrame({
    "Select": [col in numeric_cols_detected for col in df.columns],
    "Column": df.columns.tolist(),
    "Type": [str(df[col].dtype) for col in df.columns],
    "Sample": [str(df[col].iloc[0])[:30] if len(df) > 0 else "" for col in df.columns]
})

edited = st.data_editor(
    df_cols,
    column_config={
        "Select": st.column_config.CheckboxColumn("✓", width="small"),
        "Column": st.column_config.TextColumn("Column", disabled=True),
        "Type": st.column_config.TextColumn("Type", width="small", disabled=True),
        "Sample": st.column_config.TextColumn("Sample", disabled=True)
    },
    hide_index=True,
    use_container_width=True
)

numeric_cols = edited[edited["Select"]]["Column"].tolist()

if len(numeric_cols) < 4:
    st.warning(f"⚠️ Need ≥4 columns. Selected: {len(numeric_cols)}")
    st.stop()

st.success(f"✅ Selected {len(numeric_cols)} columns")

# Step 5: Rename Columns
st.subheader("5️⃣ Rename Columns")

c1, c2 = st.columns(2)
with c1:
    replicates = st.number_input("Replicates per condition:", 1, 10, 3)
with c2:
    use_default = st.checkbox("Use default names (A1, A2, B1...)", value=True)

rename_dict = {}

if use_default:
    default_names = generate_default_column_names(len(numeric_cols), replicates)
    rename_dict = dict(zip(numeric_cols, default_names))
    st.info(f"✅ Generated names: {', '.join(default_names[:6])}...")
    
    # Optional: allow editing
    if st.checkbox("Edit names"):
        with st.expander("Edit Individual Names"):
            for idx, (old, new) in enumerate(rename_dict.items()):
                c1, c2, c3 = st.columns([2, 1, 2])
                c1.text(old)
                c2.write("→")
                custom = c3.text_input("", new, label_visibility="collapsed", key=f"rn{idx}")
                if custom != new:
                    rename_dict[old] = custom

if rename_dict:
    df = df.rename(columns=rename_dict)
    numeric_cols = [rename_dict.get(c, c) for c in numeric_cols]

# Step 6: Identify Metadata
st.subheader("6️⃣ Metadata Columns")

non_numeric = [c for c in df.columns if c not in numeric_cols]

c1, c2 = st.columns(2)

with c1:
    protein_id_col = detect_protein_id_column(df)  # Helper!
    if protein_id_col not in non_numeric and non_numeric:
        protein_id_col = non_numeric[0]
    
    if non_numeric:
        protein_id_col = st.selectbox(
            "🔍 Protein/Peptide ID",
            non_numeric,
            index=non_numeric.index(protein_id_col) if protein_id_col in non_numeric else 0
        )
    else:
        st.warning("No non-numeric columns")
        protein_id_col = df.columns[0]

with c2:
    species_col = detect_species_column(df)  # Helper!
    options = ["(None)"] + non_numeric
    
    idx = options.index(species_col) if species_col in options else 0
    species_col = st.selectbox("🧬 Species (optional)", options, index=idx)
    
    if species_col == "(None)":
        species_col = None
    elif species_col:
        df[species_col] = df[species_col].apply(clean_species_name)  # Helper!

# Step 7: Drop Unused Columns
columns_to_keep = [protein_id_col] + numeric_cols
if species_col:
    columns_to_keep.append(species_col)

df = df[columns_to_keep]

# Step 8: Preview
st.subheader("7️⃣ Preview")

df_preview = df.head(10).copy()
for col in df_preview.select_dtypes(include=['float']).columns:
    df_preview[col] = df_preview[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")

st.dataframe(df_preview, use_container_width=True, height=350)

# Step 9: Statistics
st.subheader("8️⃣ Statistics")

# Create species mapping
if species_col:
    species_series = df[species_col]
else:
    species_series = df[protein_id_col].apply(infer_species_from_protein_name)
    df["__INFERRED_SPECIES__"] = species_series
    species_col = "__INFERRED_SPECIES__"

species_mapping = dict(zip(df[protein_id_col], species_series))

n_proteins = len(df)
n_samples = len(numeric_cols)
n_conditions = max(1, n_samples // replicates)

missing_count = sum((df[c].isna().sum() + (df[c] == 1.0).sum()) for c in numeric_cols)
missing_rate = (missing_count / (n_proteins * n_samples) * 100) if n_proteins > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Proteins", f"{n_proteins:,}")
c2.metric("Samples", n_samples)
c3.metric("Conditions", n_conditions)
c4.metric("Missing %", f"{missing_rate:.1f}%")

# Step 10: Optional Filtering
st.subheader("9️⃣ Finalize")

drop_invalid = st.checkbox("Drop proteins with any NaN or 1.00 intensity", value=False)

rows_dropped = 0
if drop_invalid:
    before = len(df)
    df = drop_proteins_with_invalid_intensities(df, numeric_cols, 1.0, True)  # Helper!
    rows_dropped = before - len(df)
    
    st.info(f"Dropped {rows_dropped} proteins. Remaining: {len(df)}")
    
    # Recalculate stats
    n_proteins = len(df)
    species_mapping = dict(zip(df[protein_id_col], df[species_col]))
    missing_count = sum((df[c].isna().sum() + (df[c] == 1.0).sum()) for c in numeric_cols)
    missing_rate = (missing_count / (n_proteins * n_samples) * 100) if n_proteins > 0 else 0
    
    st.markdown("**Updated:**")
    uc1, uc2, uc3, uc4 = st.columns(4)
    uc1.metric("Proteins", f"{n_proteins:,}", delta=f"-{rows_dropped}")
    uc2.metric("Samples", n_samples)
    uc3.metric("Conditions", n_conditions)
    uc4.metric("Missing %", f"{missing_rate:.1f}%")

# Create ProteinData object
protein_data = ProteinData(
    raw=df,
    numeric_cols=numeric_cols,
    species_col=species_col,
    species_mapping=species_mapping,
    index_col=protein_id_col,
    file_path=uploaded_file.name,
    file_format=file_format
)

st.session_state.protein_data = protein_data
st.session_state.column_mapping = rename_dict

# Log event
log_event(
    page="Data Upload",
    action=f"Uploaded {uploaded_file.name}",
    details={
        "filename": uploaded_file.name,
        "format": file_format,
        "n_proteins": n_proteins,
        "n_samples": n_samples,
        "missing_rate": missing_rate,
        "nan_replaced": n_nan_before,
        "zeros_replaced": n_zero_before,
        "proteins_dropped": rows_dropped
    }
)
# Add THIS section right before "# ============================================================================ # STEP 10: NEXT STEPS"

# ============================================================================
# CONFIRMATION BUTTON
# ============================================================================

st.markdown("---")
st.subheader("🎯 Confirm & Save Configuration")

st.info(f"""
**Ready to proceed?** Click below to save your data configuration.

- **File**: `{uploaded_file.name}`
- **Proteins**: {n_proteins:,}
- **Samples**: {n_samples}
- **Species**: {len(species_series.unique())} detected
""")

if st.button("✅ Confirm & Proceed to Analysis", type="primary", use_container_width=True):
    # Data already in protein_data object above - just confirm it's stored
    st.success("🎉 Data successfully confirmed and stored in session!")
    st.balloons()
    
    st.info("💡 Navigate to **2️⃣ Visual EDA** or **3️⃣ Statistical EDA** in the sidebar to continue.")

st.success("✅ Data ready for analysis!")

# Navigation hint
st.markdown("---")
st.info("**Next:** Navigate to **📊 Visual EDA** in the sidebar")
