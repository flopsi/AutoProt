"""
pages/1_Data_Upload.py
Complete upload page with checkbox table selection
Updated to use new consolidated helper modules
"""

import streamlit as st
import pandas as pd
import numpy as np
import re

from helpers.io import read_file
from helpers.core import ProteinData, get_theme
from helpers.audit import log_event
from helpers.io import drop_proteins_with_invalid_intensities

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_species_name(name: str) -> str:
    """Remove leading/trailing underscores from species names and upper-case."""
    if pd.isna(name):
        return name
    return str(name).strip().strip("_").upper()


def clean_column_name(name: str) -> str:
    """Clean column names: remove brackets, condense spaces, replace with '_'."""
    name = re.sub(r"[\(\)\[\]\{\}]", "", str(name))
    name = re.sub(r"\s+", " ", name)
    name = name.replace(" ", "_")
    name = name.strip("_")
    return name


def detect_numeric_columns(df: pd.DataFrame) -> list:
    """Detect numeric columns - only float types (float64, float32)."""
    # Only select float columns, not int (which are likely IDs or counts)
    return df.select_dtypes(include=["float64", "float32"]).columns.tolist()


def detect_protein_id_column(df: pd.DataFrame) -> str:
    """Detect protein/peptide ID column."""
    patterns = ["protein", "gene", "id", "accession", "sequence", "uniprot"]

    for col in df.columns:
        col_lower = col.lower()
        if any(p in col_lower for p in patterns):
            if df[col].dtype not in ["int64", "float64", "float32"]:
                return col

    # Fallback: first non-numeric column
    for col in df.columns:
        if df[col].dtype == "object":
            return col

    return df.columns[0]


def detect_species_column(df: pd.DataFrame) -> str:
    """
    Try to detect an explicit species column.

    For your DIA-NN tables this will usually be None, since species is encoded
    in PG.ProteinNames (e.g. NUD4B_HUMAN).
    """
    header_patterns = ["species", "organism", "taxon", "taxonomy"]
    for col in df.columns:
        col_lower = col.lower()
        if any(p in col_lower for p in header_patterns):
            return col
    return None


def infer_species_from_protein_name(name: str) -> str:
    """
    Extract species tag from a protein name like 'NUD4B_HUMAN' or
    'A0A087X1L8_HUMAN;ICOSL_HUMAN'. Returns short labels (HUMAN, YEAST, ECOLI...).
    """
    if pd.isna(name):
        return None
    s = str(name)

    tokens = [t.strip() for t in s.split(";") if t.strip()]
    for t in tokens:
        upper = t.upper()
        if upper.endswith("_HUMAN"):
            return "HUMAN"
        if upper.endswith("_MOUSE"):
            return "MOUSE"
        if upper.endswith("_YEAST"):
            return "YEAST"
        if upper.endswith("_ECOLI") or upper.endswith("_ECOL"):
            return "ECOLI"

    if "_" in s:
        tail = s.split("_")[-1].upper()
        if len(tail) >= 3:
            return tail

    return None


def longest_common_prefix(strs):
    """Find longest common prefix of a list of strings."""
    if not strs:
        return ""
    s1, s2 = min(strs), max(strs)
    for i, (c1, c2) in enumerate(zip(s1, s2)):
        if c1 != c2:
            return s1[:i]
    return s1


def generate_default_names(n_cols: int, samples_per_condition: int = 3) -> list:
    """
    Generate default column names in format A1, A2, A3, B1, B2, B3, etc.
    
    Args:
        n_cols: Total number of columns to name
        samples_per_condition: Number of replicates per condition (default 3)
    
    Returns:
        List of default names
    """
    names = []
    condition_letter = ord('A')  # Start with 'A'
    
    for i in range(n_cols):
        # Determine which condition (A, B, C, etc.)
        condition_idx = i // samples_per_condition
        # Determine replicate number (1, 2, 3, etc.)
        replicate_num = (i % samples_per_condition) + 1
        
        # Generate name like "A1", "A2", "B1", etc.
        condition = chr(condition_letter + condition_idx)
        name = f"{condition}{replicate_num}"
        names.append(name)
    
    return names


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(page_title="Data Upload", layout="wide")

with st.sidebar:
    st.title("📊 Upload Settings")
    st.info(
        """
        **Supported Formats:**
        - CSV (.csv)
        - TSV (.tsv, .txt)
        - Excel (.xlsx)
        """
    )

# ============================================================================
# MAIN PAGE
# ============================================================================

st.title("📊 Data Upload & Configuration")

st.markdown(
    """
    Upload your proteomics data file and configure analysis settings.
    """
)

# ============================================================================
# STEP 1: FILE UPLOAD
# ============================================================================

st.subheader("1️⃣ Upload File")

uploaded_file = st.file_uploader(
    "Choose a proteomics data file",
    type=["csv", "tsv", "txt", "xlsx"],
    help="Supports CSV, TSV, and Excel formats",
)

if uploaded_file is None:
    st.warning("⚠️ Please upload a data file to continue")
    st.stop()

# ============================================================================
# STEP 2: READ FILE
# ============================================================================

st.subheader("2️⃣ Loading Data...")

try:
    filename = uploaded_file.name.lower()
    
    # Use unified read_file function from helpers.io
    df = read_file(uploaded_file)
    
    # Determine file format for display
    if filename.endswith(".xlsx"):
        file_format = "Excel"
    elif filename.endswith(".tsv") or filename.endswith(".txt"):
        file_format = "TSV"
    else:
        file_format = "CSV"

    st.success(
        f"✅ Loaded {file_format} file: {len(df):,} rows × {len(df.columns)} columns"
    )

except Exception as e:
    st.error(f"❌ Error reading file: {str(e)}")
    st.stop()

# Clean column names immediately
df.columns = [clean_column_name(col) for col in df.columns]

# Auto-trim shared left prefix from long run-specific columns
long_cols = [c for c in df.columns if len(c) > 40]
prefix = longest_common_prefix(long_cols)

if prefix and len(prefix) >= 20:
    st.info(
        f"🧹 Removing shared prefix from {len(long_cols)} columns:\n\n`{prefix}`"
    )
    new_cols = []
    for c in df.columns:
        if c in long_cols and c.startswith(prefix):
            new_cols.append(c[len(prefix):].lstrip("_").lstrip("."))
        else:
            new_cols.append(c)
    df.columns = new_cols

# ============================================================================
# STEP 3: FILL NaN AND 0 WITH 1
# ============================================================================

st.subheader("3️⃣ Data Cleaning")

n_nan_before = df.isna().sum().sum()
n_zero_before = (df == 0).sum().sum()

st.info("Replacing NaN and 0 values with 1.0 for log transformation compatibility...")

numeric_cols_detected = detect_numeric_columns(df)

for col in numeric_cols_detected:
    df[col] = df[col].fillna(1.0)
    df.loc[df[col] == 0, col] = 1.0

n_nan_after = df.isna().sum().sum()
n_zero_after = (df == 0).sum().sum()

c1, c2 = st.columns(2)
with c1:
    st.metric("NaN replaced", n_nan_before, delta=f"-{n_nan_before}")
with c2:
    st.metric("Zeros replaced", n_zero_before, delta=f"-{n_zero_before}")

st.success("✅ Data cleaning complete")

# ============================================================================
# STEP 4: SELECT QUANTITATIVE COLUMNS (CHECKBOX TABLE)
# ============================================================================

st.subheader("4️⃣ Select Quantitative Columns")

st.markdown(
    """
    **Check the columns you want to use for quantitative analysis.**
    Only float-type columns (float64/float32) are selected by default.
    """
)

df_cols = pd.DataFrame(
    {
        "Select": [col in numeric_cols_detected for col in df.columns],
        "Column": df.columns.tolist(),
        "Type": [str(df[col].dtype) for col in df.columns],
        "Sample": [
            str(df[col].iloc[0])[:30] if len(df) > 0 else "" for col in df.columns
        ],
    }
)

st.info("💡 **Click checkboxes to select/deselect columns**")

edited_cols = st.data_editor(
    df_cols,
    column_config={
        "Select": st.column_config.CheckboxColumn(
            "✓ Use",
            help="Check to include in analysis",
            default=False,
            width="small",
        ),
        "Column": st.column_config.TextColumn(
            "Column Name", width="large", disabled=True
        ),
        "Type": st.column_config.TextColumn(
            "Data Type", width="small", disabled=True
        ),
        "Sample": st.column_config.TextColumn(
            "Sample Value", width="medium", disabled=True
        ),
    },
    hide_index=True,
    use_container_width=True,
    key="column_selector_table",
)

selected_numeric_cols = edited_cols[edited_cols["Select"]]["Column"].tolist()

if len(selected_numeric_cols) < 4:
    st.warning(
        f"⚠️ Need at least 4 quantitative columns for analysis. You selected {len(selected_numeric_cols)}."
    )
    st.stop()

numeric_cols = selected_numeric_cols
st.success(f"✅ Selected {len(numeric_cols)} quantitative columns")

# ============================================================================
# STEP 5: RENAME COLUMNS (DEFAULT TO A1, A2, B1, B2...)
# ============================================================================

st.subheader("5️⃣ Rename Columns")

st.markdown(
    """
    **Default naming:** Columns will be renamed to A1, A2, A3, B1, B2, B3, etc.
    You can customize the number of replicates per condition and edit individual names.
    """
)

rename_dict = {}

rc1, rc2 = st.columns(2)
with rc1:
    samples_per_condition = st.number_input(
        "Replicates per condition:",
        min_value=1,
        max_value=10,
        value=3,
        help="How many samples per experimental condition? (e.g., 3 means A1-A3, B1-B3, etc.)"
    )

with rc2:
    use_default_names = st.checkbox(
        "Use default names (A1, A2, B1, B2...)",
        value=True,
        help="Automatically generate standard condition names"
    )

if use_default_names:
    # Generate default names
    default_names = generate_default_names(len(numeric_cols), samples_per_condition)
    
    # Create initial rename mapping
    rename_dict = dict(zip(numeric_cols, default_names))
    
    st.info(f"✅ Generated {len(default_names)} default names")
    
    # Show preview with ability to edit
    with st.expander("📋 Preview & Edit Names"):
        st.markdown("**Original → New Name** (edit if needed)")
        
        # Create editable table
        for idx, (old_col, new_name) in enumerate(rename_dict.items()):
            c1, c2, c3 = st.columns([2, 1, 2])
            
            with c1:
                st.text(old_col)
            with c2:
                st.write("→")
            with c3:
                custom_name = st.text_input(
                    "New name",
                    value=new_name,
                    label_visibility="collapsed",
                    key=f"rename_{idx}",
                )
                if custom_name != new_name and custom_name.strip():
                    rename_dict[old_col] = custom_name
else:
    # Manual renaming (original behavior)
    st.markdown("**Original → New Name**")

    for col in numeric_cols:
        c1, c2, c3 = st.columns([2, 1, 2])

        with c1:
            st.text(col)
        with c2:
            st.write("→")
        with c3:
            new_name = st.text_input(
                "New name",
                value=col,
                label_visibility="collapsed",
                key=f"rename_manual_{col}",
            )
            if new_name != col and new_name.strip():
                rename_dict[col] = new_name

# Apply renaming
if rename_dict:
    df = df.rename(columns=rename_dict)
    numeric_cols = [rename_dict.get(col, col) for col in numeric_cols]
    st.success(f"✅ Renamed {len(rename_dict)} columns")

    with st.expander("📋 View Final Column Mapping"):
        mapping_df = pd.DataFrame(
            {"Original": list(rename_dict.keys()), "New": list(rename_dict.values())}
        )
        st.dataframe(mapping_df, use_container_width=True)

# ============================================================================
# STEP 6: IDENTIFY METADATA COLUMNS
# ============================================================================

st.subheader("6️⃣ Identify Metadata Columns")

st.markdown("Select columns for protein/peptide ID and species (optional).")

non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

mc1, mc2 = st.columns(2)

with mc1:
    protein_id_col = detect_protein_id_column(df)
    if protein_id_col not in non_numeric_cols and len(non_numeric_cols) > 0:
        protein_id_col = non_numeric_cols[0]

    if len(non_numeric_cols) > 0:
        protein_id_col = st.selectbox(
            "🔍 Protein/Peptide ID column",
            options=non_numeric_cols,
            index=non_numeric_cols.index(protein_id_col)
            if protein_id_col in non_numeric_cols
            else 0,
        )
    else:
        st.warning("⚠️ No non-numeric columns available for protein ID")
        protein_id_col = df.columns[0]

with mc2:
    species_col_auto = detect_species_column(df)
    species_options = ["(None)"] + non_numeric_cols

    default_idx = 0
    if species_col_auto and species_col_auto in non_numeric_cols:
        default_idx = species_options.index(species_col_auto)

    species_col = st.selectbox(
        "🧬 Species column (optional; leave '(None)' to infer from protein names)",
        options=species_options,
        index=default_idx,
    )

    if species_col == "(None)":
        species_col = None
    elif species_col:
        df[species_col] = df[species_col].apply(clean_species_name)

st.success("✅ Metadata columns identified")

# ============================================================================
# STEP 6.5: DROP UNUSED COLUMNS
# ============================================================================

st.subheader("6️⃣.5 Cleaning Dataset")

columns_to_keep = list(numeric_cols)
if protein_id_col:
    columns_to_keep.insert(0, protein_id_col)
if species_col:
    columns_to_keep.append(species_col)

columns_to_remove = [c for c in df.columns if c not in columns_to_keep]

if columns_to_remove:
    st.info(f"ℹ️ Removing {len(columns_to_remove)} unused columns")

df = df[columns_to_keep]
st.success(f"✅ Keeping {len(columns_to_keep)} columns")

# ============================================================================
# STEP 7: PREVIEW DATA (10 ROWS)
# ============================================================================

st.subheader("7️⃣ Data Preview (First 10 Rows)")

# Ensure column names are unique for Streamlit/pyarrow
if df.columns.duplicated().any():
    cols = []
    seen = {}
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            cols.append(c)
    df.columns = cols

df_display = df.head(10).copy()
for col in df_display.columns:
    if pd.api.types.is_float_dtype(df_display[col]):
        df_display[col] = df_display[col].apply(
            lambda x: "" if pd.isna(x) else f"{x:.2f}"
        )
    else:
        df_display[col] = df_display[col].astype(str).replace("nan", "")

st.dataframe(df_display, use_container_width=True, height=350)

# ============================================================================
# STEP 8: BASIC STATISTICS
# ============================================================================

st.subheader("8️⃣ Basic Statistics")

# Create species mapping
if species_col:
    species_series = df[species_col]
else:
    # infer species from protein ID column (e.g. PG.ProteinNames)
    inferred_species = df[protein_id_col].apply(infer_species_from_protein_name)
    species_series = inferred_species
    df["__INFERRED_SPECIES__"] = inferred_species
    species_col = "__INFERRED_SPECIES__"

species_mapping = dict(zip(df[protein_id_col], species_series))

n_proteins = len(df)
n_samples = len(numeric_cols)
n_conditions = max(1, n_samples // samples_per_condition)

missing_count = 0
for col in numeric_cols:
    missing_count += df[col].isna().sum()
    missing_count += (df[col] == 1.0).sum()

missing_rate = (missing_count / (n_proteins * n_samples) * 100)

bc1, bc2, bc3, bc4 = st.columns(4)

with bc1:
    st.metric("Total Proteins", f"{n_proteins:,}")
with bc2:
    st.metric("Quantitative Samples", n_samples)
with bc3:
    st.metric("Estimated Conditions", n_conditions)
with bc4:
    st.metric("Missing Values %", f"{missing_rate:.1f}%")

# ============================================================================
# STEP 9: FINALIZE WITH OPTIONAL CLEANING
# ============================================================================

st.subheader("9️⃣ Finalizing...")

# Optional cleaning: drop proteins with any NaN or 1.00 in intensity columns
drop_invalid = st.checkbox(
    "Drop proteins with any NaN or 1.00 in selected intensity columns",
    value=False,
    help=(
        "If enabled, any protein with at least one missing (NaN) or 1.00 value "
        "in the selected intensity columns will be removed."
    ),
)

rows_dropped = 0
if drop_invalid:
    before_rows = len(df)
    df = drop_proteins_with_invalid_intensities(
        df=df,
        intensity_cols=numeric_cols,
        drop_value=1.0,
        drop_nan=True,
    )
    after_rows = len(df)
    rows_dropped = before_rows - after_rows
    
    st.info(
        f"Dropped {rows_dropped} proteins with at least one NaN or 1.00 intensity. "
        f"Remaining: {after_rows} proteins."
    )
    
    # Recalculate statistics after dropping
    n_proteins = len(df)
    species_mapping = dict(zip(df[protein_id_col], df[species_col]))
    
    missing_count = 0
    for col in numeric_cols:
        missing_count += df[col].isna().sum()
        missing_count += (df[col] == 1.0).sum()
    
    missing_rate = (missing_count / (n_proteins * n_samples) * 100) if n_proteins > 0 else 0.0
    
    # Display updated statistics
    st.markdown("**Updated Statistics After Filtering:**")
    
    uc1, uc2, uc3, uc4 = st.columns(4)
    
    with uc1:
        st.metric("Total Proteins", f"{n_proteins:,}", delta=f"-{rows_dropped}")
    with uc2:
        st.metric("Quantitative Samples", n_samples)
    with uc3:
        st.metric("Estimated Conditions", n_conditions)
    with uc4:
        st.metric("Missing Values %", f"{missing_rate:.1f}%")

# Now create ProteinData with the (possibly cleaned) df
protein_data = ProteinData(
    raw=df,
    numeric_cols=numeric_cols,
    species_col=species_col,
    species_mapping=species_mapping,
    index_col=protein_id_col,
    file_path=uploaded_file.name,
    file_format=file_format,
)

st.session_state.protein_data = protein_data
st.session_state.column_mapping = rename_dict

log_event(
    page="Data Upload",
    action=f"Uploaded {uploaded_file.name}",
    details={
        "filename": uploaded_file.name,
        "file_format": file_format,
        "n_proteins": int(n_proteins),
        "n_samples": int(n_samples),
        "missing_rate": float(missing_rate),
        "columns_selected": int(len(numeric_cols)),
        "columns_renamed": int(len(rename_dict)),
        "nan_replaced": int(n_nan_before),
        "zeros_replaced": int(n_zero_before),
        "proteins_dropped_any_invalid": int(rows_dropped),
    },
)

st.success("✅ Data loaded and configured successfully!")


# ============================================================================
# STEP 10: NEXT STEPS
# ============================================================================

st.markdown("---")

st.markdown(
    """
    ### ✨ Next Steps

    Your data is ready for analysis! Use the sidebar to navigate to:

    1. **📊 EDA** - Exploratory data analysis and visualization
    2. **🔬 Preprocessing** - Data transformation and normalization
    3. **🔍 Filtering** - Quality control and filtering
    4. **📈 Analysis** - Statistical analysis and results

    Your configuration is saved and will be available across all pages.
    """
)

st.info("💡 **Tip:** You can return to this page anytime to upload a different dataset.")
