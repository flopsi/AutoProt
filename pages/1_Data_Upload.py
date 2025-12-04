"""
pages/1_Data_Upload_Enhanced_v2.py
Upload with: column selection, renaming, file name cleaning, and column cleanup
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from helpers.file_io import read_csv, read_tsv, read_excel, detect_numeric_columns, detect_protein_id_column, detect_species_column, validate_numeric_data
from helpers.dataclasses import ProteinData
from helpers.plots import create_density_plot
from helpers.audit import log_event
from helpers.peptide_protein import detect_data_level, aggregate_peptides_by_id

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_species_name(name: str) -> str:
    """Remove leading/trailing underscores from species names."""
    if pd.isna(name):
        return name
    return str(name).strip().strip('_').upper()


def clean_column_name(name: str) -> str:
    """
    Clean column names: remove special chars, standardize spacing.
    Example: "Sample_001 (Q1)" → "Sample_001_Q1"
    """
    # Remove parentheses and brackets
    name = re.sub(r'[\(\)\[\]\{\}]', '', name)
    
    # Replace multiple spaces with single space
    name = re.sub(r'\s+', ' ', name)
    
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    
    # Remove leading/trailing underscores
    name = name.strip('_')
    
    return name


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(page_title="Data Upload", layout="wide")

# ============================================================================
# SIDEBAR: THEME & HELP
# ============================================================================

with st.sidebar:
    st.title("📊 Upload Settings")
    
    theme = st.session_state.get("theme", "light")
    
    st.info("""
    **Supported Formats:**
    - CSV (.csv)
    - TSV (.tsv, .txt)
    - Excel (.xlsx)
    """)

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("📊 Data Upload")

st.markdown("""
Upload your proteomics data file. The system will:
1. Detect data format and structure
2. Let you select quantitative columns
3. Clean column names
4. Optionally rename columns
5. Remove unused columns (keep analysis-relevant only)
6. Identify protein/peptide IDs and species
7. Validate data quality
""")

# ============================================================================
# FILE UPLOADER
# ============================================================================

st.subheader("1️⃣ Upload File")

uploaded_file = st.file_uploader(
    "Choose a proteomics data file",
    type=["csv", "tsv", "txt", "xlsx"],
    help="Supports CSV, TSV, and Excel formats"
)

if uploaded_file is None:
    st.warning("⚠️ Please upload a data file to continue")
    st.stop()

# ============================================================================
# READ FILE
# ============================================================================

st.subheader("2️⃣ Loading Data...")

progress_bar = st.progress(0)

try:
    # Detect file format and read
    filename = uploaded_file.name.lower()
    
    if filename.endswith('.xlsx'):
        df = read_excel(uploaded_file)
        file_format = "Excel"
    elif filename.endswith('.tsv') or filename.endswith('.txt'):
        df = read_tsv(uploaded_file)
        file_format = "TSV"
    else:
        df = read_csv(uploaded_file)
        file_format = "CSV"
    
    progress_bar.progress(20)
    
    st.success(f"✅ Loaded {file_format} file: {len(df):,} rows × {len(df.columns)} columns")
    
except Exception as e:
    st.error(f"❌ Error reading file: {str(e)}")
    st.stop()

# ============================================================================
# STEP 3: CLEAN COLUMN NAMES
# ============================================================================

st.subheader("4️⃣ Select Quantitative Columns")

st.markdown("""
Select which columns contain quantitative measurements (intensities, abundances, etc.).
Non-numeric columns will be excluded from analysis.
""")


# Create dataframe with checkboxes
df_col_select = pd.DataFrame({
    "Select": [col in default_numeric for col in all_cols],
    "Column": all_cols,
    "Type": [str(df[col].dtype) for col in all_cols],
})

st.info("💡 **Check columns to include in analysis.**")

# Interactive checkbox table
edited_cols = st.data_editor(
    df_col_select,
    column_config={
        "Select": st.column_config.CheckboxColumn(
            "✓ Include",
            help="Check to include this column"
        ),
        "Column": st.column_config.TextColumn(
            "Column Name",
            width="large",
            disabled=True
        ),
        "Type": st.column_config.TextColumn(
            "Data Type",
            width="small",
            disabled=True
        ),
    },
    hide_index=True,
    use_container_width=True,
    key="column_selector_table"
)

# Extract selected columns
selected_numeric_cols = edited_cols[edited_cols["Select"]]["Column"].tolist()

if len(selected_numeric_cols) < 4:
    st.warning(f"⚠️ Need at least 4 columns. You selected {len(selected_numeric_cols)}.")
    st.stop()

numeric_cols = selected_numeric_cols
st.success(f"✅ Selected {len(numeric_cols)} columns")

# ============================================================================
# STEP 5: RENAME COLUMNS (OPTIONAL)
# ============================================================================

st.subheader("5️⃣ Rename Columns (Optional)")

st.markdown("**Rename columns to meaningful names.** You can keep originals or create new names.")

rename_dict = {}
rename_cols = st.columns(2)

with rename_cols[0]:
    should_rename = st.checkbox(
        "Enable column renaming?",
        value=False,
        help="Rename quantitative columns for clarity"
    )

with rename_cols[1]:
    if should_rename:
        st.info("Enter new names in the text boxes below")

if should_rename:
    st.markdown("**Original → New Name**")
    
    for col in numeric_cols:
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            st.write(f"**{col}**")
        
        with col2:
            st.write("→")
        
        with col3:
            new_name = st.text_input(
                "New name",
                value=col,
                label_visibility="collapsed",
                key=f"rename_{col}"
            )
            if new_name != col:
                rename_dict[col] = new_name
    
    # Apply renaming
    if rename_dict:
        df = df.rename(columns=rename_dict)
        numeric_cols = [rename_dict.get(col, col) for col in numeric_cols]
        st.success(f"✅ Renamed {len(rename_dict)} columns")

progress_bar.progress(50)

# ============================================================================
# STEP 6: IDENTIFY METADATA COLUMNS & CLEAN SPECIES
# ============================================================================

st.subheader("6️⃣ Identify Metadata Columns")

st.markdown("Select columns for protein/peptide ID and species (optional).")

col1, col2 = st.columns(2)

# Get non-numeric columns for ID/species
non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

with col1:
    protein_id_col = detect_protein_id_column(df)
    if protein_id_col not in df.columns:
        protein_id_col = non_numeric_cols[0] if non_numeric_cols else None
    
    protein_id_col = st.selectbox(
        "🔍 Protein/Peptide ID column",
        options=non_numeric_cols,
        index=non_numeric_cols.index(protein_id_col) if protein_id_col in non_numeric_cols else 0,
    )

with col2:
    species_col = detect_species_column(df)
    species_options = ["(None)"] + non_numeric_cols
    
    default_idx = 0
    if species_col and species_col in df.columns:
        default_idx = species_options.index(species_col)
        # CLEAN SPECIES NAMES
        df[species_col] = df[species_col].apply(clean_species_name)
    
    species_col = st.selectbox(
        "🧬 Species column",
        options=species_options,
        index=default_idx,
    )
    
    if species_col == "(None)":
        species_col = None
    elif species_col:
        # Clean species names if selected
        df[species_col] = df[species_col].apply(clean_species_name)

progress_bar.progress(60)

# ============================================================================
# STEP 7: REMOVE UNUSED COLUMNS
# ============================================================================

st.subheader("7️⃣ Remove Unused Columns")

st.markdown("""
**Columns to keep for analysis:**
- Quantitative columns (measurements)
- Protein/Peptide ID column
- Species column (if selected)

**All other columns will be removed** to keep data clean and focused.
""")

# Determine which columns to keep
columns_to_keep = list(numeric_cols)
if protein_id_col and protein_id_col not in columns_to_keep:
    columns_to_keep.insert(0, protein_id_col)
if species_col and species_col not in columns_to_keep:
    columns_to_keep.append(species_col)

# Show what will be removed
columns_to_remove = [col for col in df.columns if col not in columns_to_keep]

if columns_to_remove:
    st.info(f"ℹ️ Will remove {len(columns_to_remove)} unused columns: {', '.join(columns_to_remove[:5])}")
    if len(columns_to_remove) > 5:
        st.write(f"   ... and {len(columns_to_remove) - 5} more")

# Remove unused columns
df = df[columns_to_keep]

st.success(f"✅ Keeping {len(columns_to_keep)} analysis-relevant columns")
st.write(f"   Columns: {', '.join(columns_to_keep)}")

progress_bar.progress(70)

# ============================================================================
# STEP 8: DATA VALIDATION
# ============================================================================

st.subheader("8️⃣ Data Validation")

is_valid, validation_msg = validate_numeric_data(df, numeric_cols)

if is_valid:
    st.success(f"✅ {validation_msg}")
else:
    st.error(f"❌ {validation_msg}")
    st.stop()

progress_bar.progress(75)

# ============================================================================
# STEP 9: PEPTIDE DATA DETECTION & AGGREGATION
# ============================================================================

data_level, reasoning = detect_data_level(df, protein_id_col)

if data_level == "peptide":
    st.subheader("9️⃣ Peptide Data Detected - Aggregation Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        should_aggregate = st.checkbox(
            "Aggregate to protein level?",
            value=True,
            help="Sum peptide intensities by protein ID"
        )
    
    with col2:
        agg_method = st.selectbox(
            "Aggregation method",
            options=["sum", "mean", "median", "max"],
            index=0,
            disabled=not should_aggregate,
            help="How to combine peptide intensities"
        )
    
    if should_aggregate:
        st.info("Aggregating peptides to protein level...")
        try:
            df_aggregated, agg_metadata = aggregate_peptides_by_id(
                df,
                peptide_id_col=protein_id_col,
                protein_id_col="Protein ID",
                numeric_cols=numeric_cols,
                method=agg_method
            )
            
            df = df_aggregated.reset_index()
            protein_id_col = "Protein ID"
            
            st.success(f"✅ Aggregated {agg_metadata['n_peptides_original']:,} peptides → {agg_metadata['n_proteins']} proteins")
            st.write(f"   Average: {agg_metadata['avg_peptides_per_protein']:.1f} peptides/protein")
        
        except Exception as e:
            st.error(f"❌ Aggregation failed: {str(e)}")
            st.stop()
    
    progress_bar.progress(80)
else:
    st.subheader("9️⃣ Data Level")
    st.info(f"📈 **Data level: PROTEIN** — {reasoning}")
    progress_bar.progress(80)

# ============================================================================
# STEP 10: CREATE PROTEIN DATA OBJECT
# ============================================================================

# Map species if available
species_mapping = {}
if species_col and species_col in df.columns:
    species_mapping = dict(zip(df[protein_id_col], df[species_col]))

protein_data = ProteinData(
    raw=df,
    numeric_cols=numeric_cols,
    species_col=species_col,
    species_mapping=species_mapping,
    index_col=protein_id_col,
    file_path=uploaded_file.name,
    file_format=file_format,
)

# Store in session
st.session_state.protein_data = protein_data
st.session_state.column_mapping = rename_dict  # Store for reference

progress_bar.progress(90)

# ============================================================================
# STEP 11: DISPLAY SUMMARY
# ============================================================================

st.subheader("🔟 Data Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Proteins", f"{protein_data.n_proteins:,}")

with col2:
    st.metric("Quantitative Samples", protein_data.n_samples)

with col3:
    st.metric("Inferred Conditions", protein_data.n_conditions)

with col4:
    st.metric("Missing %", f"{protein_data.missing_rate:.1f}%")

# ============================================================================
# STEP 12: DATA PREVIEW
# ============================================================================

st.subheader("1️⃣1️⃣ Data Preview")

with st.expander("🔍 View Data (First 10 Rows)"):
    st.dataframe(
        df.head(10),
        use_container_width=True,
        height=300
    )

# ============================================================================
# STEP 13: COLUMN MAPPING REFERENCE
# ============================================================================

if rename_dict:
    st.subheader("Column Name Mapping")
    mapping_df = pd.DataFrame({
        "Original Name": list(rename_dict.keys()),
        "New Name": list(rename_dict.values())
    })
    st.dataframe(mapping_df, use_container_width=True)

# ============================================================================
# STEP 14: INITIAL VISUALIZATION
# ============================================================================

st.subheader("1️⃣2️⃣ Initial Visualization - Raw Intensity Distribution")

st.markdown("""
The plot shows the distribution of raw intensities across all samples.
""")

try:
    fig = create_density_plot(
        df[numeric_cols].mean(axis=1),
        fc_threshold=1.0,
        theme_name=theme
    )
    
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"❌ Error creating plot: {str(e)}")

# ============================================================================
# STEP 15: SPECIES BREAKDOWN (IF AVAILABLE)
# ============================================================================

if species_mapping:
    st.subheader("Species Breakdown")
    
    species_counts = pd.Series(species_mapping).value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.bar_chart(species_counts)
    
    with col2:
        for species, count in species_counts.items():
            st.metric(species, count)

# ============================================================================
# AUDIT LOGGING
# ============================================================================

log_event(
    "Data Upload",
    f"Uploaded {uploaded_file.name}",
    {
        "filename": uploaded_file.name,
        "file_format": file_format,
        "n_proteins": protein_data.n_proteins,
        "n_samples": protein_data.n_samples,
        "n_conditions": protein_data.n_conditions,
        "missing_rate": float(protein_data.missing_rate),
        "data_level": data_level,
        "columns_selected": len(numeric_cols),
        "columns_renamed": len(rename_dict),
        "columns_removed": len([c for c in col_info['Column'] if c not in df.columns]),
    }
)

progress_bar.progress(100)

# ============================================================================
# SUCCESS MESSAGE & NEXT STEPS
# ============================================================================

st.success("✅ Data loaded, cleaned, and configured successfully!")

st.markdown("""
### ✨ Next Steps

1. **Visual EDA** (2_Visual_EDA.py) - Explore data distribution
2. **Statistical EDA** (3_Statistical_EDA.py) - Check quality metrics
3. **Preprocessing** (4_Preprocessing.py) - Transform & filter
4. **Analysis** (6_Analysis.py) - Differential expression

Use the sidebar navigation to continue. Your data is stored and available for all following pages.
""")
