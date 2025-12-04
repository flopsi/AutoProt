"""
pages/1_Data_Upload.py
Upload and explore proteomics data
Detects columns, species, and shows initial visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
from helpers.file_io import read_csv, read_tsv, read_excel, detect_numeric_columns, detect_protein_id_column, detect_species_column, validate_numeric_data
from helpers.dataclasses import ProteinData
from helpers.plots import create_density_plot
from helpers.audit import log_event
from helpers.peptide_protein import detect_data_level, aggregate_peptides_by_id

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
2. Identify protein/peptide IDs and species info
3. Validate data quality
4. Show initial visualization
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
    
    progress_bar.progress(25)
    
    st.success(f"✅ Loaded {file_format} file: {len(df):,} rows × {len(df.columns)} columns")
    
except Exception as e:
    st.error(f"❌ Error reading file: {str(e)}")
    st.stop()

# ============================================================================
# DETECT COLUMNS
# ============================================================================

st.subheader("3️⃣ Detecting Data Structure...")

# Numeric columns
numeric_cols = detect_numeric_columns(df)
st.write(f"📊 Found {len(numeric_cols)} numeric columns")

if len(numeric_cols) < 4:
    st.error(f"❌ Need at least 4 numeric columns for analysis, found {len(numeric_cols)}")
    st.stop()

progress_bar.progress(50)

# Protein ID column
protein_id_col = detect_protein_id_column(df)
st.write(f"🔍 Detected protein/peptide ID column: **{protein_id_col}**")

# Species column
species_col = detect_species_column(df)
if species_col:
    st.write(f"🧬 Detected species column: **{species_col}**")
else:
    st.write("🧬 No species column detected (optional)")

# Data level detection
data_level, reasoning = detect_data_level(df, protein_id_col)
st.write(f"📈 Data level: **{data_level.upper()}** ({reasoning})")

progress_bar.progress(75)

# ============================================================================
# OPTIONAL: MANUAL COLUMN SELECTION
# ============================================================================

with st.expander("🔧 Advanced: Manual Column Selection"):
    st.markdown("Override auto-detected columns if needed:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        protein_id_col = st.selectbox(
            "Protein/Peptide ID column",
            options=df.columns,
            index=list(df.columns).index(protein_id_col) if protein_id_col in df.columns else 0,
            key="protein_col_manual"
        )
    
    with col2:
        species_options = ["(None)"] + list(df.columns)
        species_col = st.selectbox(
            "Species column",
            options=species_options,
            index=0,
            key="species_col_manual"
        )
        if species_col == "(None)":
            species_col = None

# ============================================================================
# VALIDATE DATA
# ============================================================================

st.subheader("4️⃣ Data Validation")

is_valid, validation_msg = validate_numeric_data(df, numeric_cols)

if is_valid:
    st.success(f"✅ {validation_msg}")
else:
    st.error(f"❌ {validation_msg}")
    st.stop()

progress_bar.progress(90)

# ============================================================================
# CREATE PROTEIN DATA OBJECT
# ============================================================================

# Map species if available
species_mapping = {}
if species_col and species_col in df.columns:
    species_mapping = dict(zip(df[protein_id_col], df[species_col]))

# If peptide data, optionally aggregate
if data_level == "peptide":
    st.subheader("5️⃣ Peptide Data Detected - Aggregation Options")
    
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

# Create ProteinData object
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

progress_bar.progress(100)

# ============================================================================
# DISPLAY SUMMARY
# ============================================================================

st.subheader("6️⃣ Data Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Proteins/Peptides", f"{protein_data.n_proteins:,}")

with col2:
    st.metric("Samples", protein_data.n_samples)

with col3:
    st.metric("Conditions", protein_data.n_conditions)

with col4:
    st.metric("Missing %", f"{protein_data.missing_rate:.1f}%")

# ============================================================================
# DATA PREVIEW
# ============================================================================

st.subheader("7️⃣ Data Preview")

with st.expander("🔍 View First 10 Rows"):
    st.dataframe(
        df.head(10),
        use_container_width=True,
        height=300
    )

# ============================================================================
# INITIAL PLOT: RAW INTENSITY DISTRIBUTION
# ============================================================================

st.subheader("8️⃣ Initial Visualization - Raw Intensity Distribution")

st.markdown("""
The plot below shows the distribution of raw intensities across all samples.
Each sample's median, quartiles, and range are displayed.
""")

try:
    # Create density plot from raw data
    fig = create_density_plot(
        df[numeric_cols].mean(axis=1),
        fc_threshold=1.0,
        theme_name=theme
    )
    
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"❌ Error creating plot: {str(e)}")

# ============================================================================
# SPECIES BREAKDOWN (IF AVAILABLE)
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
    }
)

# ============================================================================
# SUCCESS MESSAGE & NEXT STEPS
# ============================================================================

st.success("✅ Data loaded successfully!")

st.markdown("""
### ✨ Next Steps

1. **Visual EDA** (2_Visual_EDA.py) - Explore data distribution
2. **Statistical EDA** (3_Statistical_EDA.py) - Check quality metrics
3. **Preprocessing** (4_Preprocessing.py) - Transform & filter
4. **Analysis** (6_Analysis.py) - Differential expression

Use the sidebar navigation to continue.
""")
