
"""
pages/1_Data_Upload.py

Data upload and validation page
Handles file parsing, column detection, and initial quality assessment
"""

import streamlit as st
import pandas as pd
from helpers.io import (
    read_file, detect_numeric_columns, detect_protein_id_column,
    detect_species_column, validate_numeric_data, clean_species_name
)
from helpers.core import ProteinData
from helpers.ui import show_data_summary, download_button_csv, metric_card
from helpers.audit import log_file_upload
from helpers.viz import create_qc_dashboard
from helpers.core import get_theme

# ============================================================================
# PAGE CONFIGURATION
# Title and description
# ============================================================================

st.title("📁 Data Upload & Validation")
st.markdown("""
Upload your proteomics data file (CSV, TSV, or Excel) and validate data quality.
The system will auto-detect numeric intensity columns, protein IDs, and species annotations.
""")

# ============================================================================
# SECTION: FILE UPLOAD
# File uploader widget with format validation
# ============================================================================

st.header("1️⃣ Upload Data File")

uploaded_file = st.file_uploader(
    "Choose a proteomics data file",
    type=["csv", "tsv", "txt", "xlsx"],
    help="Supported formats: CSV, TSV, TXT, XLSX (max 100 MB)"
)

if uploaded_file is not None:
    
    # --- Load file with caching ---
    try:
        with st.spinner("📖 Reading file..."):
            df_raw = read_file(uploaded_file)
            file_size = uploaded_file.size
        
        st.success(f"✅ File loaded: **{uploaded_file.name}** ({file_size / 1024:.1f} KB)")
        
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        st.stop()
    
    # ============================================================================
    # SECTION: COLUMN DETECTION
    # Auto-detect special columns and validate structure
    # ============================================================================
    
    st.header("2️⃣ Column Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Numeric Columns")
        numeric_cols = detect_numeric_columns(df_raw)
        
        if len(numeric_cols) > 0:
            st.success(f"Found **{len(numeric_cols)}** numeric columns")
            
            # Allow user to manually adjust selection
            selected_numeric = st.multiselect(
                "Select intensity columns to use:",
                options=numeric_cols,
                default=numeric_cols,
                help="Uncheck columns to exclude from analysis"
            )
            
            if len(selected_numeric) < 4:
                st.warning("⚠️ At least 4 samples recommended for statistical analysis")
        else:
            st.error("❌ No numeric columns detected")
            st.stop()
    
    with col2:
        st.subheader("Metadata Columns")
        
        # Detect protein ID column
        protein_col = detect_protein_id_column(df_raw)
        if protein_col:
            st.info(f"🔑 Protein ID column: **{protein_col}**")
        else:
            st.warning("⚠️ No protein ID column detected")
            protein_col = st.selectbox(
                "Manually select protein ID column:",
                options=df_raw.columns
            )
        
        # Detect species column
        species_col = detect_species_column(df_raw)
        if species_col:
            st.info(f"🧬 Species column: **{species_col}**")
            
            # Clean and show species counts
            species_clean = df_raw[species_col].apply(clean_species_name)
            species_counts = species_clean.value_counts()
            st.dataframe(species_counts, use_container_width=True)
        else:
            st.info("ℹ️ No species column detected (optional)")
    
    # ============================================================================
    # SECTION: DATA VALIDATION
    # Quality checks and summary statistics
    # ============================================================================
    
    st.header("3️⃣ Data Validation")
    
    # Run validation
    is_valid, message = validate_numeric_data(df_raw, selected_numeric)
    
    if is_valid:
        st.success(message)
    else:
        st.error(message)
        st.stop()
    
    # Show summary metrics
    st.subheader("Data Summary")
    show_data_summary(df_raw, selected_numeric)
    
    # Missing data analysis
    st.subheader("Missing Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Missing per protein
        missing_per_protein = df_raw[selected_numeric].isna().sum(axis=1) / len(selected_numeric) * 100
        
        st.markdown("**Missing Rate Distribution (per protein)**")
        st.dataframe({
            "Metric": ["Mean", "Median", "Max"],
            "Value": [
                f"{missing_per_protein.mean():.1f}%",
                f"{missing_per_protein.median():.1f}%",
                f"{missing_per_protein.max():.1f}%"
            ]
        }, hide_index=True, use_container_width=True)
    
    with col2:
        # Missing per sample
        missing_per_sample = df_raw[selected_numeric].isna().sum() / len(df_raw) * 100
        
        st.markdown("**Missing Rate per Sample**")
        st.dataframe(
            pd.DataFrame({
                "Sample": missing_per_sample.index,
                "Missing %": missing_per_sample.values.round(1)
            }).sort_values("Missing %", ascending=False).head(10),
            hide_index=True,
            use_container_width=True
        )
    
    # ============================================================================
    # SECTION: DATA PREVIEW
    # Show first rows and basic statistics
    # ============================================================================
    
    st.header("4️⃣ Data Preview")
    
    tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📊 Statistics", "📈 QC Dashboard"])
    
    with tab1:
        st.dataframe(df_raw.head(20), use_container_width=True, height=400)
        
        # Download option
        download_button_csv(
            df_raw,
            filename="raw_data.csv",
            label="📥 Download Raw Data"
        )
    
    with tab2:
        st.markdown("**Descriptive Statistics**")
        stats_df = df_raw[selected_numeric].describe().T
        st.dataframe(stats_df, use_container_width=True)
    
    with tab3:
        # Generate QC dashboard
        theme = get_theme(st.session_state.get("theme", "light"))
        fig_qc = create_qc_dashboard(
            df_raw,
            selected_numeric,
            theme_name=st.session_state.get("theme", "light")
        )
        st.plotly_chart(fig_qc, use_container_width=True)
    
    # ============================================================================
    # SECTION: SAVE TO SESSION STATE
    # Create ProteinData object and store in session
    # ============================================================================
    
    st.header("5️⃣ Confirm & Proceed")
    
    if st.button("✅ Confirm Data & Continue", type="primary", use_container_width=True):
        
        # Create species mapping if species column exists
        species_mapping = {}
        if species_col:
            for idx, row in df_raw.iterrows():
                species_mapping[idx] = clean_species_name(row[species_col])
        
        # Create ProteinData object
        protein_data = ProteinData(
            raw=df_raw.copy(),
            numeric_cols=selected_numeric,
            species_col=species_col,
            species_mapping=species_mapping,
            index_col=protein_col,
            file_path=uploaded_file.name,
            file_format=uploaded_file.name.split('.')[-1]
        )
        
        # Store in session state
        st.session_state.protein_data = protein_data
        
        # Log upload event
        log_file_upload(
            filename=uploaded_file.name,
            file_size=file_size,
            n_rows=len(df_raw),
            n_cols=len(df_raw.columns),
            numeric_cols=len(selected_numeric)
        )
        
        st.success("✅ Data saved to session! Navigate to **2_Visual_EDA** to continue.")
        st.balloons()

else:
    # Show instructions when no file uploaded
    st.info("👆 Please upload a data file to begin")
    
    st.markdown("---")
    
    st.subheader("📋 Data Format Requirements")
    
    st.markdown("""
    Your data file should contain:
    
    1. **Protein/Gene ID column** (text)
       - Examples: "Protein ID", "Gene Name", "Accession"
    
    2. **Numeric intensity columns** (numbers)
       - One column per sample
       - Recommended naming: A1, A2, B1, B2 (letter = condition, number = replicate)
    
    3. **Optional: Species column** (text)
       - For multi-species experiments
       - Values: HUMAN, YEAST, ECOLI, etc.
    
    **Example structure:**
    
    | Protein | Species | A1 | A2 | A3 | B1 | B2 | B3 |
    |---------|---------|----|----|----|----|----|----|
    | P12345  | HUMAN   | 1500 | 1600 | 1550 | 2000 | 2100 | 2050 |
    | P67890  | YEAST   | 800  | 850  | 820  | 400  | 420  | 410  |
    """)

# ============================================================================
# FOOTER
# Navigation hints
# ============================================================================

st.markdown("---")
st.caption("**Next Step:** After confirming data, proceed to **📊 2_Visual_EDA** for transformation and exploration.")
