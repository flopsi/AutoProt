"""
pages/1_Data_Upload.py

Data upload and validation page
Handles file parsing, column detection, data cleaning, and species annotation
"""

import streamlit as st
import pandas as pd
from helpers.io import (
    read_file, detect_numeric_columns, detect_protein_id_column,
    detect_species_column, validate_numeric_data, clean_species_name,
    drop_proteins_with_invalid_intensities, filter_by_missing_rate, filter_by_cv
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
    # SECTION: COLUMN RENAMING (NEW)
    # Allow user to rename columns for consistency
    # ============================================================================
    
    st.header("2️⃣ Column Renaming (Optional)")
    
    with st.expander("🏷️ Rename Columns", expanded=False):
        st.markdown("""
        Rename columns to standardize naming conventions before analysis.
        This is useful for:
        - Fixing inconsistent naming (e.g., "Sample 1" → "A1")
        - Removing special characters
        - Standardizing condition labels
        """)
        
        # Create two-column layout for renaming interface
        rename_map = {}
        
        # Option 1: Bulk rename with pattern
        st.subheader("Bulk Rename with Pattern")
        col1, col2 = st.columns(2)
        
        with col1:
            find_pattern = st.text_input(
                "Find pattern:",
                placeholder="e.g., 'Sample_'",
                help="Text to find in column names"
            )
        
        with col2:
            replace_pattern = st.text_input(
                "Replace with:",
                placeholder="e.g., ''",
                help="Text to replace with (leave empty to remove)"
            )
        
        if find_pattern:
            preview_renames = {
                col: col.replace(find_pattern, replace_pattern)
                for col in df_raw.columns
                if find_pattern in col
            }
            
            if preview_renames:
                st.info(f"📋 Preview: {len(preview_renames)} columns will be renamed")
                st.dataframe(
                    pd.DataFrame({
                        "Original": list(preview_renames.keys()),
                        "New Name": list(preview_renames.values())
                    }),
                    hide_index=True,
                    height=200
                )
                
                if st.button("✅ Apply Bulk Rename"):
                    df_raw.rename(columns=preview_renames, inplace=True)
                    st.success(f"✅ Renamed {len(preview_renames)} columns")
                    st.rerun()
        
        st.markdown("---")
        
        # Option 2: Individual column rename
        st.subheader("Individual Column Rename")
        
        selected_col_to_rename = st.selectbox(
            "Select column to rename:",
            options=[""] + list(df_raw.columns),
            help="Choose a column to give it a new name"
        )
        
        if selected_col_to_rename:
            new_col_name = st.text_input(
                f"New name for '{selected_col_to_rename}':",
                value=selected_col_to_rename,
                key="individual_rename"
            )
            
            if st.button("✅ Rename Column") and new_col_name != selected_col_to_rename:
                if new_col_name in df_raw.columns:
                    st.error(f"❌ Column '{new_col_name}' already exists")
                else:
                    df_raw.rename(columns={selected_col_to_rename: new_col_name}, inplace=True)
                    st.success(f"✅ Renamed '{selected_col_to_rename}' → '{new_col_name}'")
                    st.rerun()
    
    # ============================================================================
    # SECTION: COLUMN DETECTION
    # Auto-detect special columns and validate structure
    # ============================================================================
    
    st.header("3️⃣ Column Detection")
    
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
        
        # Allow manual selection
        protein_col = st.selectbox(
            "Protein ID column:",
            options=df_raw.columns,
            index=list(df_raw.columns).index(protein_col) if protein_col else 0,
            help="Column containing unique protein/gene identifiers"
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
            st.info("ℹ️ No species column detected")
    
    # ============================================================================
    # SECTION: MANUAL SPECIES ANNOTATION (NEW)
    # Add species column if not present
    # ============================================================================
    
    if species_col is None or species_col not in df_raw.columns:
        st.header("4️⃣ Add Species Annotation (Optional)")
        
        with st.expander("🧬 Manually Add Species Column", expanded=False):
            st.markdown("""
            If your data doesn't have a species column, you can add one manually.
            This is useful for multi-species experiments (e.g., HUMAN + YEAST + ECOLI spike-ins).
            """)
            
            annotation_method = st.radio(
                "Choose annotation method:",
                options=[
                    "Extract from Protein ID (pattern matching)",
                    "Upload separate annotation file",
                    "Manual entry by protein ID"
                ],
                help="Different ways to add species information"
            )
            
            if annotation_method == "Extract from Protein ID (pattern matching)":
                st.subheader("Pattern-Based Extraction")
                
                # Define patterns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Pattern → Species Mapping**")
                    patterns = {}
                    patterns['HUMAN'] = st.text_input("Pattern for HUMAN:", value="_HUMAN")
                    patterns['YEAST'] = st.text_input("Pattern for YEAST:", value="_YEAST")
                    patterns['ECOLI'] = st.text_input("Pattern for ECOLI:", value="_ECOLI")
                
                with col2:
                    st.markdown("**Preview Matches**")
                    preview_counts = {}
                    
                    for species, pattern in patterns.items():
                        if pattern:
                            count = df_raw[protein_col].str.contains(pattern, case=False, na=False).sum()
                            preview_counts[species] = count
                    
                    if preview_counts:
                        st.dataframe(
                            pd.DataFrame({
                                "Species": list(preview_counts.keys()),
                                "Matches": list(preview_counts.values())
                            }),
                            hide_index=True
                        )
                
                if st.button("✅ Apply Pattern-Based Annotation"):
                    species_col_new = "Species"
                    df_raw[species_col_new] = "UNKNOWN"
                    
                    for species, pattern in patterns.items():
                        if pattern:
                            mask = df_raw[protein_col].str.contains(pattern, case=False, na=False)
                            df_raw.loc[mask, species_col_new] = species
                    
                    species_col = species_col_new
                    st.success(f"✅ Added species column with {(df_raw[species_col_new] != 'UNKNOWN').sum()} annotations")
                    st.rerun()
            
            elif annotation_method == "Upload separate annotation file":
                st.subheader("Upload Annotation File")
                
                st.markdown("""
                Upload a CSV file with two columns:
                1. Protein ID (matching your data)
                2. Species (HUMAN, YEAST, ECOLI, etc.)
                """)
                
                annotation_file = st.file_uploader(
                    "Upload annotation CSV:",
                    type=["csv"],
                    key="annotation_upload"
                )
                
                if annotation_file:
                    try:
                        annot_df = pd.read_csv(annotation_file)
                        
                        if len(annot_df.columns) < 2:
                            st.error("❌ File must have at least 2 columns")
                        else:
                            # Let user select columns
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                annot_id_col = st.selectbox(
                                    "Protein ID column:",
                                    options=annot_df.columns
                                )
                            
                            with col2:
                                annot_species_col = st.selectbox(
                                    "Species column:",
                                    options=annot_df.columns,
                                    index=1 if len(annot_df.columns) > 1 else 0
                                )
                            
                            # Preview merge
                            st.markdown("**Preview Annotation**")
                            st.dataframe(annot_df[[annot_id_col, annot_species_col]].head(10))
                            
                            if st.button("✅ Merge Annotations"):
                                # Create mapping
                                annot_map = dict(zip(
                                    annot_df[annot_id_col],
                                    annot_df[annot_species_col]
                                ))
                                
                                # Apply to dataframe
                                species_col_new = "Species"
                                df_raw[species_col_new] = df_raw[protein_col].map(annot_map).fillna("UNKNOWN")
                                
                                species_col = species_col_new
                                matched = (df_raw[species_col_new] != "UNKNOWN").sum()
                                st.success(f"✅ Merged annotations: {matched}/{len(df_raw)} proteins matched")
                                st.rerun()
                    
                    except Exception as e:
                        st.error(f"❌ Error reading annotation file: {e}")
            
            elif annotation_method == "Manual entry by protein ID":
                st.subheader("Manual Entry")
                st.warning("⚠️ This method is only practical for small datasets")
                
                # Show first 10 proteins for manual entry
                st.markdown("**Annotate Proteins**")
                
                manual_annotations = {}
                for idx, protein_id in enumerate(df_raw[protein_col].head(10)):
                    species_input = st.selectbox(
                        f"{protein_id}:",
                        options=["UNKNOWN", "HUMAN", "YEAST", "ECOLI", "MOUSE"],
                        key=f"manual_species_{idx}"
                    )
                    manual_annotations[protein_id] = species_input
                
                if st.button("✅ Apply Manual Annotations"):
                    species_col_new = "Species"
                    df_raw[species_col_new] = df_raw[protein_col].map(manual_annotations).fillna("UNKNOWN")
                    species_col = species_col_new
                    st.success(f"✅ Added {len(manual_annotations)} manual annotations")
                    st.rerun()
    
    # ============================================================================
    # SECTION: DATA CLEANING (NEW)
    # Drop invalid intensities and filter by quality metrics
    # ============================================================================
    
    st.header("5️⃣ Data Cleaning (Optional)")
    
    with st.expander("🧹 Clean Data", expanded=False):
        st.markdown("""
        Remove low-quality proteins based on:
        - Invalid intensity values (e.g., 1.0 placeholder values)
        - High missing data rate
        - High coefficient of variation (CV)
        """)
        
        # Track original size
        n_proteins_original = len(df_raw)
        
        # --- Option 1: Drop invalid intensities ---
        st.subheader("1. Remove Invalid Intensities")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            drop_nan = st.checkbox(
                "Drop rows with NaN",
                value=False,
                help="Remove proteins with any missing values"
            )
        
        with col2:
            drop_value_enabled = st.checkbox(
                "Drop specific value",
                value=False,
                help="Remove proteins with specific placeholder value"
            )
        
        with col3:
            drop_value = st.number_input(
                "Value to drop:",
                value=1.0,
                disabled=not drop_value_enabled,
                help="Common placeholder value (e.g., 1.0)"
            )
        
        if st.button("🗑️ Apply Invalid Value Filter"):
            df_cleaned = drop_proteins_with_invalid_intensities(
                df_raw,
                selected_numeric,
                drop_value=drop_value if drop_value_enabled else None,
                drop_nan=drop_nan
            )
            
            n_removed = n_proteins_original - len(df_cleaned)
            st.success(f"✅ Removed {n_removed} proteins ({n_removed/n_proteins_original*100:.1f}%)")
            df_raw = df_cleaned
            st.rerun()
        
        st.markdown("---")
        
        # --- Option 2: Filter by missing rate ---
        st.subheader("2. Filter by Missing Data Rate")
        
        max_missing_rate = st.slider(
            "Maximum missing rate per protein:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            format="%.0f%%",
            help="Proteins exceeding this threshold will be removed"
        )
        
        # Preview impact
        missing_per_protein = df_raw[selected_numeric].isna().sum(axis=1) / len(selected_numeric)
        n_would_remove = (missing_per_protein > max_missing_rate).sum()
        
        st.info(f"📊 Preview: {n_would_remove} proteins would be removed ({n_would_remove/len(df_raw)*100:.1f}%)")
        
        if st.button("🗑️ Apply Missing Rate Filter"):
            df_cleaned = filter_by_missing_rate(
                df_raw,
                selected_numeric,
                max_missing_rate=max_missing_rate
            )
            
            n_removed = len(df_raw) - len(df_cleaned)
            st.success(f"✅ Removed {n_removed} proteins")
            df_raw = df_cleaned
            st.rerun()
        
        st.markdown("---")
        
        # --- Option 3: Filter by CV ---
        st.subheader("3. Filter by Coefficient of Variation")
        
        max_cv = st.slider(
            "Maximum CV per protein:",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="CV = std/mean, measures relative variability"
        )
        
        # Preview impact
        means = df_raw[selected_numeric].mean(axis=1)
        stds = df_raw[selected_numeric].std(axis=1)
        cvs = stds / means
        n_would_remove_cv = (cvs > max_cv).sum()
        
        st.info(f"📊 Preview: {n_would_remove_cv} proteins would be removed ({n_would_remove_cv/len(df_raw)*100:.1f}%)")
        
        if st.button("🗑️ Apply CV Filter"):
            df_cleaned = filter_by_cv(
                df_raw,
                selected_numeric,
                max_cv=max_cv
            )
            
            n_removed = len(df_raw) - len(df_cleaned)
            st.success(f"✅ Removed {n_removed} proteins")
            df_raw = df_cleaned
            st.rerun()
    
    # ============================================================================
    # SECTION: DATA VALIDATION
    # Quality checks and summary statistics
    # ============================================================================
    
    st.header("6️⃣ Data Validation")
    
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
    
    st.header("7️⃣ Data Preview")
    
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
    
    st.header("8️⃣ Confirm & Proceed")
    
    if st.button("✅ Confirm Data & Continue", type="primary", use_container_width=True):
        
        # Create species mapping if species column exists
        species_mapping = {}
        if species_col and species_col in df_raw.columns:
            for idx, row in df_raw.iterrows():
                species_mapping[idx] = clean_species_name(row[species_col])
        
        # Create ProteinData object
        protein_data = ProteinData(
            raw=df_raw.copy(),
            numeric_cols=selected_numeric,
            species_col=species_col if species_col in df_raw.columns else None,
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
