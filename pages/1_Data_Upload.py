"""
pages/1_Data_Upload.py

Data upload and validation page
Handles file parsing, column detection, data cleaning, and species annotation
"""

import streamlit as st
import pandas as pd
import numpy as np
from helpers.io import (
    read_file, detect_numeric_columns, detect_protein_id_column,
    detect_species_column, validate_numeric_data, clean_species_name
)
from helpers.core import ProteinData
from helpers.ui import show_data_summary, download_button_csv
from helpers.audit import log_file_upload
from helpers.viz import create_qc_dashboard
from helpers.core import get_theme
from helpers.analysis import detect_conditions_from_columns, group_columns_by_condition

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
    # SECTION: COLUMN RENAMING
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
    # SECTION: MANUAL SPECIES ANNOTATION
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
    # SECTION: DATA CLEANING
    # Replace missing/invalid values and filter by quality metrics
    # ============================================================================
    
    st.header("5️⃣ Data Cleaning (Optional)")
    
    with st.expander("🧹 Clean Data", expanded=False):
        st.markdown("""
        **Data Cleaning Strategy:**
        1. Replace NaN and zero values with **1.0** (placeholder for missing/invalid)
        2. Filter proteins by missing rate (count intensities == 1.0)
        3. Filter by coefficient of variation (CV) - overall and per-condition
        
        **Why 1.0?** Using 1.0 as a placeholder allows easy counting:
        - `sum(intensities == 1.0)` = number of missing values
        - Distinguishes missing from actual low intensities
        """)
        
        # Track original data
        n_proteins_original = len(df_raw)
        n_samples = len(selected_numeric)
        
        # --- Step 1: Replace NaN and 0 with 1.0 ---
        st.subheader("1. Replace Missing & Zero Values")
        
        col1, col2 = st.columns(2)
        
        with col1:
            replace_nan = st.checkbox(
                "Replace NaN with 1.0",
                value=True,
                help="Convert all missing values to 1.0"
            )
        
        with col2:
            replace_zero = st.checkbox(
                "Replace 0 with 1.0",
                value=True,
                help="Convert zero intensities to 1.0 (often invalid)"
            )
        
        # Preview impact
        if replace_nan or replace_zero:
            preview_stats = {
                "Metric": [],
                "Count": [],
                "Percentage": []
            }
            
            if replace_nan:
                n_nan = df_raw[selected_numeric].isna().sum().sum()
                preview_stats["Metric"].append("NaN values")
                preview_stats["Count"].append(n_nan)
                preview_stats["Percentage"].append(f"{n_nan / (len(df_raw) * n_samples) * 100:.2f}%")
            
            if replace_zero:
                n_zero = (df_raw[selected_numeric] == 0).sum().sum()
                preview_stats["Metric"].append("Zero values")
                preview_stats["Count"].append(n_zero)
                preview_stats["Percentage"].append(f"{n_zero / (len(df_raw) * n_samples) * 100:.2f}%")
            
            st.dataframe(
                pd.DataFrame(preview_stats),
                hide_index=True,
                use_container_width=True
            )
        
        if st.button("🔄 Apply Replacement", key="replace_btn"):
            df_cleaned = df_raw.copy()
            
            total_replaced = 0
            
            # Replace NaN with 1.0
            if replace_nan:
                n_nan_before = df_cleaned[selected_numeric].isna().sum().sum()
                df_cleaned[selected_numeric] = df_cleaned[selected_numeric].fillna(1.0)
                total_replaced += n_nan_before
                st.info(f"✅ Replaced {n_nan_before} NaN values with 1.0")
            
            # Replace 0 with 1.0
            if replace_zero:
                mask_zero = df_cleaned[selected_numeric] == 0
                n_zero = mask_zero.sum().sum()
                df_cleaned[selected_numeric] = df_cleaned[selected_numeric].mask(mask_zero, 1.0)
                total_replaced += n_zero
                st.info(f"✅ Replaced {n_zero} zero values with 1.0")
            
            df_raw = df_cleaned
            st.success(f"✅ Total replaced: {total_replaced} values ({total_replaced / (len(df_raw) * n_samples) * 100:.2f}%)")
            st.rerun()
        
        st.markdown("---")
        
        # --- Step 2: Filter by missing rate (count of 1.0 values) ---
        st.subheader("2. Filter by Missing Data Rate")
        
        st.markdown("""
        Remove proteins with too many missing/invalid values (intensities == 1.0).
        """)
        
        max_missing_rate = st.slider(
            "Maximum missing rate per protein:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            format="%.0f%%",
            help="Proteins exceeding this threshold will be removed"
        )
        
        # Calculate missing rate by counting 1.0 values
        count_ones = (df_raw[selected_numeric] == 1.0).sum(axis=1)
        missing_rate_per_protein = count_ones / n_samples
        n_would_remove = (missing_rate_per_protein > max_missing_rate).sum()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Proteins to remove",
                n_would_remove,
                delta=f"-{n_would_remove/len(df_raw)*100:.1f}%"
            )
        
        with col2:
            st.metric(
                "Proteins remaining",
                len(df_raw) - n_would_remove,
                delta=f"{(len(df_raw) - n_would_remove)/len(df_raw)*100:.1f}%"
            )
        
        # Show distribution of missing rates
        st.markdown("**Missing Rate Distribution**")
        missing_dist = pd.DataFrame({
            "Missing Rate": ["0-25%", "25-50%", "50-75%", "75-100%"],
            "Count": [
                ((missing_rate_per_protein >= 0) & (missing_rate_per_protein < 0.25)).sum(),
                ((missing_rate_per_protein >= 0.25) & (missing_rate_per_protein < 0.5)).sum(),
                ((missing_rate_per_protein >= 0.5) & (missing_rate_per_protein < 0.75)).sum(),
                (missing_rate_per_protein >= 0.75).sum(),
            ]
        })
        st.dataframe(missing_dist, hide_index=True, use_container_width=True)
        
        if st.button("🗑️ Apply Missing Rate Filter", key="missing_filter_btn"):
            # Keep proteins below threshold
            df_cleaned = df_raw[missing_rate_per_protein <= max_missing_rate].copy()
            
            n_removed = len(df_raw) - len(df_cleaned)
            st.success(f"✅ Removed {n_removed} proteins ({n_removed/len(df_raw)*100:.1f}%)")
            
            # Update dataframe
            df_raw = df_cleaned
            st.rerun()
        
        st.markdown("---")
        
        # --- Step 3: Filter by Coefficient of Variation ---
        st.subheader("3. Filter by Coefficient of Variation")
        
        st.markdown("""
        Remove proteins with high variability across samples.
        **CV = std / mean** (excludes intensities == 1.0 from calculation)
        """)
        
        # Detect conditions from column names
        conditions = detect_conditions_from_columns(selected_numeric)
        
        # Calculate CVs
        cv_data = {
            'Protein': [],
            'CV_Overall': [],
        }
        
        # Add per-condition CV columns
        for cond in conditions:
            cv_data[f'CV_{cond}'] = []
        
        for protein_id, row in df_raw.iterrows():
            cv_data['Protein'].append(protein_id)
            
            # --- Overall CV ---
            vals_overall = row[selected_numeric]
            valid_vals_overall = vals_overall[vals_overall != 1.0]
            
            if len(valid_vals_overall) >= 2:
                mean_val = valid_vals_overall.mean()
                std_val = valid_vals_overall.std()
                cv_overall = std_val / mean_val if mean_val > 0 else np.nan
            else:
                cv_overall = np.nan
            
            cv_data['CV_Overall'].append(cv_overall)
            
            # --- Per-condition CV ---
            for cond in conditions:
                cond_cols = group_columns_by_condition(selected_numeric, cond)
                
                if len(cond_cols) >= 2:
                    vals_cond = row[cond_cols]
                    valid_vals_cond = vals_cond[vals_cond != 1.0]
                    
                    if len(valid_vals_cond) >= 2:
                        mean_cond = valid_vals_cond.mean()
                        std_cond = valid_vals_cond.std()
                        cv_cond = std_cond / mean_cond if mean_cond > 0 else np.nan
                    else:
                        cv_cond = np.nan
                else:
                    cv_cond = np.nan
                
                cv_data[f'CV_{cond}'].append(cv_cond)
        
        # Create CV DataFrame
        cv_df = pd.DataFrame(cv_data).set_index('Protein')
        
        # --- Filter Options ---
        st.markdown("**Filter Strategy**")
        
        filter_strategy = st.radio(
            "Choose CV filter strategy:",
            options=[
                "Overall CV only",
                "Per-condition CV (all must pass)",
                "Per-condition CV (any must pass)",
                "Custom thresholds per condition"
            ],
            help="""
            - Overall: Filter by CV across all samples
            - All must pass: Protein must have low CV in ALL conditions
            - Any must pass: Protein must have low CV in AT LEAST ONE condition
            - Custom: Set different thresholds for each condition
            """
        )
        
        # --- Threshold Selection ---
        if filter_strategy == "Overall CV only":
            max_cv_overall = st.slider(
                "Maximum overall CV:",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="CV > 1.0 means std > mean (high variability)"
            )
            
            # Apply filter
            mask_pass = cv_df['CV_Overall'] <= max_cv_overall
            n_would_remove = (~mask_pass).sum()
        
        elif filter_strategy == "Per-condition CV (all must pass)":
            max_cv_condition = st.slider(
                "Maximum CV per condition:",
                min_value=0.1,
                max_value=2.0,
                value=0.8,
                step=0.1,
                help="All conditions must be below this threshold"
            )
            
            # Apply filter: all conditions must pass
            mask_pass = True
            for cond in conditions:
                mask_pass = mask_pass & (cv_df[f'CV_{cond}'] <= max_cv_condition)
            
            n_would_remove = (~mask_pass).sum()
        
        elif filter_strategy == "Per-condition CV (any must pass)":
            max_cv_condition = st.slider(
                "Maximum CV per condition:",
                min_value=0.1,
                max_value=2.0,
                value=0.8,
                step=0.1,
                help="At least one condition must be below this threshold"
            )
            
            # Apply filter: any condition can pass
            mask_pass = False
            for cond in conditions:
                mask_pass = mask_pass | (cv_df[f'CV_{cond}'] <= max_cv_condition)
            
            n_would_remove = (~mask_pass).sum()
        
        else:  # Custom thresholds
            st.markdown("**Set Custom Thresholds**")
            
            custom_thresholds = {}
            cols = st.columns(len(conditions))
            
            for idx, cond in enumerate(conditions):
                with cols[idx]:
                    custom_thresholds[cond] = st.number_input(
                        f"Max CV for {cond}:",
                        min_value=0.1,
                        max_value=2.0,
                        value=1.0,
                        step=0.1,
                        key=f"cv_threshold_{cond}"
                    )
            
            # Apply custom filters (all must pass)
            mask_pass = True
            for cond, threshold in custom_thresholds.items():
                mask_pass = mask_pass & (cv_df[f'CV_{cond}'] <= threshold)
            
            n_would_remove = (~mask_pass).sum()
        
        # --- Preview Impact ---
        st.markdown("**Filter Impact Preview**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Proteins to remove",
                int(n_would_remove),
                delta=f"-{n_would_remove/len(cv_df)*100:.1f}%"
            )
        
        with col2:
            if len(cv_df['CV_Overall'].dropna()) > 0:
                st.metric(
                    "Median Overall CV",
                    f"{cv_df['CV_Overall'].median():.2f}",
                    help="Lower is better (less variable)"
                )
        
        # --- CV Statistics Table ---
        st.markdown("**CV Statistics by Condition**")
        
        cv_stats = pd.DataFrame({
            'Condition': ['Overall'] + conditions,
            'Mean CV': [cv_df['CV_Overall'].mean()] + [cv_df[f'CV_{cond}'].mean() for cond in conditions],
            'Median CV': [cv_df['CV_Overall'].median()] + [cv_df[f'CV_{cond}'].median() for cond in conditions],
            'Std CV': [cv_df['CV_Overall'].std()] + [cv_df[f'CV_{cond}'].std() for cond in conditions],
            'Min CV': [cv_df['CV_Overall'].min()] + [cv_df[f'CV_{cond}'].min() for cond in conditions],
            'Max CV': [cv_df['CV_Overall'].max()] + [cv_df[f'CV_{cond}'].max() for cond in conditions],
        })
        
        st.dataframe(
            cv_stats.style.format({
                'Mean CV': '{:.3f}',
                'Median CV': '{:.3f}',
                'Std CV': '{:.3f}',
                'Min CV': '{:.3f}',
                'Max CV': '{:.3f}',
            }),
            hide_index=True,
            use_container_width=True
        )
        
        # --- CV Distribution Visualization ---
        with st.expander("📊 View CV Distributions", expanded=False):
            
            # Create distribution table
            st.markdown("**Overall CV Distribution**")
            cv_overall_clean = cv_df['CV_Overall'].dropna()
            
            cv_dist_overall = pd.DataFrame({
                "CV Range": ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0", "1.0-1.5", ">1.5"],
                "Count": [
                    ((cv_overall_clean >= 0) & (cv_overall_clean < 0.25)).sum(),
                    ((cv_overall_clean >= 0.25) & (cv_overall_clean < 0.5)).sum(),
                    ((cv_overall_clean >= 0.5) & (cv_overall_clean < 0.75)).sum(),
                    ((cv_overall_clean >= 0.75) & (cv_overall_clean < 1.0)).sum(),
                    ((cv_overall_clean >= 1.0) & (cv_overall_clean < 1.5)).sum(),
                    (cv_overall_clean >= 1.5).sum(),
                ]
            })
            cv_dist_overall['Percentage'] = (cv_dist_overall['Count'] / len(cv_overall_clean) * 100).round(1)
            
            st.dataframe(cv_dist_overall, hide_index=True, use_container_width=True)
            
            st.markdown("---")
            
            # Per-condition distributions
            for cond in conditions:
                st.markdown(f"**Condition {cond} - CV Distribution**")
                cv_cond_clean = cv_df[f'CV_{cond}'].dropna()
                
                if len(cv_cond_clean) > 0:
                    cv_dist_cond = pd.DataFrame({
                        "CV Range": ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0", "1.0-1.5", ">1.5"],
                        "Count": [
                            ((cv_cond_clean >= 0) & (cv_cond_clean < 0.25)).sum(),
                            ((cv_cond_clean >= 0.25) & (cv_cond_clean < 0.5)).sum(),
                            ((cv_cond_clean >= 0.5) & (cv_cond_clean < 0.75)).sum(),
                            ((cv_cond_clean >= 0.75) & (cv_cond_clean < 1.0)).sum(),
                            ((cv_cond_clean >= 1.0) & (cv_cond_clean < 1.5)).sum(),
                            (cv_cond_clean >= 1.5).sum(),
                        ]
                    })
                    cv_dist_cond['Percentage'] = (cv_dist_cond['Count'] / len(cv_cond_clean) * 100).round(1)
                    
                    st.dataframe(cv_dist_cond, hide_index=True, use_container_width=True)
                else:
                    st.warning(f"No valid CV data for condition {cond}")
        
        # --- Top Variable Proteins ---
        with st.expander("🔍 View Most Variable Proteins", expanded=False):
            st.markdown("**Top 20 Proteins by Overall CV**")
            
            top_variable = cv_df.nlargest(20, 'CV_Overall')[['CV_Overall'] + [f'CV_{cond}' for cond in conditions]]
            
            st.dataframe(
                top_variable.style.format({col: '{:.3f}' for col in top_variable.columns}),
                use_container_width=True
            )
        
        # --- Apply CV Filter ---
        if st.button("🗑️ Apply CV Filter", key="cv_filter_btn"):
            # Get proteins that pass the filter
            keep_proteins = cv_df[mask_pass].index
            df_cleaned = df_raw.loc[keep_proteins].copy()
            
            n_removed = len(df_raw) - len(df_cleaned)
            st.success(f"✅ Removed {n_removed} proteins ({n_removed/len(df_raw)*100:.1f}%)")
            
            # Show per-condition removal stats
            st.markdown("**Removal Details by Condition:**")
            removal_details = []
            
            for cond in conditions:
                cv_cond = cv_df[f'CV_{cond}']
                if filter_strategy == "Custom thresholds per condition":
                    threshold = custom_thresholds[cond]
                elif filter_strategy == "Overall CV only":
                    threshold = max_cv_overall
                else:
                    threshold = max_cv_condition
                
                n_fail = (cv_cond > threshold).sum()
                removal_details.append({
                    'Condition': cond,
                    'Threshold': f'{threshold:.2f}',
                    'Failed': int(n_fail),
                    'Pass Rate': f'{(1 - n_fail/len(cv_cond))*100:.1f}%'
                })
            
            st.dataframe(
                pd.DataFrame(removal_details),
                hide_index=True,
                use_container_width=True
            )
            
            # Update dataframe
            df_raw = df_cleaned
            st.rerun()
        
        # --- Export CV Data ---
        st.markdown("**Export CV Data**")
        download_button_csv(
            cv_df,
            filename="cv_analysis.csv",
            label="📥 Download CV Data"
        )
        
        st.markdown("---")
        
        # --- Summary of all cleaning operations ---
        st.subheader("📊 Cleaning Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Original Proteins",
                n_proteins_original
            )
        
        with col2:
            st.metric(
                "Current Proteins",
                len(df_raw),
                delta=f"{len(df_raw) - n_proteins_original}"
            )
        
        with col3:
            retention_rate = len(df_raw) / n_proteins_original * 100
            st.metric(
                "Retention Rate",
                f"{retention_rate:.1f}%"
            )
        
        # Count current missing values (1.0)
        current_missing = (df_raw[selected_numeric] == 1.0).sum().sum()
        total_values = len(df_raw) * n_samples
        
        st.info(f"""
        **Current Data Quality:**
        - Missing/Invalid values (==1.0): {current_missing} ({current_missing/total_values*100:.2f}%)
        - Valid measurements: {total_values - current_missing} ({(total_values - current_missing)/total_values*100:.2f}%)
        """)
    
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
        # Missing per protein (count 1.0 values)
        missing_per_protein = (df_raw[selected_numeric] == 1.0).sum(axis=1) / len(selected_numeric) * 100
        
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
        # Missing per sample (count 1.0 values)
        missing_per_sample = (df_raw[selected_numeric] == 1.0).sum() / len(df_raw) * 100
        
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
            filename="cleaned_data.csv",
            label="📥 Download Cleaned Data"
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
