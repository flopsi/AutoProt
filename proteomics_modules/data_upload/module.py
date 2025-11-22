"""
Main data upload module orchestrator - COMPLETE VERSION WITH NAME TRIMMING
"""

import streamlit as st
import pandas as pd
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid
import shutil


class DataUploadModule:
    """Main orchestrator for data upload workflow"""
    
    def __init__(self):
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables"""
        if 'upload_step' not in st.session_state:
            st.session_state.upload_step = 1
        
        if 'data_validated' not in st.session_state:
            st.session_state.data_validated = False
        
        if 'upload_complete' not in st.session_state:
            st.session_state.upload_complete = False
    
    def _trim_column_name(self, col_name: str) -> str:
        """
        Trim common prefixes/suffixes from column names
        
        Examples:
            '20240115_Sample1.raw.PG.Quantity' -> 'Sample1'
            'MP01_DIA_Sample2.PG.MaxLFQ' -> 'Sample2'
        """
        trimmed = col_name
        
        # Remove date prefix (e.g., '20240115_')
        trimmed = re.sub(r'^\d{8}_', '', trimmed)
        
        # Remove technical suffixes
        suffixes = [
            r'\.raw$',
            r'\.PG\.Quantity$',
            r'\.PG\.Normalized$',
            r'\.PG\.MaxLFQ$',
            r'\.Intensity$',
            r'\.LFQ$'
        ]
        for suffix in suffixes:
            trimmed = re.sub(suffix, '', trimmed)
        
        # Remove method prefixes (e.g., 'MP01_', 'DIA_', 'DDA_')
        trimmed = re.sub(r'^(MP\d+_|DIA_|DDA_|SPD_|LFQ_|IO\d+_)', '', trimmed)
        
        # Remove concentration (e.g., '100pg_')
        trimmed = re.sub(r'\d+pg_', '', trimmed)
        
        return trimmed.strip()
    
    def run(self):
        """Execute the complete data upload workflow"""
        
        st.title("🧬 Proteomics Data Upload")
        
        # Progress indicator
        self._render_progress()
        
        # Execute current step
        if st.session_state.upload_step == 1:
            self._step1_file_upload()
        
        elif st.session_state.upload_step == 2:
            self._step2_data_preview()
        
        elif st.session_state.upload_step == 3:
            self._step3_column_mapping()
        
        elif st.session_state.upload_step == 4:
            self._step4_sample_annotation()
        
        elif st.session_state.upload_step == 5:
            self._step5_workflow_suggestion()
        
        # Navigation buttons
        self._render_navigation()
    
    def _render_progress(self):
        """Render progress bar"""
        steps = ["Upload", "Preview", "Column Mapping", "Annotation", "Workflow"]
        
        progress = st.session_state.upload_step / len(steps)
        st.progress(progress)
        
        col_steps = st.columns(len(steps))
        for i, (col, step) in enumerate(zip(col_steps, steps)):
            with col:
                if i + 1 < st.session_state.upload_step:
                    st.markdown(f"✅ **{step}**")
                elif i + 1 == st.session_state.upload_step:
                    st.markdown(f"▶️ **{step}**")
                else:
                    st.markdown(f"⚪ {step}")
        
        st.divider()
    
    def _step1_file_upload(self):
        """Step 1: File upload and basic validation"""
        
        st.header("Step 1: Upload Data File")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV/TSV file",
            type=['csv', 'tsv', 'txt','xlsx'],
            key="main_data_file_uploader",
            help="Upload DIA-NN, Spectronaut, or MaxQuant output"
        )
        
        if uploaded_file is not None:
            st.success("✅ File uploaded successfully!")
            
            # Load data
            with st.spinner("Loading data..."):
                # Detect separator
                first_line = uploaded_file.getvalue().decode('utf-8').split('\n')[0]
                sep = '\t' if '\t' in first_line else ','
                
                # Reset file pointer
                uploaded_file.seek(0)
                
                df = pd.read_csv(uploaded_file, sep=sep, low_memory=False)
                st.session_state.raw_data = df
                st.session_state.data_validated = True
            
            st.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    
    def _step2_data_preview(self):
        """Step 2: Data preview"""
        
        st.header("Step 2: Data Preview")
        
        if 'raw_data' not in st.session_state:
            st.error("No data loaded. Please go back to Step 1.")
            return
        
        df = st.session_state.raw_data
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", f"{len(df.columns):,}")

        
        st.dataframe(df.head(10), use_container_width=True, height=400)
    
    def _step3_column_mapping(self):
        """Step 3: Column classification and name trimming"""
        
        st.header("Step 3: Column Mapping")
        
        if 'raw_data' not in st.session_state:
            st.error("No data loaded. Please go back to Step 1.")
            return
        
        df = st.session_state.raw_data
        
        # Auto-detect columns
        metadata_cols = [col for col in df.columns if df[col].dtype in ["string"]]
        
        quantity_cols = [col for col in df.columns if col not in metadata_cols 
                        and df[col].dtype in ['float64', 'int64']]
        
        st.markdown("**Metadata Columns**")
        selected_metadata = st.multiselect(
            "Select metadata columns",
            options=df.columns.tolist(),
            default=metadata_cols,
            key="metadata_columns_selector"
        )
        
        st.markdown("**Quantification Columns**")
        selected_quantity = st.multiselect(
            "Select quantity columns",
            options=df.columns.tolist(),
            default=quantity_cols,
            key="quantity_columns_selector"
        )
        
        st.session_state.selected_metadata_cols = selected_metadata
        st.session_state.selected_quantity_cols = selected_quantity
        
        # Column name trimming with preview
        st.divider()
        st.subheader("✂️ Sample Name Trimming")
        
        st.markdown("""
        Trimmed sample names are shown below. You can edit any name if needed.
        """)
        
        # Create trimmed mapping
        trimmed_mapping = {}
        for col in selected_quantity:
            trimmed_mapping[col] = self._trim_column_name(col)
        
        # Show preview table
        preview_df = pd.DataFrame({
            'Original Column Name': list(trimmed_mapping.keys()),
            'Trimmed Name': list(trimmed_mapping.values())
        })
        
        st.dataframe(preview_df, use_container_width=True, height=300)
        
        # Allow editing
        enable_edit = st.checkbox("Enable editing of trimmed names", key="enable_trim_edit")
        
        if enable_edit:
            st.markdown("**Edit individual names:**")
            final_mapping = {}
            
            for i, (orig, trimmed) in enumerate(trimmed_mapping.items()):
                edited = st.text_input(
                    f"{orig}",
                    value=trimmed,
                    key=f"trim_edit_{i}"
                )
                final_mapping[orig] = edited
            
            st.session_state.column_name_mapping = final_mapping
        else:
            st.session_state.column_name_mapping = trimmed_mapping
    
    def _step4_sample_annotation(self):
        """Step 4: Species annotation with EDITABLE DEFAULTS"""
        
        st.header("Step 4: Sample Annotation")
        
        if 'raw_data' not in st.session_state:
            st.error("No data loaded. Please go back to Step 1.")
            return
        
        df = st.session_state.raw_data
        
        st.subheader("🧬 Species Annotation")
        
        st.markdown("""
        Define keywords to identify species in your protein identifiers.
        Default keywords are provided but can be edited.
        """)
        
        if 'species_keywords' not in st.session_state:
            st.session_state.species_keywords = [
                {'keyword': 'HUMAN', 'species': 'Human'},
                {'keyword': 'YEAST', 'species': 'Yeast'},
                {'keyword': 'ECOLI', 'species': 'E. coli'}
            ]
        
        num_species = st.number_input(
            "Number of species", 
            min_value=1, 
            max_value=10, 
            value=len(st.session_state.species_keywords),
            key="num_species_input"
        )
        
        current_count = len(st.session_state.species_keywords)
        if num_species > current_count:
            for i in range(num_species - current_count):
                st.session_state.species_keywords.append({'keyword': '', 'species': ''})
        elif num_species < current_count:
            st.session_state.species_keywords = st.session_state.species_keywords[:num_species]
        
        mapping = {}
        st.markdown("**Species Keywords** (edit as needed)")
        
        for i in range(num_species):
            col1, col2 = st.columns(2)
            
            current_kw = st.session_state.species_keywords[i]['keyword']
            current_sp = st.session_state.species_keywords[i]['species']
            
            with col1:
                keyword = st.text_input(
                    f"Keyword {i+1}", 
                    value=current_kw,
                    key=f"species_kw_{i}", 
                    placeholder="e.g., HUMAN"
                )
            with col2:
                species = st.text_input(
                    f"Species {i+1}", 
                    value=current_sp,
                    key=f"species_sp_{i}", 
                    placeholder="e.g., Human"
                )
            
            st.session_state.species_keywords[i] = {'keyword': keyword, 'species': species}
            
            if keyword and species:
                mapping[keyword] = species
        
        if mapping:
            text_cols = [col for col in df.columns if df[col].dtype == 'object']
            
            if text_cols:
                selected_col = st.selectbox(
                    "Column with species identifiers", 
                    options=text_cols,
                    key="species_column_selector"
                )
                
                def assign_species(val):
                    if pd.isna(val):
                        return "Unknown"
                    val_str = str(val).upper()
                    for kw, sp in mapping.items():
                        if kw.upper() in val_str:
                            return sp
                    return "Unknown"
                
                species_series = df[selected_col].apply(assign_species)
                st.session_state.species_assignments = species_series
                
                st.markdown("**Species Distribution**")
                
                species_counts = species_series.value_counts()
                chart_data = pd.DataFrame({
                    'Species': species_counts.index.tolist(),
                    'Count': species_counts.values.tolist()
                })
                
                st.bar_chart(chart_data.set_index('Species'))
                
                cols = st.columns(len(species_counts))
                for idx, (species, count) in enumerate(species_counts.items()):
                    with cols[idx]:
                        st.metric(species, f"{count:,}")
                
                detected_species = [s for s in species_series.unique() if s != 'Unknown']
                st.success(f"✅ Detected {len(detected_species)} species: {', '.join(detected_species)}")
                
                unknown_count = (species_series == 'Unknown').sum()
                if unknown_count > 0:
                    unknown_pct = (unknown_count / len(species_series)) * 100
                    st.warning(f"⚠️ {unknown_count:,} proteins ({unknown_pct:.1f}%) could not be assigned to any species")
            else:
                st.warning("No text columns found for species assignment")
        else:
            st.warning("Please enter at least one species keyword")


    
    def _step5_workflow_suggestion(self):
        """Step 5: Workflow selection"""
        
        st.header("Step 5: Workflow Selection")
        
        if 'raw_data' not in st.session_state:
            st.error("No data loaded.")
            return
        
        df = st.session_state.raw_data
        species_series = st.session_state.get('species_assignments', pd.Series())
        quantity_cols = st.session_state.get('selected_quantity_cols', [])
        
        # Show summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Proteins", len(df))
        with col2:
            st.metric("Samples", len(quantity_cols))
        with col3:
            n_species = species_series.nunique() if len(species_series) > 0 else 1
            st.metric("Species", n_species)

        
        # Workflow selection
        workflow = st.selectbox(
            "Select workflow",
            ["LFQbench", "Standard DIA"],
            key="workflow_choice_selector"
        )
        
        st.session_state.workflow_choice = workflow
        st.session_state.upload_complete = True
        
        st.success("✅ Data upload complete! Ready for analysis.")
    
    def _render_navigation(self):
        """Render navigation buttons"""
        
        st.divider()
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.session_state.upload_step > 1:
                if st.button("⬅️ Previous", use_container_width=True, key="nav_prev_btn"):
                    st.session_state.upload_step -= 1
                    st.rerun()
        
        with col2:
            if st.button("🔄 Reset", use_container_width=True, key="nav_reset_btn"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col3:
            can_proceed = self._can_proceed_to_next_step()
            
            if st.session_state.upload_step < 5:
                if st.button("Next ➡️", use_container_width=True, disabled=not can_proceed, key="nav_next_btn"):
                    st.session_state.upload_step += 1
                    st.rerun()
    
    def _can_proceed_to_next_step(self) -> bool:
        """Check if can proceed"""
        step = st.session_state.upload_step
        
        if step == 1:
            return st.session_state.get('data_validated', False)
        elif step == 2:
            return 'raw_data' in st.session_state
        elif step == 3:
            return 'selected_quantity_cols' in st.session_state and 'column_name_mapping' in st.session_state
        elif step == 4:
            return 'species_assignments' in st.session_state
        else:
            return True


def run_upload_module():
    """Run the upload module"""
    module = DataUploadModule()
    module.run()
