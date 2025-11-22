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
            type=['csv', 'tsv', 'txt'],
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
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", f"{len(df.columns):,}")
        with col3:
            missing_pct = (df.isna().sum().sum() / df.size) * 100
            st.metric("Missing Values", f"{missing_pct:.1f}%")
        
        st.dataframe(df.head(10), use_container_width=True, height=400)
    
    def _step3_column_mapping(self):
        """Step 3: Column classification and name trimming"""
        
        st.header("Step 3: Column Mapping")
        
        if 'raw_data' not in st.session_state:
            st.error("No data loaded. Please go back to Step 1.")
            return
        
        df = st.session_state.raw_data
        
        # Auto-detect columns
        metadata_cols = [col for col in df.columns if any(x in col for x in 
                        ['Protein', 'Gene', 'Description', 'Q.Value', 'PEP'])]
        
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
        
        # Initialize default species in session state
        if 'species_keywords' not in st.session_state:
            st.session_state.species_keywords = [
                {'keyword': 'HUMAN', 'species': 'Human'},
                {'keyword': 'YEAST', 'species': 'Yeast'},
                {'keyword': 'ECOLI', 'species': 'E. coli'}
            ]
        
        # Number of species selector
        num_species = st.number_input(
            "Number of species", 
            min_value=1, 
            max_value=10, 
            value=len(st.session_state.species_keywords),
            key="num_species_input"
        )
        
        # Adjust list size if changed
        current_count = len(st.session_state.species_keywords)
        if num_species > current_count:
            # Add empty entries
            for i in range(num_species - current_count):
                st.session_state.species_keywords.append({'keyword': '', 'species': ''})
        elif num_species < current_count:
            # Remove extra entries
            st.session_state.species_keywords = st.session_state.species_keywords[:num_species]
        
        # Editable species inputs
        mapping = {}
        
        st.markdown("**Species Keywords** (edit as needed)")
        
        for i in range(num_species):
            col1, col2 = st.columns(2)
            
            # Get current values
            current_kw = st.session_state.species_keywords[i]['keyword']
            current_sp = st.session_state.species_keywords[i]['species']
            
            with col1:
                keyword = st.text_input(
                    f"Keyword {i+1}", 
                    value=current_kw,
                    key=f"species_kw_{i}", 
                    placeholder="e.g., HUMAN"
                )
            with
