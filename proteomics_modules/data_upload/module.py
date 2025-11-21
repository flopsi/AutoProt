"""
Main data upload module orchestrator.
Coordinates the complete upload workflow from file selection to data export.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from .config import get_config
from .session_manager import get_session_manager
from .validators import get_validator
from .parsers import get_data_parser, get_metadata_parser
from .column_detector import get_column_detector, get_species_manager
from .ui_components import (
    FileUploadUI,
    DataPreviewUI,
    ColumnMappingUI,
    SampleAnnotationUI,
    WorkflowSuggestionUI,
    DataSummaryUI
)


class DataUploadModule:
    """
    Main orchestrator for data upload workflow
    
    Workflow Steps:
    1. File upload and validation
    2. Data preview and column detection
    3. Optional metadata upload
    4. Sample name trimming and annotation
    5. Species detection
    6. Workflow suggestion
    7. Data export for next module
    """
    
    def __init__(self):
        self.config = get_config()
        self.session_manager = get_session_manager()
        self.validator = get_validator()
        self.data_parser = get_data_parser()
        self.metadata_parser = get_metadata_parser()
        self.column_detector = get_column_detector()
        self.species_manager = get_species_manager()
        
        # Initialize session state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables"""
        if 'upload_step' not in st.session_state:
            st.session_state.upload_step = 1
        
        if 'data_validated' not in st.session_state:
            st.session_state.data_validated = False
        
        if 'upload_complete' not in st.session_state:
            st.session_state.upload_complete = False
    
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
        steps = [
            "Upload",
            "Preview",
            "Column Mapping",
            "Annotation",
            "Workflow"
        ]
        
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
        
        ui = FileUploadUI()
        
        # File upload
        uploaded_file = ui.render_file_uploader()
        
        # Metadata upload (optional)
        metadata_file = ui.render_metadata_uploader()
        
        if uploaded_file is not None:
            # Validate file
            validation_results = []
            
            # Extension validation
            ext_result = self.validator.validate_file_extension(uploaded_file.name)
            validation_results.append(ext_result)
            
            # Size validation
            size_result = self.validator.validate_file_size(uploaded_file.size)
            validation_results.append(size_result)
            
            # Display validation
            ui.display_validation_results(validation_results)
            
            # Check if valid
            if all(r.is_valid for r in validation_results if r.severity == 'error'):
                # Save file
                file_path = self.session_manager.save_uploaded_file(uploaded_file)
                st.session_state.data_file_path = str(file_path)
                
                # Structure validation
                struct_result = self.validator.validate_csv_structure(file_path)
                
                if struct_result.is_valid:
                    st.success("✅ File uploaded successfully!")
                    
                    # Load data
                    with st.spinner("Loading data..."):
                        df = self.data_parser.load_dataframe(file_path)
                        st.session_state.raw_data = df
                        st.session_state.data_validated = True
                    
                    # Save metadata file if provided
                    if metadata_file is not None:
                        meta_path = self.session_manager.save_uploaded_file(
                            metadata_file,
                            filename="metadata.csv"
                        )
                        st.session_state.metadata_file_path = str(meta_path)
                        st.success("✅ Metadata file uploaded!")
                else:
                    st.error(f"❌ {struct_result.message}")
    
    def _step2_data_preview(self):
        """Step 2: Data preview and initial quality checks"""
        
        st.header("Step 2: Data Preview")
        
        if 'raw_data' not in st.session_state:
            st.error("No data loaded. Please go back to Step 1.")
            return
        
        df = st.session_state.raw_data
        
        # Preview UI
        preview_ui = DataPreviewUI()
        preview_ui.render_dataframe_preview(df)
        
        # Column statistics
        with st.expander("📊 Column Statistics", expanded=False):
            preview_ui.render_column_statistics(df)
        
        # Validate proteomics data
        validation_results = self.validator.validate_proteomics_data(df)
        
        st.subheader("Data Quality Checks")
        FileUploadUI().display_validation_results(validation_results)
        
        # Store validation results
        st.session_state.validation_results = validation_results
    
    def _step3_column_mapping(self):
        """Step 3: Column classification and name trimming"""
        
        st.header("Step 3: Column Mapping")
        
        if 'raw_data' not in st.session_state:
            st.error("No data loaded. Please go back to Step 1.")
            return
        
        df = st.session_state.raw_data
        
        # Auto-detect columns
        if 'column_classification' not in st.session_state:
            classification = self.column_detector.classify_columns(df.columns.tolist())
            st.session_state.column_classification = classification
        else:
            classification = st.session_state.column_classification
        
        # Render classification UI
        mapping_ui = ColumnMappingUI()
        
        selected_cols = mapping_ui.render_column_classifier(
            df.columns.tolist(),
            classification.metadata_columns,
            classification.quantity_columns
        )
        
        # Update classification
        st.session_state.selected_metadata_cols = selected_cols['metadata']
        st.session_state.selected_quantity_cols = selected_cols['quantity']
        
        # Column name trimming
        st.divider()
        
        quantity_mapping = classification.quantity_column_mapping
        
        trimmed_names = [quantity_mapping.get(col, col) 
                        for col in selected_cols['quantity']]
        
        name_mapping = mapping_ui.render_quantity_column_preview(
            selected_cols['quantity'],
            trimmed_names
        )
        
        st.session_state.column_name_mapping = name_mapping
        
        # Preview missing values
        preview_ui = DataPreviewUI()
        preview_ui.render_missing_value_heatmap(df, selected_cols['quantity'])
    
    def _step4_sample_annotation(self):
        """Step 4: Sample annotation and species detection"""
        
        st.header("Step 4: Sample Annotation")
        
        if 'raw_data' not in st.session_state:
            st.error("No data loaded. Please go back to Step 1.")
            return
        
        df = st.session_state.raw_data
        quantity_cols = st.session_state.get('selected_quantity_cols', [])
        name_mapping = st.session_state.get('column_name_mapping', {})
        
        # Get trimmed names
        trimmed_names = [name_mapping.get(col, col) for col in quantity_cols]
        
        # Suggest replicate grouping
        suggested_groups = self.column_detector.suggest_replicate_groups(trimmed_names)
        st.session_state.suggested_groups = suggested_groups
        
        # Sample annotation UI
        annotation_ui = SampleAnnotationUI()
        
        # Replicate grouping
        annotations = annotation_ui.render_replicate_grouping(
            trimmed_names,
            suggested_groups
        )
        st.session_state.sample_annotations = annotations
        
        # Species detection
        st.divider()
        
        with st.spinner("Detecting species..."):
            species_series = self.column_detector.detect_species(df)
            st.session_state.species_assignments = species_series
        
        species_series, custom_patterns = annotation_ui.render_species_assignment(
            df,
            species_series
        )
        
        if custom_patterns:
            # Add custom patterns and re-detect
            for species, pattern in custom_patterns.items():
                self.species_manager.add_custom_species(species, pattern)
            
            species_series = self.species_manager.detect_with_custom_patterns(df)
            st.session_state.species_assignments = species_series
    
    def _step5_workflow_suggestion(self):
        """Step 5: Workflow suggestion and summary"""
        
        st.header("Step 5: Workflow Selection")
        
        if 'raw_data' not in st.session_state:
            st.error("No data loaded. Please go back to Step 1.")
            return
        
        df = st.session_state.raw_data
        species_series = st.session_state.get('species_assignments', pd.Series())
        quantity_cols = st.session_state.get('selected_quantity_cols', [])
        
        # Calculate data characteristics
        data_chars = {
            'n_proteins': len(df),
            'n_samples': len(quantity_cols),
            'n_species': species_series.nunique() if len(species_series) > 0 else 1,
            'has_replicates': len(quantity_cols) >= 3,
            'missing_percent': (df[quantity_cols].isna().sum().sum() / df[quantity_cols].size) * 100 if quantity_cols else 0
        }
        
        # Workflow suggestion
        workflow_ui = WorkflowSuggestionUI()
        selected_workflow = workflow_ui.render_workflow_suggestion(data_chars)
        
        st.session_state.selected_workflow = selected_workflow
        
        # Data summary
        st.divider()
        
        summary_data = {
            'n_proteins': data_chars['n_proteins'],
            'n_samples': data_chars['n_samples'],
            'n_species': data_chars['n_species'],
            'missing_percent': data_chars['missing_percent'],
            'column_info': {
                'metadata': st.session_state.get('selected_metadata_cols', []),
                'quantity': st.session_state.get('selected_quantity_cols', [])
            },
            'quality_checks': [
                {'passed': True, 'message': f"{data_chars['n_proteins']} proteins identified"},
                {'passed': True, 'message': f"{data_chars['n_samples']} samples detected"},
                {'passed': data_chars['missing_percent'] < 50, 
                 'message': f"{data_chars['missing_percent']:.1f}% missing values"}
            ]
        }
        
        summary_ui = DataSummaryUI()
        summary_ui.render_upload_summary(summary_data)
        
        # Next steps
        st.divider()
        summary_ui.render_next_steps(selected_workflow)
        
        # Mark upload complete
        st.session_state.upload_complete = True
    
    def _render_navigation(self):
        """Render navigation buttons"""
        
        st.divider()
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.session_state.upload_step > 1:
                if st.button("⬅️ Previous", use_container_width=True):
                    st.session_state.upload_step -= 1
                    st.rerun()
        
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                # Clear session
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col3:
            can_proceed = self._can_proceed_to_next_step()
            
            if st.session_state.upload_step < 5:
                if st.button("Next ➡️", use_container_width=True, disabled=not can_proceed):
                    st.session_state.upload_step += 1
                    st.rerun()
            elif st.session_state.upload_complete:
                if st.button("Continue to Analysis ➡️", use_container_width=True):
                    st.success("✅ Proceeding to analysis module...")
                    # This will be integrated with next module
    
    def _can_proceed_to_next_step(self) -> bool:
        """Check if can proceed to next step"""
        
        step = st.session_state.upload_step
        
        if step == 1:
            return st.session_state.get('data_validated', False)
        elif step == 2:
            return 'raw_data' in st.session_state
        elif step == 3:
            return 'selected_quantity_cols' in st.session_state
        elif step == 4:
            return 'species_assignments' in st.session_state
        else:
            return True
    
    def export_data_for_analysis(self) -> Dict[str, Any]:
        """
        Export prepared data for next analysis module
        
        Returns:
            Dictionary with all necessary data and metadata
        """
        if not st.session_state.get('upload_complete', False):
            raise ValueError("Upload workflow not complete")
        
        df = st.session_state.raw_data
        metadata_cols = st.session_state.selected_metadata_cols
        quantity_cols = st.session_state.selected_quantity_cols
        name_mapping = st.session_state.column_name_mapping
        
        # Prepare data matrices
        metadata_df, quantity_df = self.data_parser.prepare_analysis_data(
            df, metadata_cols, quantity_cols
        )
        
        # Rename quantity columns
        quantity_df.columns = [name_mapping.get(col, col) for col in quantity_df.columns]
        
        # Add species assignments
        metadata_df['Species'] = st.session_state.species_assignments
        
        # Export package
        export_data = {
            'metadata': metadata_df,
            'quantification': quantity_df,
            'sample_annotations': st.session_state.sample_annotations,
            'workflow': st.session_state.selected_workflow,
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_manager.get_or_create_session_id()
        }
        
        return export_data


def run_upload_module():
    """Convenience function to run the upload module"""
    module = DataUploadModule()
    module.run()
