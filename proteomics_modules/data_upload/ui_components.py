"""
Streamlit UI components for data upload module.
Contains reusable widgets and interface elements.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from .config import get_config
from .validators import ValidationResult


class FileUploadUI:
    """UI components for file upload"""
    
    def __init__(self):
        self.config = get_config()
    
    def render_file_uploader(self, key: str = "data_file") -> Optional[object]:
        """
        Render file upload widget
        
        Args:
            key: Unique key for widget
            
        Returns:
            Uploaded file object or None
        """
        st.subheader("📁 Upload Proteomics Data")
        
        st.markdown(f"""
        **Supported formats:** {', '.join(self.config.ALLOWED_EXTENSIONS)}
        
        **Maximum file size:** {self.config.MAX_FILE_SIZE_MB} MB
        
        **Expected structure:**
        - First columns: Protein identifiers and metadata
        - Remaining columns: Quantification values per sample
        """)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV/TSV file",
            type=[ext.replace('.', '') for ext in self.config.ALLOWED_EXTENSIONS],
            key=key,
            help="Upload DIA-NN, Spectronaut, or MaxQuant output"
        )
        
        return uploaded_file
    
    def render_metadata_uploader(self, key: str = "metadata_file") -> Optional[object]:
        """Render metadata file upload widget"""
        
        with st.expander("📋 Optional: Upload Metadata File", expanded=False):
            st.markdown("""
            Upload a CSV file with sample annotations:
            
            **Required column:**
            - `sample_name`: Sample identifiers matching data columns
            
            **Optional columns:**
            - `condition`: Experimental condition (e.g., Control, Treatment)
            - `replicate`: Replicate number
            - `species`: Species name
            - `batch`: Batch identifier
            - `injection_order`: MS injection order
            """)
            
            metadata_file = st.file_uploader(
                "Choose metadata CSV file",
                type=['csv', 'txt', 'tsv'],
                key=key,
                help="Sample annotation file"
            )
            
            return metadata_file
    
    def display_validation_results(self, results: List[ValidationResult]):
        """Display validation results with appropriate styling"""
        
        if not results:
            return
        
        errors = [r for r in results if r.severity == 'error' and not r.is_valid]
        warnings = [r for r in results if r.severity == 'warning']
        infos = [r for r in results if r.severity == 'info' and r.is_valid]
        
        # Display errors
        if errors:
            with st.expander("❌ Errors", expanded=True):
                for result in errors:
                    st.error(result.message)
        
        # Display warnings
        if warnings:
            with st.expander("⚠️ Warnings", expanded=False):
                for result in warnings:
                    st.warning(result.message)
        
        # Display info
        if infos:
            with st.expander("ℹ️ File Information", expanded=False):
                for result in infos:
                    st.info(result.message)


class DataPreviewUI:
    """UI components for data preview"""
    
    @staticmethod
    def render_dataframe_preview(df: pd.DataFrame, 
                                n_rows: int = 10,
                                title: str = "Data Preview"):
        """Render interactive dataframe preview"""
        
        st.subheader(title)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", f"{len(df.columns):,}")
        with col3:
            missing_pct = (df.isna().sum().sum() / df.size) * 100
            st.metric("Missing Values", f"{missing_pct:.1f}%")
        
        # Display first rows
        st.dataframe(
            df.head(n_rows),
            use_container_width=True,
            height=400
        )
    
    @staticmethod
    def render_column_statistics(df: pd.DataFrame):
        """Display column-level statistics"""
        
        st.subheader("Column Statistics")
        
        stats_data = []
        for col in df.columns:
            col_data = df[col]
            
            stats = {
                'Column': col,
                'Type': str(col_data.dtype),
                'Non-Null': col_data.notna().sum(),
                'Null': col_data.isna().sum(),
                'Null %': f"{(col_data.isna().sum() / len(col_data)) * 100:.1f}%"
            }
            
            if pd.api.types.is_numeric_dtype(col_data):
                stats.update({
                    'Min': f"{col_data.min():.2e}" if col_data.notna().any() else "N/A",
                    'Max': f"{col_data.max():.2e}" if col_data.notna().any() else "N/A",
                    'Mean': f"{col_data.mean():.2e}" if col_data.notna().any() else "N/A"
                })
            
            stats_data.append(stats)
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, height=400)
    
    @staticmethod
    def render_missing_value_heatmap(df: pd.DataFrame, sample_cols: List[str]):
        """Render heatmap of missing values"""
        
        st.subheader("Missing Value Pattern")
        
        if not sample_cols:
            st.info("No quantification columns detected")
            return
        
        # Calculate missing values per sample
        missing_df = df[sample_cols].isna().astype(int)
        missing_pct = (missing_df.sum() / len(df)) * 100
        
        # Create bar chart
        fig = px.bar(
            x=sample_cols,
            y=missing_pct.values,
            labels={'x': 'Sample', 'y': 'Missing (%)'},
            title='Missing Values by Sample'
        )
        fig.update_xaxes(tickangle=45)
        fig.update_layout(height=400, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)


class ColumnMappingUI:
    """UI components for column mapping and classification"""
    
    @staticmethod
    def render_column_classifier(columns: List[str],
                                metadata_cols: List[str],
                                quantity_cols: List[str]) -> Dict[str, List[str]]:
        """
        Interactive column classification interface
        
        Returns:
            Dict with 'metadata' and 'quantity' lists
        """
        st.subheader("🔍 Column Classification")
        
        st.markdown("""
        Review the automatic classification. You can modify if needed.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Metadata Columns**")
            st.caption(f"Detected: {len(metadata_cols)}")
            
            selected_metadata = st.multiselect(
                "Metadata columns",
                options=columns,
                default=metadata_cols,
                key="metadata_columns",
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown("**Quantification Columns**")
            st.caption(f"Detected: {len(quantity_cols)}")
            
            # Default to quantity cols, but allow user to modify
            selected_quantity = st.multiselect(
                "Quantity columns",
                options=columns,
                default=quantity_cols,
                key="quantity_columns",
                label_visibility="collapsed"
            )
        
        return {
            'metadata': selected_metadata,
            'quantity': selected_quantity
        }
    
    @staticmethod
    def render_quantity_column_preview(original_names: List[str],
                                      trimmed_names: List[str]) -> Dict[str, str]:
        """
        Show preview of column name trimming with edit capability
        
        Returns:
            Dict mapping original to trimmed (possibly edited) names
        """
        st.subheader("✂️ Sample Name Trimming")
        
        st.markdown("""
        Trimmed sample names are shown below. Click to edit any name.
        """)
        
        mapping = {}
        
        # Create editable table
        df_preview = pd.DataFrame({
            'Original': original_names,
            'Trimmed': trimmed_names
        })
        
        st.dataframe(df_preview, use_container_width=True, height=300)
        
        # Allow bulk edit or individual edits
        edit_mode = st.checkbox("Enable editing", key="enable_name_edit")
        
        if edit_mode:
            st.markdown("**Edit individual names:**")
            
            for i, (orig, trim) in enumerate(zip(original_names, trimmed_names)):
                edited_name = st.text_input(
                    f"{orig}",
                    value=trim,
                    key=f"trim_edit_{i}"
                )
                mapping[orig] = edited_name
        else:
            # Use trimmed names as-is
            mapping = dict(zip(original_names, trimmed_names))
        
        return mapping


class SampleAnnotationUI:
    """UI components for sample annotation and grouping"""
    
    @staticmethod
    def render_replicate_grouping(sample_names: List[str],
                                  suggested_groups: Dict[str, List[str]]) -> Dict[str, Dict]:
        """
        Interface for annotating replicates and conditions
        
        Returns:
            Dict mapping sample name to annotations
        """
        st.subheader("🏷️ Sample Annotation")
        
        st.markdown("""
        Assign samples to conditions and replicates.
        """)
        
        # Show suggested grouping
        st.markdown("**Suggested Replicate Groups:**")
        for group_name, members in suggested_groups.items():
            with st.expander(f"📊 {group_name} ({len(members)} samples)", expanded=False):
                st.write(members)
        
        # Allow manual annotation
        annotations = {}
        
        st.markdown("**Manual Annotation:**")
        
        # Create form for batch annotation
        with st.form("sample_annotation_form"):
            for sample in sample_names:
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.text(sample)
                
                with col2:
                    condition = st.text_input(
                        "Condition",
                        key=f"cond_{sample}",
                        label_visibility="collapsed",
                        placeholder="e.g., Control"
                    )
                
                with col3:
                    replicate = st.text_input(
                        "Replicate",
                        key=f"rep_{sample}",
                        label_visibility="collapsed",
                        placeholder="e.g., 1"
                    )
                
                annotations[sample] = {
                    'condition': condition,
                    'replicate': replicate
                }
            
            submitted = st.form_submit_button("Apply Annotations")
            
            if submitted:
                st.success("Annotations applied!")
        
        return annotations
    
    @staticmethod
    def render_species_assignment(df: pd.DataFrame,
                                 species_series: pd.Series) -> Tuple[pd.Series, Dict[str, str]]:
        """
        Interface for species assignment
        
        Returns:
            Tuple of (species_series, custom_patterns)
        """
        st.subheader("🧬 Species Detection")
        
        # Show distribution
        species_counts = species_series.value_counts()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Detected Species:**")
            for species, count in species_counts.items():
                pct = (count / len(species_series)) * 100
                st.metric(species, f"{count} ({pct:.1f}%)")
        
        with col2:
            # Pie chart
            fig = px.pie(
                values=species_counts.values,
                names=species_counts.index,
                title='Species Distribution'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Custom species patterns
        with st.expander("➕ Add Custom Species Pattern", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                custom_species = st.text_input(
                    "Species Name",
                    key="custom_species_name",
                    placeholder="e.g., Mouse"
                )
            
            with col2:
                custom_pattern = st.text_input(
                    "Regex Pattern",
                    key="custom_species_pattern",
                    placeholder="e.g., MOUSE|Mus musculus"
                )
            
            if st.button("Add Pattern"):
                if custom_species and custom_pattern:
                    st.success(f"Added pattern for {custom_species}")
                    return species_series, {custom_species: custom_pattern}
        
        return species_series, {}



class WorkflowSuggestionUI:
    """UI components for workflow recommendation"""
    
    @staticmethod
    def render_workflow_suggestion(data_characteristics: Dict) -> str:
        """
        Suggest analysis workflow based on data characteristics
        
        Args:
            data_characteristics: Dict with data properties
            
        Returns:
            Selected workflow name
        """
        st.subheader("🔬 Workflow Recommendation")
        
        # Extract characteristics
        n_proteins = data_characteristics.get('n_proteins', 0)
        n_samples = data_characteristics.get('n_samples', 0)
        species_count = data_characteristics.get('n_species', 0)
        has_replicates = data_characteristics.get('has_replicates', False)
        missing_pct = data_characteristics.get('missing_percent', 0)
        
        # Determine recommendations
        workflows = []
        
        # LFQbench workflow (for benchmark data)
        if species_count > 1:
            workflows.append({
                'name': 'LFQbench',
                'description': 'Benchmark evaluation with multi-species samples',
                'suitability': 'High',
                'reasons': [
                    f'✓ Multiple species detected ({species_count})',
                    '✓ Suitable for method comparison',
                    '✓ Provides ground-truth metrics'
                ]
            })
        
        # Standard DIA workflow
        workflows.append({
            'name': 'Standard DIA',
            'description': 'Standard proteomics analysis with statistical testing',
            'suitability': 'High' if has_replicates else 'Medium',
            'reasons': [
                f'✓ {n_proteins} proteins identified',
                f'✓ {n_samples} samples available',
                f'{"✓" if has_replicates else "⚠"} {"Replicates detected" if has_replicates else "Limited replicates"}',
                f'{"✓" if missing_pct < 30 else "⚠"} {missing_pct:.1f}% missing values'
            ]
        })
        
        # Display recommendations
        for wf in workflows:
            color = 'green' if wf['suitability'] == 'High' else 'orange'
            
            with st.container():
                st.markdown(f"""
                <div style='padding: 15px; border-left: 4px solid {color}; background-color: #f0f2f6; margin-bottom: 15px;'>
                    <h4>{wf['name']}</h4>
                    <p><b>Suitability:</b> {wf['suitability']}</p>
                    <p>{wf['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                for reason in wf['reasons']:
                    st.markdown(f"  {reason}")
        
        # Let user select
        selected_workflow = st.selectbox(
            "Select workflow to proceed",
            options=[wf['name'] for wf in workflows],
            key="selected_workflow"
        )
        
        return selected_workflow


class DataSummaryUI:
    """UI components for data summary"""
    
    @staticmethod
    def render_upload_summary(summary_data: Dict):
        """Display comprehensive summary of uploaded data"""
        
        st.subheader("📊 Data Upload Summary")
        
        tab1, tab2, tab3 = st.tabs(["Overview", "Columns", "Quality"])
        
        with tab1:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Proteins",
                    f"{summary_data.get('n_proteins', 0):,}"
                )
            
            with col2:
                st.metric(
                    "Samples",
                    f"{summary_data.get('n_samples', 0)}"
                )
            
            with col3:
                st.metric(
                    "Species",
                    f"{summary_data.get('n_species', 0)}"
                )
            
            with col4:
                st.metric(
                    "Missing",
                    f"{summary_data.get('missing_percent', 0):.1f}%"
                )
        
        with tab2:
            st.markdown("**Column Classification:**")
            
            col_info = summary_data.get('column_info', {})
            
            st.json({
                'Metadata Columns': col_info.get('metadata', []),
                'Quantification Columns': col_info.get('quantity', [])[:10]  # Show first 10
            })
        
        with tab3:
            st.markdown("**Data Quality Checks:**")
            
            quality_checks = summary_data.get('quality_checks', [])
            
            for check in quality_checks:
                if check.get('passed', False):
                    st.success(f"✓ {check.get('message', '')}")
                else:
                    st.error(f"✗ {check.get('message', '')}")
    
    @staticmethod
    def render_next_steps(workflow: str):
        """Display next steps"""
        
        st.subheader("🚀 Next Steps")
        
        if workflow == 'LFQbench':
            st.markdown("""
            **Proceed to LFQbench Analysis:**
            
            1. **Data Filtering**: Remove proteins with excessive missing values
            2. **Species Assignment**: Confirm species classifications
            3. **Normalization**: Apply appropriate normalization method
            4. **Statistical Analysis**: Calculate fold-changes and significance
            5. **Performance Metrics**: Evaluate accuracy, precision, trueness
            
            Click **Continue** to proceed with the analysis.
            """)
        else:
            st.markdown("""
            **Proceed to Standard Analysis:**
            
            1. **Quality Control**: Review data quality metrics
            2. **Preprocessing**: Filter and normalize data
            3. **Statistical Testing**: Differential expression analysis
            4. **Visualization**: Generate plots and reports
            
            Click **Continue** to proceed with the analysis.
            """)


# Export all UI classes
__all__ = [
    'FileUploadUI',
    'DataPreviewUI',
    'ColumnMappingUI',
    'SampleAnnotationUI',
    'WorkflowSuggestionUI',
    'DataSummaryUI'
]

