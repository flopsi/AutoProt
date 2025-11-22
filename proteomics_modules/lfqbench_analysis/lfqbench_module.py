"""
LFQbench Analysis Streamlit Module
Interactive UI for benchmark proteomics analysis with comprehensive visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from .lfqbench_analysis import get_lfqbench_analyzer, BenchmarkConfig
from .lfqbench_visualizations import get_lfqbench_visualizer


class LFQbenchModule:
    """Streamlit interface for LFQbench analysis"""
    
    def __init__(self):
        self.analyzer = None
        self.visualizer = None
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables"""
        if 'lfqbench_config' not in st.session_state:
            st.session_state.lfqbench_config = None
        
        if 'lfqbench_results' not in st.session_state:
            st.session_state.lfqbench_results = None
        
        if 'lfqbench_step' not in st.session_state:
            st.session_state.lfqbench_step = 1
    
    def run(self):
        """Execute LFQbench analysis workflow"""
        
        st.title("🧪 LFQbench Analysis")
        
        st.markdown("""
        Benchmark proteomics analysis using multi-species samples with known fold-changes.
        Calculate performance metrics including **accuracy**, **precision**, **trueness**, and **sensitivity/specificity**.
        """)
        
        # Check if data from upload module is available
        if 'raw_data' not in st.session_state:
            st.error("❌ No data loaded. Please complete the Data Upload module first.")
            return
        
        # Progress indicator
        self._render_progress()
        
        # Execute current step
        if st.session_state.lfqbench_step == 1:
            self._step1_configuration()
        
        elif st.session_state.lfqbench_step == 2:
            self._step2_run_analysis()
        
        elif st.session_state.lfqbench_step == 3:
            self._step3_visualizations()
        
        # Navigation
        self._render_navigation()
    
    def _render_progress(self):
        """Render progress bar"""
        steps = ["Configuration", "Analysis", "Visualizations"]
        
        progress = st.session_state.lfqbench_step / len(steps)
        st.progress(progress)
        
        cols = st.columns(len(steps))
        for i, (col, step) in enumerate(zip(cols, steps)):
            with col:
                if i + 1 < st.session_state.lfqbench_step:
                    st.markdown(f"✅ **{step}**")
                elif i + 1 == st.session_state.lfqbench_step:
                    st.markdown(f"▶️ **{step}**")
                else:
                    st.markdown(f"⚪ {step}")
        
        st.divider()
    
    def _step1_configuration(self):
        """Step 1: Configure analysis parameters"""
        
        st.header("Step 1: Analysis Configuration")
        
        st.subheader("Expected Fold-Changes")
        
        st.markdown("""
        Define the expected log2 fold-changes for each species in your benchmark samples.
        These are used to calculate accuracy and classify true/false positives.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Species A (e.g., Human)**")
            fc_human = st.number_input(
                "Human Log2 FC",
                value=0.0,
                step=0.5,
                format="%.1f",
                key="fc_human_input",
                help="Expected fold-change for human proteins (typically 0 = no change)"
            )
            
            st.markdown("**Species B (e.g., Yeast)**")
            fc_yeast = st.number_input(
                "Yeast Log2 FC",
                value=1.0,
                step=0.5,
                format="%.1f",
                key="fc_yeast_input",
                help="Expected fold-change for yeast proteins (e.g., 1 = 2x increase)"
            )
        
        with col2:
            st.markdown("**Species C (e.g., E. coli)**")
            fc_ecoli = st.number_input(
                "E. coli Log2 FC",
                value=-2.0,
                step=0.5,
                format="%.1f",
                key="fc_ecoli_input",
                help="Expected fold-change for E. coli proteins (e.g., -2 = 4x decrease)"
            )
            
            st.markdown("**Species D (e.g., C. elegans)**")
            fc_celegans = st.number_input(
                "C. elegans Log2 FC",
                value=-1.0,
                step=0.5,
                format="%.1f",
                key="fc_celegans_input",
                help="Expected fold-change for C. elegans proteins"
            )
        
        st.divider()
        
        st.subheader("Filter Thresholds")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            limit_mv = st.slider(
                "Missing Value Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.67,
                step=0.1,
                key="limit_mv_slider",
                help="Maximum fraction of missing values allowed per protein (2/3 = 67%)"
            )
        
        with col2:
            limit_cv = st.slider(
                "CV Threshold (%)",
                min_value=5.0,
                max_value=50.0,
                value=20.0,
                step=5.0,
                key="limit_cv_slider",
                help="Maximum coefficient of variation (%) for filtering"
            )
        
        with col3:
            limit_fc = st.slider(
                "Fold-Change Threshold",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1,
                key="limit_fc_slider",
                help="Minimum |log2 FC| to classify as differentially abundant"
            )
        
        st.divider()
        
        st.subheader("Statistical Parameters")
        
        alpha_limma = st.selectbox(
            "Adjusted P-value Threshold (α)",
            options=[0.001, 0.01, 0.05, 0.1],
            index=1,
            key="alpha_limma_select",
            help="Significance threshold for differential abundance (typical: 0.01)"
        )
        
        st.divider()
        
        st.subheader("Sample Group Assignment")
        
        st.markdown("""
        Assign your samples to **Control** and **Experimental** groups.
        Use the trimmed sample names from the upload module.
        """)
        
        # Get available sample columns
        quantity_cols = st.session_state.get('selected_quantity_cols', [])
        name_mapping = st.session_state.get('column_name_mapping', {})
        trimmed_names = [name_mapping.get(col, col) for col in quantity_cols]
        
        if not trimmed_names:
            st.error("No quantification columns available. Please complete Data Upload first.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Control Samples**")
            control_samples = st.multiselect(
                "Select control samples",
                options=trimmed_names,
                default=trimmed_names[:len(trimmed_names)//2] if len(trimmed_names) >= 2 else [],
                key="control_samples_select"
            )
        
        with col2:
            st.markdown("**Experimental Samples**")
            experimental_samples = st.multiselect(
                "Select experimental samples",
                options=[s for s in trimmed_names if s not in control_samples],
                default=[s for s in trimmed_names if s not in control_samples][:len(trimmed_names)//2],
                key="experimental_samples_select"
            )
        
        # Validate selection
        if len(control_samples) == 0 or len(experimental_samples) == 0:
            st.warning("⚠️ Please select at least one sample for both Control and Experimental groups.")
            return
        
        # Store configuration (FIXED: use different names than widget keys)
        config = BenchmarkConfig(
            expected_fc_human=fc_human,
            expected_fc_yeast=fc_yeast,
            expected_fc_ecoli=fc_ecoli,
            expected_fc_celegans=fc_celegans,
            limit_mv=limit_mv,
            limit_cv=limit_cv,
            limit_fc=limit_fc,
            alpha_limma=alpha_limma
        )
        
        st.session_state.lfqbench_config = config
        st.session_state.lfq_control_samples = control_samples  # FIXED: different name
        st.session_state.lfq_experimental_samples = experimental_samples  # FIXED: different name
        
        # Show summary
        with st.expander("📋 Configuration Summary", expanded=False):
            st.json({
                "Expected Fold-Changes": {
                    "Human": fc_human,
                    "Yeast": fc_yeast,
                    "E. coli": fc_ecoli,
                    "C. elegans": fc_celegans
                },
                "Filters": {
                    "Missing Value": f"{limit_mv*100:.0f}%",
                    "CV": f"{limit_cv}%",
                    "Fold-Change": limit_fc
                },
                "Statistical": {
                    "Alpha": alpha_limma
                },
                "Samples": {
                    "Control": control_samples,
                    "Experimental": experimental_samples
                }
            })
    
    def _step2_run_analysis(self):
        """Step 2: Run LFQbench analysis"""
        
        st.header("Step 2: Run Analysis")
        
        if st.session_state.lfqbench_config is None:
            st.error("Configuration not set. Please go back to Step 1.")
            return
        
        # Get data
        df = st.session_state.raw_data.copy()
        species_assignments = st.session_state.get('species_assignments', pd.Series())
        
        if len(species_assignments) == 0:
            st.error("Species assignments not found. Please complete Data Upload module.")
            return
        
        # Add species to dataframe
        df['Species'] = species_assignments
        
        # Get sample columns (FIXED: use different session state keys)
        control_samples = st.session_state.lfq_control_samples
        experimental_samples = st.session_state.lfq_experimental_samples
        
        # Map trimmed names back to original column names
        name_mapping = st.session_state.get('column_name_mapping', {})
        reverse_mapping = {v: k for k, v in name_mapping.items()}
        
        control_cols = [reverse_mapping.get(s, s) for s in control_samples]
        experimental_cols = [reverse_mapping.get(s, s) for s in experimental_samples]
        
        # Initialize analyzer
        self.analyzer = get_lfqbench_analyzer(st.session_state.lfqbench_config)
        self.visualizer = get_lfqbench_visualizer()
        
        # Run analysis button
        if st.button("▶️ Run Analysis", type="primary", use_container_width=True, key="run_analysis_btn"):
            
            with st.spinner("Running LFQbench analysis..."):
                
                # Progress container
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                try:
                    # Run analysis
                    progress_text.text("Step 1/8: Filtering by data completeness...")
                    progress_bar.progress(12)
                    
                    results_df, metrics, asymmetry_df = self.analyzer.run_complete_analysis(
                        df, experimental_cols, control_cols
                    )
                    
                    progress_bar.progress(100)
                    progress_text.text("✅ Analysis complete!")
                    
                    # Store results
                    st.session_state.lfqbench_results = {
                        'results_df': results_df,
                        'metrics': metrics,
                        'asymmetry_df': asymmetry_df,
                        'control_cols': control_cols,
                        'experimental_cols': experimental_cols
                    }
                    
                    st.success("✅ Analysis completed successfully!")
                    
                    # Show quick summary
                    st.divider()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Proteins Analyzed", metrics['n_proteins'])
                    
                    with col2:
                        st.metric("Sensitivity (%)", f"{metrics['sensitivity']:.1f}")
                    
                    with col3:
                        st.metric("Specificity (%)", f"{metrics['specificity']:.1f}")
                    
                    with col4:
                        st.metric("Empirical FDR (%)", f"{metrics['de_fdr']:.1f}")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Show existing results if available
        elif st.session_state.lfqbench_results is not None:
            st.info("✅ Analysis results available. Proceed to Step 3 for visualizations.")
            
            metrics = st.session_state.lfqbench_results['metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Proteins Analyzed", metrics['n_proteins'])
            
            with col2:
                st.metric("Sensitivity (%)", f"{metrics['sensitivity']:.1f}")
            
            with col3:
                st.metric("Specificity (%)", f"{metrics['specificity']:.1f}")
            
            with col4:
                st.metric("Empirical FDR (%)", f"{metrics['de_fdr']:.1f}")
    
    def _step3_visualizations(self):
        """Step 3: Display comprehensive visualizations"""
        
        st.header("Step 3: Results & Visualizations")
        
        if st.session_state.lfqbench_results is None:
            st.error("No analysis results available. Please run analysis in Step 2.")
            return
        
        # Get results
        results = st.session_state.lfqbench_results
        results_df = results['results_df']
        metrics = results['metrics']
        asymmetry_df = results['asymmetry_df']
        
        # Initialize visualizer if not done
        if self.visualizer is
