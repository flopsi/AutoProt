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
        
        if 'lfqbench_color_scheme' not in st.session_state:
            st.session_state.lfqbench_color_scheme = 'Default'
        
        if 'lfqbench_auto_run' not in st.session_state:
            st.session_state.lfqbench_auto_run = False
    
    def run(self):
        """Execute LFQbench analysis workflow"""
        
        st.title("🧪 LFQbench Analysis")
        
        st.markdown("""
        Benchmark proteomics analysis using multi-species samples with known fold-changes.
        Calculate performance metrics including **accuracy**, **precision**, **trueness**, and **sensitivity/specificity**.
        """)
        
        if 'raw_data' not in st.session_state:
            st.error("❌ No data loaded. Please complete the Data Upload module first.")
            return
        
        self._render_progress()
        
        if st.session_state.lfqbench_step == 1:
            self._step1_configuration()
        elif st.session_state.lfqbench_step == 2:
            self._step2_run_analysis()
        elif st.session_state.lfqbench_step == 3:
            self._step3_visualizations()
        
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
        st.subheader("🎨 Color Scheme")
        
        color_schemes = {
            'Default': {'Human': '#199d76', 'Yeast': '#d85f02', 'E. coli': '#7570b2', 'C.elegans': 'darkred'},
            'Viridis': {'Human': '#440154', 'Yeast': '#31688e', 'E. coli': '#35b779', 'C.elegans': '#fde724'},
            'Pastel': {'Human': '#ffadad', 'Yeast': '#ffd6a5', 'E. coli': '#caffbf', 'C.elegans': '#a0c4ff'},
            'Bold': {'Human': '#e63946', 'Yeast': '#f77f00', 'E. coli': '#06d6a0', 'C.elegans': '#118ab2'},
            'Earth': {'Human': '#8b4513', 'Yeast': '#d2691e', 'E. coli': '#556b2f', 'C.elegans': '#2f4f4f'}
        }
        
        selected_scheme = st.selectbox("Choose color palette for visualizations", options=list(color_schemes.keys()), index=0, key="color_scheme_select")
        st.session_state.lfqbench_color_scheme = selected_scheme
        st.divider()
        
        st.subheader("Expected Fold-Changes")
        st.markdown("Define the expected log2 fold-changes for each species in your benchmark samples.")
        
        species_assignments = st.session_state.get('species_assignments', pd.Series())
        detected_species = species_assignments.unique().tolist() if len(species_assignments) > 0 else []
        detected_species = [s for s in detected_species if s != 'Unknown']
        
        if len(detected_species) == 0:
            st.error("❌ No species detected. Please complete Data Upload module and assign species.")
            return
        
        st.info(f"**Detected species**: {', '.join(detected_species)}")
        
        species_fold_changes = {}
        n_species = len(detected_species)
        cols = st.columns(min(2, n_species))
        
        default_fc_map = {'Human': 0.0, 'Yeast': 1.0, 'E. coli': -2.0, 'C.elegans': -1.0}
        
        for i, species in enumerate(detected_species):
            col_idx = i % 2
            with cols[col_idx]:
                st.markdown(f"**{species}**")
                fc = st.number_input(f"{species} Log2 FC", value=default_fc_map.get(species, 0.0), step=0.5, format="%.1f", key=f"fc_{species}_input")
                species_fold_changes[species] = fc
        
        st.divider()
        st.subheader("Filter Thresholds")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            limit_mv = st.slider("Missing Value Threshold", min_value=0.0, max_value=1.0, value=0.67, step=0.1, key="limit_mv_slider")
        with col2:
            limit_cv = st.slider("CV Threshold (%)", min_value=5.0, max_value=50.0, value=20.0, step=5.0, key="limit_cv_slider")
        with col3:
            limit_fc = st.slider("Fold-Change Threshold", min_value=0.1, max_value=2.0, value=0.5, step=0.1, key="limit_fc_slider")
        
        st.divider()
        st.subheader("Statistical Parameters")
        alpha_limma = st.selectbox("Adjusted P-value Threshold (α)", options=[0.001, 0.01, 0.05, 0.1], index=1, key="alpha_limma_select")
        st.divider()
        
        st.subheader("Sample Group Assignment")
        quantity_cols = st.session_state.get('selected_quantity_cols', [])
        name_mapping = st.session_state.get('column_name_mapping', {})
        trimmed_names = [name_mapping.get(col, col) for col in quantity_cols]
        
        if not trimmed_names:
            st.error("No quantification columns available. Please complete Data Upload first.")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Control Samples**")
            control_samples = st.multiselect("Select control samples", options=trimmed_names, default=trimmed_names[:len(trimmed_names)//2] if len(trimmed_names) >= 2 else [], key="control_samples_select")
        
        with col2:
            st.markdown("**Experimental Samples**")
            experimental_samples = st.multiselect("Select experimental samples", options=[s for s in trimmed_names if s not in control_samples], default=[s for s in trimmed_names if s not in control_samples][:len(trimmed_names)//2], key="experimental_samples_select")
        
        if len(control_samples) == 0 or len(experimental_samples) == 0:
            st.warning("⚠️ Please select at least one sample for both groups.")
            return
        
        config = BenchmarkConfig(limit_mv=limit_mv, limit_cv=limit_cv, limit_fc=limit_fc, alpha_limma=alpha_limma)
        
        for species, fc in species_fold_changes.items():
            if species == 'Human':
                config.expected_fc_human = fc
            elif species == 'Yeast':
                config.expected_fc_yeast = fc
            elif species == 'E. coli':
                config.expected_fc_ecoli = fc
            elif species == 'C.elegans':
                config.expected_fc_celegans = fc
        
        st.session_state.lfqbench_config = config
        st.session_state.lfq_control_samples = control_samples
        st.session_state.lfq_experimental_samples = experimental_samples
        st.session_state.detected_species = detected_species
    
    def _step2_run_analysis(self):
        """Step 2: Run LFQbench analysis"""
        
        st.header("Step 2: Run Analysis")
        
        if st.session_state.lfqbench_config is None:
            st.error("Configuration not set. Please go back to Step 1.")
            return
        
        df = st.session_state.raw_data.copy()
        species_assignments = st.session_state.get('species_assignments', pd.Series())
        
        if len(species_assignments) == 0:
            st.error("Species assignments not found.")
            return
        
        df['Species'] = species_assignments
        
        control_samples = st.session_state.lfq_control_samples
        experimental_samples = st.session_state.lfq_experimental_samples
        
        name_mapping = st.session_state.get('column_name_mapping', {})
        reverse_mapping = {v: k for k, v in name_mapping.items()}
        
        control_cols = [reverse_mapping.get(s, s) for s in control_samples]
        experimental_cols = [reverse_mapping.get(s, s) for s in experimental_samples]
        
        self.analyzer = get_lfqbench_analyzer(st.session_state.lfqbench_config)
        
        color_schemes = {
            'Default': {'Human': '#199d76', 'Yeast': '#d85f02', 'E. coli': '#7570b2', 'C.elegans': 'darkred'},
            'Viridis': {'Human': '#440154', 'Yeast': '#31688e', 'E. coli': '#35b779', 'C.elegans': '#fde724'},
            'Pastel': {'Human': '#ffadad', 'Yeast': '#ffd6a5', 'E. coli': '#caffbf', 'C.elegans': '#a0c4ff'},
            'Bold': {'Human': '#e63946', 'Yeast': '#f77f00', 'E. coli': '#06d6a0', 'C.elegans': '#118ab2'},
            'Earth': {'Human': '#8b4513', 'Yeast': '#d2691e', 'E. coli': '#556b2f', 'C.elegans': '#2f4f4f'}
        }
        
        selected_colors = color_schemes[st.session_state.lfqbench_color_scheme]
        self.visualizer = get_lfqbench_visualizer(selected_colors)
        
        should_auto_run = (st.session_state.lfqbench_results is None and st.session_state.get('lfqbench_auto_run', False))
        
        if should_auto_run or st.session_state.lfqbench_results is None:
            with st.spinner("Running LFQbench analysis..."):
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                try:
                    progress_text.text("Running analysis...")
                    progress_bar.progress(50)
                    
                    results_df, metrics, asymmetry_df = self.analyzer.run_complete_analysis(df, experimental_cols, control_cols)
                    
                    progress_bar.progress(100)
                    progress_text.text("✅ Analysis complete!")
                    
                    st.session_state.lfqbench_results = {
                        'results_df': results_df,
                        'metrics': metrics,
                        'asymmetry_df': asymmetry_df,
                        'control_cols': control_cols,
                        'experimental_cols': experimental_cols
                    }
                    
                    st.session_state.lfqbench_auto_run = False
                    st.success("✅ Analysis completed successfully!")
                    
                    st.divider()
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Proteins Analyzed", int(metrics['n_proteins']))
                    with col2:
                        st.metric("Sensitivity (%)", f"{metrics['sensitivity']:.1f}")
                    with col3:
                        st.metric("Specificity (%)", f"{metrics['specificity']:.1f}")
                    with col4:
                        st.metric("Empirical FDR (%)", f"{metrics['de_fdr']:.1f}")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
        
        elif st.session_state.lfqbench_results is not None:
            st.info("✅ Analysis results available. Proceed to Step 3.")
            metrics = st.session_state.lfqbench_results['metrics']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Proteins Analyzed", int(metrics['n_proteins']))
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
            st.error("No analysis results available.")
            return
        
        results = st.session_state.lfqbench_results
        results_df = results['results_df']
        metrics = results['metrics']
        asymmetry_df = results['asymmetry_df']
        
        if self.analyzer is None:
            self.analyzer = get_lfqbench_analyzer(st.session_state.lfqbench_config)
        
        if self.visualizer is None:
            color_schemes = {
                'Default': {'Human': '#199d76', 'Yeast': '#d85f02', 'E. coli': '#7570b2', 'C.elegans': 'darkred'},
                'Viridis': {'Human': '#440154', 'Yeast': '#31688e', 'E. coli': '#35b779', 'C.elegans': '#fde724'},
                'Pastel': {'Human': '#ffadad', 'Yeast': '#ffd6a5', 'E. coli': '#caffbf', 'C.elegans': '#a0c4ff'},
                'Bold': {'Human': '#e63946', 'Yeast': '#f77f00', 'E. coli': '#06d6a0', 'C.elegans': '#118ab2'},
                'Earth': {'Human': '#8b4513', 'Yeast': '#d2691e', 'E. coli': '#556b2f', 'C.elegans': '#2f4f4f'}
            }
            selected_colors = color_schemes[st.session_state.lfqbench_color_scheme]
            self.visualizer = get_lfqbench_visualizer(selected_colors)
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Summary", "🎯 Performance", "📈 Distributions", 
            "🌋 Volcano", "🔍 Detailed", "💾 Export"
        ])
        
        with tab1:
            st.subheader("Analysis Summary")
            fig_metrics = self.visualizer.create_summary_metrics_table(metrics)
            st.plotly_chart(fig_metrics, use_container_width=True)
            
            st.subheader("Confusion Matrix")
            fig_confusion = self.visualizer.plot_confusion_matrix(metrics)
            st.plotly_chart(fig_confusion, use_container_width=True)
        
        with tab2:
            st.subheader("Performance Metrics")
            
            st.markdown("**Fold-Change Accuracy**")
            fig_accuracy = self.visualizer.plot_fc_boxplot(results_df)
            st.plotly_chart(fig_accuracy, use_container_width=True)
            
            st.markdown("**Quantitative Precision (CV)**")
            fig_cv = self.visualizer.plot_cv_violin(results_df)
            st.plotly_chart(fig_cv, use_container_width=True)
            
            st.markdown("**Asymmetry Factors**")
            fig_asymmetry = self.visualizer.plot_asymmetry_table(asymmetry_df)
            st.plotly_chart(fig_asymmetry, use_container_width=True)
        
        with tab3:
            st.subheader("Fold-Change Distributions")
            fig_density = self.visualizer.plot_density(results_df)
            st.plotly_chart(fig_density, use_container_width=True)
            
            st.markdown("**MA Plot**")
            fig_ma = self.visualizer.plot_ma(results_df)
            st.plotly_chart(fig_ma, use_container_width=True)
        
        with tab4:
            st.subheader("Volcano Plot")
            fig_volcano = self.visualizer.plot_volcano(results_df, fc_threshold=st.session_state.lfqbench_config.limit_fc, alpha=st.session_state.lfqbench_config.alpha_limma)
            st.plotly_chart(fig_volcano, use_container_width=True)
        
        with tab5:
            st.subheader("Detailed Analysis")
            
            st.markdown("**Species-Specific Fold-Changes**")
            fig_facet = self.visualizer.plot_facet_scatter(results_df)
            st.plotly_chart(fig_facet, use_container_width=True)
            
            if len(results['experimental_cols']) >= 2:
                st.markdown("**Sample PCA**")
                try:
                    all_cols = results['control_cols'] + results['experimental_cols']
                    pca_result, var_explained = self.analyzer.perform_pca(results_df, all_cols)
                    
                    name_mapping = st.session_state.get('column_name_mapping', {})
                    sample_names = [name_mapping.get(col, col) for col in all_cols]
                    
                    fig_pca = self.visualizer.plot_pca(pca_result, var_explained, sample_names)
                    st.plotly_chart(fig_pca, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate PCA plot: {str(e)}")
        
        with tab6:
            st.subheader("Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_results = results_df.to_csv(index=False)
                st.download_button(label="📥 Download Results (CSV)", data=csv_results, file_name="lfqbench_results.csv", mime="text/csv", use_container_width=True, key="download_results_btn")
            
            with col2:
                metrics_df = pd.DataFrame([metrics])
                csv_metrics = metrics_df.to_csv(index=False)
                st.download_button(label="📥 Download Metrics (CSV)", data=csv_metrics, file_name="lfqbench_metrics.csv", mime="text/csv", use_container_width=True, key="download_metrics_btn")
            
            with col3:
                try:
                    zip_data = self.visualizer.export_all_figures()
                    st.download_button(label="📦 Download All Figures (ZIP)", data=zip_data, file_name="lfqbench_figures.zip", mime="application/zip", use_container_width=True, key="download_figures_btn")
                except Exception as e:
                    st.error(f"Could not export figures: {str(e)}")
    
    def _render_navigation(self):
        """Render navigation buttons"""
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.session_state.lfqbench_step > 1:
                if st.button("⬅️ Previous", use_container_width=True, key="lfq_prev_btn"):
                    st.session_state.lfqbench_step -= 1
                    st.rerun()
        
        with col2:
            if st.button("🔄 Reset", use_container_width=True, key="lfq_reset_btn"):
                st.session_state.lfqbench_step = 1
                st.session_state.lfqbench_config = None
                st.session_state.lfqbench_results = None
                st.session_state.lfqbench_auto_run = False
                st.rerun()
        
        with col3:
            can_proceed = self._can_proceed_to_next_step()
            
            if st.session_state.lfqbench_step < 3:
                if st.button("Next ➡️", use_container_width=True, disabled=not can_proceed, key="lfq_next_btn"):
                    if st.session_state.lfqbench_step == 1:
                        st.session_state.lfqbench_auto_run = True
                    
                    st.session_state.lfqbench_step += 1
                    st.rerun()
    
    def _can_proceed_to_next_step(self) -> bool:
        """Check if can proceed to next step"""
        step = st.session_state.lfqbench_step
        
        if step == 1:
            return (st.session_state.lfqbench_config is not None and
                   len(st.session_state.get('lfq_control_samples', [])) > 0 and
                   len(st.session_state.get('lfq_experimental_samples', [])) > 0)
        elif step == 2:
            return st.session_state.lfqbench_results is not None
        else:
            return True


def run_lfqbench_module():
    """Convenience function to run LFQbench module"""
    module = LFQbenchModule()
    module.run()
