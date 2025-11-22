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
    
    """
LFQbench Visualization Module
Generates all plots for benchmark analysis following R script specifications.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional


class LFQbenchVisualizer:
    """Generate all LFQbench visualizations"""
    
    def __init__(self, color_map: Optional[Dict[str, str]] = None):
        self.color_map = color_map or {
            'Human': '#199d76',
            'Yeast': '#d85f02',
            'E. coli': '#7570b2',
            'C.elegans': 'darkred'
        }
    
    def plot_density(self, df: pd.DataFrame, title: str = "Log2 Fold-Change Distribution") -> go.Figure:
        """
        Create density plot of log2 fold-changes by species
        Replicates: ggplot geom_density
        """
        fig = go.Figure()
        
        for species in df['Species'].unique():
            species_data = df[df['Species'] == species]['log2_fc'].dropna()
            
            if len(species_data) > 0:
                # Create density using histogram with KDE
                fig.add_trace(go.Violin(
                    x=species_data,
                    name=species,
                    line_color=self.color_map.get(species, 'gray'),
                    fillcolor=self.color_map.get(species, 'gray'),
                    opacity=0.6,
                    showlegend=True
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Log2 Fold-Change",
            yaxis_title="Density",
            template="plotly_white",
            height=500,
            showlegend=True
        )
        
        return fig
    
    def plot_volcano(self, df: pd.DataFrame, 
                    fc_threshold: float = 0.5,
                    alpha: float = 0.01,
                    title: str = "Volcano Plot") -> go.Figure:
        """
        Create volcano plot (log2FC vs -log10 p-value)
        Replicates: ggplot geom_point with volcano layout
        """
        df = df.copy()
        df['-log10_pval'] = -np.log10(df['p_adj'].replace(0, 1e-300))
        
        fig = go.Figure()
        
        for species in df['Species'].unique():
            species_data = df[df['Species'] == species]
            
            fig.add_trace(go.Scatter(
                x=species_data['log2_fc'],
                y=species_data['-log10_pval'],
                mode='markers',
                name=species,
                marker=dict(
                    color=self.color_map.get(species, 'gray'),
                    size=6,
                    opacity=0.6
                ),
                text=species_data.get('Protein.Names', species_data.index),
                hovertemplate='<b>%{text}</b><br>Log2FC: %{x:.2f}<br>-log10(padj): %{y:.2f}<extra></extra>'
            ))
        
        # Add threshold lines
        fig.add_hline(y=-np.log10(alpha), line_dash="dot", line_color="gray", opacity=0.7)
        fig.add_vline(x=fc_threshold, line_dash="dot", line_color="gray", opacity=0.7)
        fig.add_vline(x=-fc_threshold, line_dash="dot", line_color="gray", opacity=0.7)
        
        fig.update_layout(
            title=title,
            xaxis_title="Log2 Fold-Change",
            yaxis_title="-log10(adjusted p-value)",
            template="plotly_white",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def plot_fc_boxplot(self, df: pd.DataFrame, 
                       title: str = "Fold-Change Accuracy") -> go.Figure:
        """
        Boxplot of |measured - expected| fold-changes by species
        Shows accuracy metric
        """
        fig = go.Figure()
        
        for species in df['Species'].unique():
            species_data = df[df['Species'] == species]
            deviations = species_data['fc_deviation'].dropna()
            
            fig.add_trace(go.Box(
                y=deviations,
                name=species,
                marker_color=self.color_map.get(species, 'gray'),
                boxmean='sd'  # Show mean and std dev
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Species",
            yaxis_title="|Expected - Measured| Log2 FC",
            template="plotly_white",
            height=500,
            showlegend=False
        )
        
        return fig
    
    def plot_cv_violin(self, df: pd.DataFrame,
                      title: str = "Coefficient of Variation") -> go.Figure:
        """
        Violin plot of CV distribution by species
        Shows precision metric
        """
        # Reshape data for plotting
        cv_data = []
        for _, row in df.iterrows():
            cv_data.append({'Species': row['Species'], 'CV': row['exp_cv'], 'Condition': 'Experimental'})
            cv_data.append({'Species': row['Species'], 'CV': row['ctr_cv'], 'Condition': 'Control'})
        
        cv_df = pd.DataFrame(cv_data)
        
        fig = px.violin(
            cv_df,
            x='Species',
            y='CV',
            color='Species',
            color_discrete_map=self.color_map,
            box=True,
            points=False
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Species",
            yaxis_title="CV (%)",
            template="plotly_white",
            height=500,
            showlegend=False,
            yaxis_range=[0, 30]
        )
        
        return fig
    
    def plot_ma(self, df: pd.DataFrame,
                title: str = "MA Plot") -> go.Figure:
        """
        MA plot: log2(mean intensity) vs log2(fold-change)
        Replicates: ggplot geom_point for MA plot
        """
        df = df.copy()
        df['log2_mean'] = np.log2((df['exp_mean'] + df['ctr_mean']) / 2)
        
        fig = go.Figure()
        
        for species in df['Species'].unique():
            species_data = df[df['Species'] == species]
            
            fig.add_trace(go.Scatter(
                x=species_data['log2_mean'],
                y=species_data['log2_fc'],
                mode='markers',
                name=species,
                marker=dict(
                    color=self.color_map.get(species, 'gray'),
                    size=5,
                    opacity=0.5
                ),
                text=species_data.get('Protein.Names', species_data.index),
                hovertemplate='<b>%{text}</b><br>Mean: %{x:.2f}<br>Log2FC: %{y:.2f}<extra></extra>'
            ))
        
        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        fig.update_layout(
            title=title,
            xaxis_title="Log2(Mean Intensity)",
            yaxis_title="Log2 Fold-Change",
            template="plotly_white",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def plot_facet_scatter(self, df: pd.DataFrame,
                          title: str = "Control vs Experimental") -> go.Figure:
        """
        Faceted scatter plot: control mean vs log2FC
        One panel per species
        """
        species_list = df['Species'].unique()
        n_species = len(species_list)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=n_species,
            subplot_titles=list(species_list),
            horizontal_spacing=0.1
        )
        
        for i, species in enumerate(species_list, 1):
            species_data = df[df['Species'] == species]
            
            fig.add_trace(
                go.Scatter(
                    x=np.log2(species_data['ctr_mean']),
                    y=species_data['log2_fc'],
                    mode='markers',
                    marker=dict(
                        color=self.color_map.get(species, 'gray'),
                        size=5,
                        opacity=0.5
                    ),
                    name=species,
                    showlegend=False
                ),
                row=1, col=i
            )
            
            # Add median line
            median_fc = species_data['log2_fc'].median()
            fig.add_hline(
                y=median_fc,
                line_dash="dash",
                line_color=self.color_map.get(species, 'gray'),
                row=1, col=i
            )
        
        fig.update_layout(
            title=title,
            template="plotly_white",
            height=400,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Log2(Control Mean)")
        fig.update_yaxes(title_text="Log2 Fold-Change")
        
        return fig
    
    def plot_pca(self, pca_result: np.ndarray,
                variance_explained: np.ndarray,
                sample_names: List[str],
                title: str = "PCA - Sample Overview") -> go.Figure:
        """
        PCA plot of samples
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            mode='markers+text',
            text=sample_names,
            textposition='top center',
            marker=dict(size=12, color='steelblue'),
            textfont=dict(size=10)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=f"PC1 ({variance_explained[0]*100:.1f}%)",
            yaxis_title=f"PC2 ({variance_explained[1]*100:.1f}%)" if len(variance_explained) > 1 else "PC2",
            template="plotly_white",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def plot_asymmetry_table(self, asymmetry_df: pd.DataFrame) -> go.Figure:
        """
        Display asymmetry factors as interactive table
        """
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(asymmetry_df.columns),
                fill_color='lightgray',
                align='left',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=[asymmetry_df[col] for col in asymmetry_df.columns],
                fill_color='white',
                align='left',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(
            title="Asymmetry Factors by Species",
            height=300
        )
        
        return fig
    
    def plot_confusion_matrix(self, metrics: Dict[str, float]) -> go.Figure:
        """
        Heatmap of confusion matrix
        """
        confusion = np.array([
            [metrics['tn'], metrics['fp']],
            [metrics['fn'], metrics['tp']]
        ])
        
        fig = go.Figure(data=go.Heatmap(
            z=confusion,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'],
            text=confusion,
            texttemplate='%{text}',
            textfont={"size": 16},
            colorscale='Blues',
            showscale=False
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400,
            template="plotly_white"
        )
        
        return fig
    
    def create_summary_metrics_table(self, metrics: Dict[str, float]) -> go.Figure:
        """
        Create summary metrics display table
        """
        metrics_display = {
            'Metric': [
                'Sensitivity (%)',
                'Specificity (%)',
                'Empirical FDR (%)',
                'Accuracy',
                'Trueness',
                'CV Mean (%)',
                'CV Median (%)',
                'True Positives',
                'False Positives',
                'True Negatives',
                'False Negatives',
                'Total Proteins'
            ],
            'Value': [
                f"{metrics['sensitivity']:.2f}",
                f"{metrics['specificity']:.2f}",
                f"{metrics['de_fdr']:.2f}",
                f"{metrics['accuracy']:.3f}",
                f"{metrics['trueness']:.3f}",
                f"{metrics['cv_mean']:.2f}",
                f"{metrics['cv_median']:.2f}",
                str(int(metrics['tp'])),
                str(int(metrics['fp'])),
                str(int(metrics['tn'])),
                str(int(metrics['fn'])),
                str(int(metrics['n_proteins']))
            ]
        }
        
        df_metrics = pd.DataFrame(metrics_display)
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Metric</b>', '<b>Value</b>'],
                fill_color='steelblue',
                align='left',
                font=dict(size=13, color='white')
            ),
            cells=dict(
                values=[df_metrics['Metric'], df_metrics['Value']],
                fill_color='white',
                align='left',
                font=dict(size=12)
            )
        )])
        
        fig.update_layout(
            title="Performance Metrics Summary",
            height=500
        )
        
        return fig


def get_lfqbench_visualizer(color_map: Optional[Dict[str, str]] = None) -> LFQbenchVisualizer:
    """Get LFQbench visualizer instance"""
    return LFQbenchVisualizer(color_map)

        
        with col3:
            can_proceed = self._can_proceed_to_next_step()
            
            if st.session_state.lfqbench_step < 3:
                if st.button("Next ➡️", use_container_width=True, disabled=not can_proceed, key="lfq_next_btn"):
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
