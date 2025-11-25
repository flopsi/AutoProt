"""
Enhanced ProteoFlow - Proteomics QC and Analysis Platform
Complete guided workflow from raw data to statistical results
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# Import utilities
from utils.data_generator import generate_mock_proteins
from utils.stats import (
    check_normality,
    log2_transform,
    batch_process_proteins
)
from components.qc_plots import render_qc_dashboard
from components.plots import create_volcano_plot
from components.tables import render_data_table
from components.stats import render_stats_cards
from services.gemini_service import analyze_proteins, chat_with_data


# Page configuration
st.set_page_config(
    page_title="ProteoFlow - QC & Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'replicate_mapping' not in st.session_state:
    st.session_state.replicate_mapping = {}
if 'transformed_data' not in st.session_state:
    st.session_state.transformed_data = None
if 'log_transformed' not in st.session_state:
    st.session_state.log_transformed = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = None


def main():
    """Main application"""
    
    st.title("🔬 ProteoFlow - Proteomics QC & Analysis Platform")
    st.markdown("### Guided workflow from raw data to publication-ready results")
    
    # Progress indicator
    render_progress_bar()
    
    st.markdown("---")
    
    # Step router
    if st.session_state.step == 1:
        step1_load_and_map()
    elif st.session_state.step == 2:
        step2_check_normality()
    elif st.session_state.step == 3:
        step3_transform_data()
    elif st.session_state.step == 4:
        step4_qc_analysis()
    elif st.session_state.step == 5:
        step5_statistical_analysis()


def render_progress_bar():
    """Render workflow progress indicator"""
    steps = [
        "1️⃣ Load & Map",
        "2️⃣ Normality Check",
        "3️⃣ Transform",
        "4️⃣ QC Analysis",
        "5️⃣ Statistics"
    ]
    
    cols = st.columns(5)
    for idx, (col, step_name) in enumerate(zip(cols, steps)):
        with col:
            if idx + 1 < st.session_state.step:
                st.success(step_name)
            elif idx + 1 == st.session_state.step:
                st.info(f"**{step_name}**")
            else:
                st.text(step_name)


def step1_load_and_map():
    """Step 1: Load data and map replicates to conditions"""
    st.header("Step 1: Load Data & Map Replicates")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📂 Upload Your Data")
        uploaded_file = st.file_uploader(
            "Upload CSV or TSV file with protein intensities",
            type=['csv', 'tsv', 'txt']
        )
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file, index_col=0)
                else:
                    df = pd.read_csv(uploaded_file, sep='\t', index_col=0)
                
                st.session_state.raw_data = df
                st.success(f"✅ Loaded {len(df)} proteins with {len(df.columns)} columns")
                
                with st.expander("Preview data"):
                    st.dataframe(df.head(10), use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    with col2:
        st.subheader("🧪 Or Try Demo Data")
        if st.button("Load Demo Dataset", use_container_width=True):
            df = generate_mock_proteins(n_proteins=500)
            st.session_state.raw_data = df
            st.success("✅ Demo dataset loaded!")
            st.rerun()
    
    # Replicate mapping interface
    if st.session_state.raw_data is not None:
        st.markdown("---")
        st.subheader("🔗 Map Replicates to Conditions")
        
        df = st.session_state.raw_data
        available_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_conditions = st.number_input(
            "Number of experimental conditions",
            min_value=2, max_value=4, value=2
        )
        
        condition_mapping = {}
        
        for i in range(n_conditions):
            st.markdown(f"**Condition {i+1}:**")
            condition_name = st.text_input(
                f"Name for condition {i+1}",
                value=f"Condition_{chr(65+i)}",
                key=f"cond_name_{i}"
            )
            
            selected_cols = st.multiselect(
                f"Select replicate columns for {condition_name}",
                options=available_cols,
                key=f"cond_cols_{i}"
            )
            
            if selected_cols:
                condition_mapping[condition_name] = selected_cols
        
        if len(condition_mapping) >= 2:
            if st.button("✅ Confirm Mapping & Proceed", type="primary", use_container_width=True):
                st.session_state.replicate_mapping = condition_mapping
                st.session_state.step = 2
                st.rerun()


def step2_check_normality():
    """Step 2: Check data normality and recommend transformation"""
    st.header("Step 2: Check Data Normality")
    
    df = st.session_state.raw_data
    mapping = st.session_state.replicate_mapping
    
    # Get all replicate columns
    all_replicates = []
    for cols in mapping.values():
        all_replicates.extend(cols)
    
    st.info("📊 Testing normality of intensity distributions using Shapiro-Wilk test")
    
    if st.button("🔍 Run Normality Tests", type="primary"):
        
        with st.spinner("Running normality tests..."):
            results = []
            
            for col in all_replicates:
                data = df[col].dropna()
                if len(data) > 3:
                    normality = check_normality(data)
                    results.append({
                        'Sample': col,
                        'Normal': '✅' if normality['is_normal'] else '❌',
                        'p-value': f"{normality['p_value']:.4f}",
                        'Skewness': f"{normality['skewness']:.2f}",
                        'Kurtosis': f"{normality['kurtosis']:.2f}"
                    })
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            # Recommendation
            normal_count = results_df['Normal'].str.contains('✅').sum()
            total_count = len(results_df)
            
            st.markdown("### 💡 Recommendation")
            
            if normal_count / total_count < 0.5:
                st.warning(
                    f"⚠️ Only {normal_count}/{total_count} samples show normal distribution. "
                    "**Log2 transformation is recommended** to improve normality and stabilize variance."
                )
                recommend_transform = True
            else:
                st.success(
                    f"✅ {normal_count}/{total_count} samples show normal distribution. "
                    "Data appears reasonably normal, but log transformation may still improve results."
                )
                recommend_transform = False
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⏭️ Skip Transformation", use_container_width=True):
            st.session_state.transformed_data = df
            st.session_state.log_transformed = False
            st.session_state.step = 4  # Skip step 3
            st.rerun()
    
    with col2:
        if st.button("➡️ Proceed to Transformation", type="primary", use_container_width=True):
            st.session_state.step = 3
            st.rerun()


def step3_transform_data():
    """Step 3: Apply log2 transformation"""
    st.header("Step 3: Data Transformation")
    
    df = st.session_state.raw_data
    mapping = st.session_state.replicate_mapping
    
    # Get all replicate columns
    all_replicates = []
    for cols in mapping.values():
        all_replicates.extend(cols)
    
    st.markdown("### Log2 Transformation")
    st.info("Log2 transformation normalizes skewed distributions and stabilizes variance across intensity ranges.")
    
    pseudocount = st.number_input(
        "Pseudocount value (added before log to avoid log(0))",
        min_value=0.1, max_value=10.0, value=1.0, step=0.1
    )
    
    if st.button("🔄 Apply Transformation", type="primary"):
        
        with st.spinner("Applying log2 transformation..."):
            transformed_df = log2_transform(df, all_replicates, pseudocount)
            st.session_state.transformed_data = transformed_df
            st.session_state.log_transformed = True
            
            st.success("✅ Transformation complete!")
            
            # Show before/after comparison
            st.markdown("### Before & After Comparison")
            
            sample_col = all_replicates[0]
            
            from plotly.subplots import make_subplots
            
            # Create side-by-side subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Original Distribution", "Log2 Transformed Distribution"),
                horizontal_spacing=0.15
            )
            
            # Before (left)
            fig.add_trace(
                go.Histogram(
                    x=df[sample_col].dropna(),
                    name="Original",
                    marker_color='#3498db',
                    nbinsx=50
                ),
                row=1, col=1
            )
            
            # After (right)
            fig.add_trace(
                go.Histogram(
                    x=transformed_df[sample_col].dropna(),
                    name="Log2 Transformed",
                    marker_color='#2ecc71',
                    nbinsx=50
                ),
                row=1, col=2
            )
            
            # Update axes
            fig.update_xaxes(title_text="Intensity", row=1, col=1)
            fig.update_xaxes(title_text="Log2 Intensity", row=1, col=2)
            fig.update_yaxes(title_text="Count", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=2)
            
            fig.update_layout(
                title_text=f"Distribution Comparison: {sample_col}",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

            
            # Re-test normality
            st.markdown("### Updated Normality Test")
            normality = check_normality(transformed_df[sample_col])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("p-value", f"{normality['p_value']:.4f}")
            with col2:
                st.metric("Skewness", f"{normality['skewness']:.2f}")
            with col3:
                st.metric("Kurtosis", f"{normality['kurtosis']:.2f}")
            
            if normality['is_normal']:
                st.success("✅ Data now shows normal distribution!")
            else:
                st.info("ℹ️ Distribution improved but still non-normal. This is common for proteomics data.")
    
    st.markdown("---")
    
    if st.session_state.transformed_data is not None:
        if st.button("➡️ Proceed to QC Analysis", type="primary", use_container_width=True):
            st.session_state.step = 4
            st.rerun()


def step4_qc_analysis():
    """Step 4: Comprehensive QC analysis"""
    st.header("Step 4: Quality Control Analysis")
    
    # Get working data
    df = st.session_state.transformed_data if st.session_state.transformed_data is not None else st.session_state.raw_data
    mapping = st.session_state.replicate_mapping
    
    # Get all replicate columns
    all_replicates = []
    for cols in mapping.values():
        all_replicates.extend(cols)
    
    # Render complete QC dashboard
    render_qc_dashboard(df, all_replicates, mapping, st.session_state.log_transformed)
    
    st.markdown("---")
    
    if st.button("➡️ Proceed to Statistical Analysis", type="primary", use_container_width=True):
        st.session_state.step = 5
        st.rerun()


def step5_statistical_analysis():
    """Step 5: Statistical analysis with t-tests and volcano plot"""
    st.header("Step 5: Statistical Analysis")
    
    # Get working data
    df = st.session_state.transformed_data if st.session_state.transformed_data is not None else st.session_state.raw_data
    mapping = st.session_state.replicate_mapping
    
    # For simplicity, compare first two conditions
    conditions = list(mapping.keys())
    
    if len(conditions) < 2:
        st.error("Need at least 2 conditions for statistical comparison")
        return
    
    st.info(f"📊 Performing t-tests: **{conditions[0]}** vs **{conditions[1]}**")
    
    # Run batch analysis
    if st.session_state.results_df is None:
        with st.spinner("Running statistical tests..."):
            results_df = batch_process_proteins(
                df,
                mapping[conditions[0]],
                mapping[conditions[1]],
                st.session_state.log_transformed
            )
            
            # Calculate -log10(p-value)
            results_df['negLog10PValue'] = -np.log10(results_df['p_value'].clip(lower=1e-300))
            
            # Add gene names if available
            if 'gene' in df.columns:
                results_df['gene'] = df['gene']
            else:
                results_df['gene'] = results_df['protein_id']
            
            st.session_state.results_df = results_df
    
    results_df = st.session_state.results_df
    
    # Controls
    col1, col2 = st.columns(2)
    
    with col1:
        p_threshold = st.slider(
            "p-value threshold (-log10)",
            min_value=0.5, max_value=5.0, value=1.3, step=0.1
        )
    
    with col2:
        fc_threshold = st.slider(
            "Fold change threshold (log2)",
            min_value=0.0, max_value=3.0, value=1.0, step=0.1
        )
    
    # Assign significance
    results_df['significance'] = 'NS'
    results_df.loc[
        (results_df['negLog10PValue'] >= p_threshold) & 
        (results_df['log2_fold_change'] >= fc_threshold), 
        'significance'
    ] = 'UP'
    results_df.loc[
        (results_df['negLog10PValue'] >= p_threshold) & 
        (results_df['log2_fold_change'] <= -fc_threshold), 
        'significance'
    ] = 'DOWN'
    
    # Summary stats
    render_stats_cards(results_df)
    
    st.markdown("---")
    
    # Volcano plot
    st.subheader("🌋 Volcano Plot")
    fig_volcano = create_volcano_plot(results_df, p_threshold, fc_threshold)
    if fig_volcano:
        st.plotly_chart(fig_volcano, use_container_width=True)
    
    st.markdown("---")
    
    # Results table
    st.subheader("📊 Significant Proteins")
    
    sig_df = results_df[results_df['significance'] != 'NS'].copy()
    sig_df = sig_df.sort_values('negLog10PValue', ascending=False)
    
    display_cols = [
        'gene', 'log2_fold_change', 'p_value', 'negLog10PValue',
        'mean_condition_a', 'mean_condition_b', 'significance'
    ]
    display_cols = [col for col in display_cols if col in sig_df.columns]
    
    st.dataframe(
        sig_df[display_cols].head(50),
        use_container_width=True,
        hide_index=True
    )
    
    st.info(f"Showing top 50 of {len(sig_df)} significant proteins")
    
    # Download options
    st.markdown("---")
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download all results
        csv_all = results_df.to_csv(index=False)
        st.download_button(
            "📥 Download All Results (CSV)",
            data=csv_all,
            file_name="proteomics_results_all.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Download significant only
        csv_sig = sig_df.to_csv(index=False)
        st.download_button(
            "📥 Download Significant Only (CSV)",
            data=csv_sig,
            file_name="proteomics_results_significant.csv",
            mime="text/csv",
            use_container_width=True
        )


if __name__ == "__main__":
    main()
