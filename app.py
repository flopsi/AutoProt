"""
Enhanced ProteoFlow - Proteomics QC and Analysis Platform
Complete guided workflow from raw data to statistical results
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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


# ==================== HELPER FUNCTIONS ====================

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame: strip whitespace from column names and index
    Convert all numeric columns to float
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Strip whitespace from index if it's string type
    if df.index.dtype == 'object':
        df.index = df.index.str.strip()
    
    # Convert all numeric-like columns to float
    for col in df.columns:
        # Try to convert to numeric, coerce errors to NaN
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Ensure it's float
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(float)
    
    return df


def generate_mock_proteins(n_proteins: int = 500) -> pd.DataFrame:
    """Generate mock protein intensity data for demo"""
    np.random.seed(42)
    
    proteins = [f"Protein_{i:04d}" for i in range(n_proteins)]
    
    # Generate data for two conditions with 3 replicates each
    data = {}
    
    # Condition A (Young) - 3 replicates
    for i in range(1, 4):
        base = np.random.lognormal(10, 2, n_proteins)
        noise = np.random.normal(1, 0.15, n_proteins)
        data[f'Condition_A_R{i}'] = base * noise
    
    # Condition B (Old) - 3 replicates
    for i in range(1, 4):
        base = np.random.lognormal(10, 2, n_proteins)
        # Some proteins upregulated
        upregulated = np.random.choice([False, True], n_proteins, p=[0.9, 0.1])
        base[upregulated] *= 2
        noise = np.random.normal(1, 0.15, n_proteins)
        data[f'Condition_B_R{i}'] = base * noise
    
    df = pd.DataFrame(data, index=proteins)
    
    # Add some missing values
    mask = np.random.random(df.shape) > 0.95
    df[mask] = np.nan
    
    return df


# ==================== WORKFLOW STEPS ====================

def step1_load_and_map():
    """Step 1: Load data and map replicates to conditions with clean naming"""
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
                    raw_df = pd.read_csv(uploaded_file, index_col=0)
                else:
                    raw_df = pd.read_csv(uploaded_file, sep='\t', index_col=0)
                
                # Clean the raw dataframe
                raw_df = clean_dataframe(raw_df)
                
                # Store raw data temporarily
                st.session_state.raw_data_temp = raw_df
                st.success(f"✅ Loaded {len(raw_df)} proteins with {len(raw_df.columns)} columns")
                
                with st.expander("Preview raw data"):
                    st.dataframe(raw_df.head(10), use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    with col2:
        st.subheader("🧪 Or Try Demo Data")
        if st.button("Load Demo Dataset", use_container_width=True):
            raw_df = generate_mock_proteins(n_proteins=500)
            st.session_state.raw_data_temp = raw_df
            st.success("✅ Demo dataset loaded!")
            st.rerun()
    
    # Replicate mapping interface with clean naming
    if hasattr(st.session_state, 'raw_data_temp'):
        st.markdown("---")
        st.subheader("🔗 Map Replicates to Conditions")
        
        raw_df = st.session_state.raw_data_temp
        
        # Get all columns (any type)
        available_cols = raw_df.columns.tolist()
        
        st.info(f"Found {len(available_cols)} columns in your data")
        
        n_conditions = st.number_input(
            "Number of experimental conditions",
            min_value=2, max_value=4, value=2
        )
        
        # Store column mapping: {clean_name: original_column}
        column_mapping = {}
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
                # Generate clean names for each replicate
                clean_names = []
                
                st.markdown(f"*Clean names for {condition_name}:*")
                
                for j, orig_col in enumerate(selected_cols):
                    # Auto-generate clean name
                    default_clean = f"{condition_name}_R{j+1}"
                    
                    clean_name = st.text_input(
                        f"Rename: {orig_col[:50]}...",
                        value=default_clean,
                        key=f"clean_name_{i}_{j}"
                    )
                    
                    column_mapping[clean_name] = orig_col
                    clean_names.append(clean_name)
                
                condition_mapping[condition_name] = clean_names
        
        # Create clean DataFrame button
        if len(condition_mapping) >= 2 and len(column_mapping) > 0:
            st.markdown("---")
            
            with st.expander("📋 Preview Column Mapping"):
                mapping_preview = pd.DataFrame([
                    {"Clean Name": clean, "Original Column": orig}
                    for clean, orig in column_mapping.items()
                ])
                st.dataframe(mapping_preview, use_container_width=True, hide_index=True)
            
            if st.button("✅ Create Clean Dataset & Proceed", type="primary", use_container_width=True):
                
                with st.spinner("Creating clean dataset..."):
                    # Create new clean DataFrame
                    clean_df = pd.DataFrame(index=raw_df.index)
                    
                    conversion_issues = []
                    
                    for clean_name, orig_col in column_mapping.items():
                        # Extract and convert to numeric
                        try:
                            clean_df[clean_name] = pd.to_numeric(
                                raw_df[orig_col], 
                                errors='coerce'
                            ).astype(float)
                            
                            # Check if conversion caused issues
                            orig_valid = raw_df[orig_col].notna().sum()
                            new_valid = clean_df[clean_name].notna().sum()
                            
                            if new_valid < orig_valid:
                                conversion_issues.append(
                                    f"{clean_name}: {orig_valid - new_valid} non-numeric values converted to NaN"
                                )
                        except Exception as e:
                            st.error(f"Error converting {orig_col} to {clean_name}: {str(e)}")
                            return
                    
                    # Store clean data and mappings
                    st.session_state.raw_data = clean_df.copy()  # Working data (can be transformed)
                    st.session_state.raw_data_original = clean_df.copy()  # NEVER MODIFIED - for CV only
                    st.session_state.replicate_mapping = condition_mapping
                    st.session_state.column_mapping = column_mapping  # For reference
                    
                    # Clean up temp
                    del st.session_state.raw_data_temp
                    
                    st.success(f"✅ Created clean dataset with {len(clean_df.columns)} replicates")
                    
                    # Show conversion issues if any
                    if conversion_issues:
                        with st.expander("⚠️ Conversion Warnings"):
                            for issue in conversion_issues:
                                st.warning(issue)
                    
                    # Show clean data preview
                    with st.expander("Preview clean dataset"):
                        st.dataframe(clean_df.head(10), use_container_width=True)
                        st.markdown("**Data types:**")
                        st.text("All columns: float64")
                    
                    # Proceed to next step
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
            else:
                st.success(
                    f"✅ {normal_count}/{total_count} samples show normal distribution. "
                    "Data appears reasonably normal, but log transformation may still improve results."
                )
    
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
            # Apply transformation
            st.session_state.raw_data = transformed_df
            # DON'T overwrite raw_data_for_cv - it stays original!

            st.session_state.log_transformed = True
            
            st.success("✅ Transformation complete!")
            
            # Show before/after comparison - SIDE BY SIDE
            st.markdown("### Before & After Comparison")
            
            sample_col = all_replicates[0]
            
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


# ==================== MAIN ====================

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="ProteoFlow - Proteomics Analysis",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧬 ProteoFlow - Proteomics Analysis Platform")
    
    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Analysis Workflow")
        
        steps = [
            "1️⃣ Load & Map Data",
            "2️⃣ Check Normality",
            "3️⃣ Transform Data",
            "4️⃣ QC Analysis",
            "5️⃣ Statistical Analysis"
        ]
        
        for i, step_name in enumerate(steps, 1):
            if st.session_state.step == i:
                st.markdown(f"**→ {step_name}**")
            else:
                st.markdown(f"{step_name}")
        
        st.markdown("---")
        
        # Navigation buttons
        if st.session_state.step > 1:
            if st.button("⬅️ Previous Step", use_container_width=True):
                st.session_state.step -= 1
                st.rerun()
        
        if st.button("🔄 Restart Analysis", use_container_width=True):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.step = 1
            st.rerun()
    
    # Route to appropriate step
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


if __name__ == "__main__":
    main()
