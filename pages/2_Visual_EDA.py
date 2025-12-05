"""
pages/2_Visual_EDA.py
Complete transformation comparison dashboard
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from helpers.transforms import apply_transformation, TRANSFORM_NAMES, TRANSFORM_DESCRIPTIONS
from helpers.comparison import compare_transformations, find_best_transformation
from helpers.constants import get_theme

st.set_page_config(page_title="Visual EDA", layout="wide")

st.title("🔬 Transformation Comparison Dashboard")

# Load data
protein_data = st.session_state.get("protein_data")
if not protein_data:
    st.error("❌ No data loaded. Please upload data first.")
    st.stop()

st.success(f"✅ Loaded: {len(protein_data.raw):,} proteins × {len(protein_data.numeric_cols)} samples")

# ============================================================================
# TABS: Single View vs Comparison
# ============================================================================

tab1, tab2 = st.tabs(["📊 Single Transformation", "⚖️ Compare All Transformations"])

with tab1:
    # === SINGLE TRANSFORMATION VIEW (your working 6-panel) ===
    st.subheader("Single Transformation Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        transform = st.selectbox("Transformation", list(TRANSFORM_NAMES.keys()), 
                               format_func=lambda x: TRANSFORM_NAMES[x], index=1)
    with col2:
        max_plots = st.slider("Max samples", 1, min(12, len(protein_data.numeric_cols)), 6)
    
    # Apply transformation
    df_trans, trans_cols = apply_transformation(
        protein_data.raw, protein_data.numeric_cols[:max_plots], transform
    )
    
    # 6-panel plots (your working version)
    theme = get_theme(st.session_state.get("theme", "light"))
    fig = make_subplots(rows=2, cols=3, subplot_titles=trans_cols[:6])
    
    for i, col in enumerate(trans_cols[:6]):
        row, pos = (i//3)+1, (i%3)+1
        values = df_trans[col][df_trans[col]>1.0].dropna()
        if len(values)>0:
            fig.add_trace(go.Histogram(x=values, nbinsx=40, opacity=0.75, 
                                     marker_color=theme.get('primary', '#1f77b4'),
                                     showlegend=False), row=row, col=pos)
            fig.add_vline(values.mean(), line_dash="dash", line_color="red", 
                         row=row, col=pos)
    
    fig.update_layout(height=650, showlegend=False, 
                     title=f"{TRANSFORM_NAMES[transform]} Distributions")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # === COMPARISON DASHBOARD ===
    st.subheader("Multi-Transformation Comparison")
    
    col1, col2 = st.columns(2)
    with col1:
        transformations = st.multiselect(
            "Select transformations to compare",
            options=list(TRANSFORM_NAMES.keys())[1:],  # Skip 'raw'
            default=['log2', 'log10', 'sqrt', 'arcsinh', 'vst'],
            format_func=lambda x: TRANSFORM_NAMES[x]
        )
    with col2:
        max_cols_analysis = st.slider("Max samples for analysis", 3, 12, 6)
    
    if st.button("🚀 Run Comparison Analysis", type="primary"):
        with st.spinner("Comparing transformations..."):
            summary_df, all_transformed_data = compare_transformations(
                protein_data.raw,
                protein_data.numeric_cols,
                transformations,
                max_cols_analysis
            )
        
        # === COMPUTE COMBINED SCORE FIRST ===
        summary_df['combined_score'] = (
            summary_df['shapiro_p'].rank(ascending=False) +
            (1 - summary_df['mean_var_corr'].abs()).rank(ascending=False) +
            summary_df['n_significant'].rank(ascending=False)
        )
        
        # === SUMMARY TABLE (BULLETPROOF STYLING) ===
        st.subheader("📋 Comparison Summary")
        
        # Compute best index first (outside styler)
        if not summary_df.empty:
            best_idx = summary_df['combined_score'].idxmax()
            
            # Simple conditional formatting without complex styler
            display_df = summary_df.copy()
            display_df['Rank'] = display_df['combined_score'].rank(ascending=False).astype(int)
            
            # Manual highlighting
            def color_rank(val):
                if pd.isna(val):
                    return ''
                rank = int(val)
                if rank == 1:
                    return 'background-color: #B5BD00; color: white; font-weight: bold'
                elif rank == 2:
                    return 'background-color: #D4AF37; color: white'
                elif rank == 3:
                    return 'background-color: #C0C0C0; color: black'
                return ''
            
            # Format numeric columns
            styled_df = display_df.style.format({
                'shapiro_p': '{:.2e}',
                'mean_var_corr': '{:.3f}',
                'n_significant': '{:,}',
                'n_up': '{:,}',
                'n_down': '{:,}'
            }).applymap(color_rank, subset=['Rank'])
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Show best explicitly
            best_row = summary_df.loc[best_idx]
            st.info(f"**🥇 #1: {TRANSFORM_NAMES[best_row['method']]}** (Score: {best_row['combined_score']:.1f})")
        else:
            st.warning("No comparison results available")

        # === SAVE RESULTS ===
        csv = summary_df.round(4).to_csv(index=False)
        st.download_button(
            label="💾 Download Results",
            data=csv,
            file_name=f"transformation_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
        
        st.session_state.all_transformed_data = all_transformed_data
        st.session_state.comparison_summary = summary_df

        
        # === SAVE RESULTS ===
        st.download_button(
            label="💾 Download Comparison Results",
            data=summary_df.to_csv(index=False),
            file_name=f"transformation_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
        
        st.session_state.all_transformed_data = all_transformed_data
        st.session_state.comparison_summary = summary_df
