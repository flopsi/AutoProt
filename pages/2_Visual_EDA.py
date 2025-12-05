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
        
        # === SUMMARY TABLE ===
        st.subheader("📋 Comparison Summary")
        
        # Style best transformation
        def highlight_best(row):
            if row.name == summary_df['combined_score'].idxmax():
                return ["background-color: #B5BD00; color: white"] * len(row)
            return [""]
        
        styled_summary = summary_df.style.apply(highlight_best, axis=1).format({
            'shapiro_p': '{:.2e}',
            'mean_var_corr': '{:.3f}'
        })
        
        st.dataframe(styled_summary, use_container_width=True)
        
        # === BEST TRANSFORMATION ===
        best_method, reason = find_best_transformation(summary_df)
        st.success(f"🏆 **Best: {TRANSFORM_NAMES[best_method]}** | Reason: {reason}")
        
        # === METRICS ===
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Best Shapiro p", f"{summary_df['shapiro_p'].max():.2e}")
        with col2:
            st.metric("Best Variance Stabil.", f"{summary_df['mean_var_corr'].abs().min():.3f}")
        with col3:
            st.metric("Most Significant", f"{summary_df['n_significant'].max():,}")
        with col4:
            st.metric("Transformations Tested", len(summary_df))
        
        # === TOP 3 RANKING ===
        st.subheader("🏅 Top 3 Transformations")
        top3 = summary_df.nlargest(3, 'combined_score')
        for _, row in top3.iterrows():
            st.metric(
                TRANSFORM_NAMES[row['method']],
                f"p={row['shapiro_p']:.2e} | DE={row['n_significant']:,}",
                delta=f"corr={row['mean_var_corr']:.3f}"
            )
        
        # === SAVE RESULTS ===
        st.download_button(
            label="💾 Download Comparison Results",
            data=summary_df.to_csv(index=False),
            file_name=f"transformation_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
        
        st.session_state.all_transformed_data = all_transformed_data
        st.session_state.comparison_summary = summary_df
