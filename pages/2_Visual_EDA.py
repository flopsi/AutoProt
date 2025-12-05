"""
pages/2_Visual_EDA.py
Clean 2-row layout: Raw | Transformed + Comparison Table
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from helpers.transforms import apply_transformation, TRANSFORM_NAMES, TRANSFORM_DESCRIPTIONS
from helpers.comparison import compare_transformations
from helpers.constants import get_theme

st.set_page_config(page_title="Visual EDA", layout="wide")

st.title("📊 Visual Exploratory Data Analysis")

# Load data
protein_data = st.session_state.get("protein_data")
if not protein_data:
    st.error("❌ No data loaded. Please upload data first.")
    st.stop()

st.success(f"✅ Loaded: {len(protein_data.raw):,} proteins × {len(protein_data.numeric_cols)} samples")

# ============================================================================
# CONTROLS
# ============================================================================

col1, col2 = st.columns([2, 1])
with col1:
    transform_options = list(TRANSFORM_NAMES.keys())[1:]  # Skip raw for dropdown
    selected_transform = st.selectbox(
        "Transformation", 
        options=transform_options,
        format_func=lambda x: TRANSFORM_NAMES[x],
        index=0  # log2 default
    )
    st.caption(TRANSFORM_DESCRIPTIONS.get(selected_transform, ""))
    
with col2:
    max_plots = st.slider("Max samples", 3, min(12, len(protein_data.numeric_cols)), 6)

# Run comparison table (always visible)
if st.checkbox("Show Comparison Table"):
    col3, col4 = st.columns(2)
    with col3:
        compare_transforms = st.multiselect(
            "Compare these", 
            options=transform_options,
            default=['log2', 'log10', 'sqrt', 'arcsinh', 'vst']
        )
    with col4:
        st.button("🔄 Refresh Comparison")

# ============================================================================
# ROW 1: RAW DATA PLOTS (LEFT)
# ROW 2: TRANSFORMED PLOTS (RIGHT)  
# ============================================================================

theme = get_theme(st.session_state.get("theme", "light"))

# Create 2-row layout
row1_col1, row1_col2 = st.columns([1, 1])
row2_col1, row2_col2 = st.columns([1, 1])

# === ROW 1: RAW DATA ===
with row1_col1:
    st.subheader("📈 Raw Data Distributions")
    
    raw_fig = make_subplots(rows=2, cols=3, subplot_titles=[""]*6)
    
    for i, col in enumerate(protein_data.numeric_cols[:max_plots]):
        row, pos = (i//3)+1, (i%3)+1
        values = protein_data.raw[col][protein_data.raw[col]>1.0].dropna()
        
        if len(values) > 0:
            raw_fig.add_trace(
                go.Histogram(x=values, nbinsx=40, opacity=0.75,
                           marker_color="#1f77b4", showlegend=False),
                row=row, col=pos
            )
            raw_fig.add_vline(values.mean(), line_dash="dash", line_color="red",
                            annotation_text=f"μ={values.mean():.1f}", row=row, col=pos)
    
    raw_fig.update_layout(height=400, showlegend=False, 
                         title="Raw Intensity Distributions")
    st.plotly_chart(raw_fig, use_container_width=True)

# === ROW 2: TRANSFORMED DATA ===
with row2_col2:
    st.subheader(f"🔄 {TRANSFORM_NAMES[selected_transform]} Distributions")
    
    # Apply transformation
    df_trans, trans_cols = apply_transformation(
        protein_data.raw, protein_data.numeric_cols[:max_plots], selected_transform
    )
    
    trans_fig = make_subplots(rows=2, cols=3, subplot_titles=[""]*6)
    
    for i, col in enumerate(trans_cols[:max_plots]):
        row, pos = (i//3)+1, (i%3)+1
        values = df_trans[col][df_trans[col]>1.0].dropna()
        
        if len(values) > 0:
            trans_fig.add_trace(
                go.Histogram(x=values, nbinsx=40, opacity=0.75,
                           marker_color="#ff7f0e", showlegend=False),
                row=row, col=pos
            )
            trans_fig.add_vline(values.mean(), line_dash="dash", line_color="darkred",
                              annotation_text=f"μ={values.mean():.1f}", row=row, col=pos)
    
    trans_fig.update_layout(height=400, showlegend=False,
                           title=f"{TRANSFORM_NAMES[selected_transform]} Distributions")
    st.plotly_chart(trans_fig, use_container_width=True)

# ============================================================================
# SUMMARY STATISTICS (BOTH)
# ============================================================================

with row1_col2:
    st.subheader("📊 Raw Statistics")
    raw_stats = []
    for col in protein_data.numeric_cols[:max_plots]:
        values = protein_data.raw[col][protein_data.raw[col]>1.0].dropna()
        if len(values)>0:
            raw_stats.append({
                'Sample': col, 'N': len(values), 'Mean': values.mean(),
                'Std': values.std(), 'CV%': values.std()/values.mean()*100
            })
    if raw_stats:
        pd.DataFrame(raw_stats).round(1).style.format({'CV%': '{:.1f}%'}).to_html()

with row2_col1:
    st.subheader(f"📊 {TRANSFORM_NAMES[selected_transform]} Statistics")
    trans_stats = []
    for col in trans_cols[:max_plots]:
        values = df_trans[col][df_trans[col]>1.0].dropna()
        if len(values)>0:
            trans_stats.append({
                'Sample': col.replace('_transformed',''), 'N': len(values),
                'Mean': values.mean(), 'Std': values.std(), 'CV%': values.std()/values.mean()*100
            })
    if trans_stats:
        pd.DataFrame(trans_stats).round(1)

st.divider()

# ============================================================================
# COMPARISON TABLE (BOTTOM)
# ============================================================================

st.subheader("⚖️ All Transformations Comparison")
if st.button("🔄 Run Full Comparison"):
    with st.spinner("Comparing all transformations..."):
        summary_df, _ = compare_transformations(
            protein_data.raw, protein_data.numeric_cols[:6],
            ['log2', 'log10', 'sqrt', 'arcsinh', 'vst', 'yeo-johnson']
        )
    
    if not summary_df.empty:
        # Add rank and sort
        summary_df['Rank'] = summary_df['combined_score'].rank(ascending=False).astype(int)
        summary_df = summary_df.sort_values('Rank')
        
        # Simple table
        display_cols = ['Rank', 'method', 'shapiro_p', 'mean_var_corr', 'n_significant', 'improvement']
        st.dataframe(
            summary_df[display_cols].round(4),
            use_container_width=True,
            column_config={
                "method": st.column_config.Column("Method", width="medium"),
                "shapiro_p": st.column_config.Column("Shapiro p", format="%.2e"),
                "Rank": st.column_config.Column("Rank", width="medium")
            }
        )
        
        # Best result
        best = summary_df.iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("🥇 Best Method", TRANSFORM_NAMES[best['method']])
        col2.metric("Shapiro p", f"{best['shapiro_p']:.2e}")
        col3.metric("DE Results", f"{best['n_significant']:,}")
        
        # Download
        st.download_button(
            "💾 Download Results",
            summary_df.to_csv(index=False),
            f"transformations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv"
        )
