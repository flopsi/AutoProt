"""
pages/2_Visual_EDA.py
SIMPLE 6-panel intensity plots - no caching, no filtering issues
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from helpers.constants import get_theme

st.set_page_config(page_title="Visual EDA", layout="wide")

st.title("📊 Visual Exploratory Data Analysis")

# Load data
protein_data = st.session_state.get("protein_data")
if not protein_data:
    st.error("❌ No data loaded. Please upload data first.")
    st.stop()

st.success(f"✅ Loaded: {len(protein_data.raw)} proteins × {len(protein_data.numeric_cols)} samples")

# Simple controls
st.subheader("🎛️ Controls")
col1, col2 = st.columns(2)

with col1:
    transform = st.selectbox(
        "Transformation", 
        ["raw", "log2", "log10", "sqrt"],
        index=0
    )

with col2:
    max_cols = st.slider("Max samples to plot", 1, min(12, len(protein_data.numeric_cols)), 6)

st.divider()

# APPLY TRANSFORMATION (SIMPLE)
df = protein_data.raw.copy()
numeric_cols = protein_data.numeric_cols[:max_cols]

if transform == "log2":
    for col in numeric_cols:
        df[col] = np.log2(df[col].clip(lower=1.0))
elif transform == "log10":
    for col in numeric_cols:
        df[col] = np.log10(df[col].clip(lower=1.0))
elif transform == "sqrt":
    for col in numeric_cols:
        df[col] = np.sqrt(df[col].clip(lower=0))

# 6-PANEL INTENSITY PLOTS (WORKING VERSION)
st.subheader("📈 Sample Intensity Distributions")

theme = get_theme(st.session_state.get("theme", "light"))

# Create 2x3 grid
fig = make_subplots(
    rows=2, 
    cols=3,
    subplot_titles=[f"<b>{col}</b>" for col in numeric_cols[:6]],
    vertical_spacing=0.15,
    horizontal_spacing=0.10
)

# Plot each sample
for idx, col in enumerate(numeric_cols[:6]):
    row = (idx // 3) + 1
    col_pos = (idx % 3) + 1
    
    # Get values > 1.0 (standard proteomics filter)
    values = df[col][df[col] > 1.0].dropna()
    
    if len(values) == 0:
        continue
    
    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=40,
            opacity=0.75,
            marker_color=theme.get('primary', '#1f77b4'),
            name=col,
            showlegend=False
        ),
        row=row, col=col_pos
    )
    
    # Add mean line
    mean_val = values.mean()
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="red",
        line_width=2,
        row=row, col=col_pos
    )
    
    # Update axes
    fig.update_xaxes(title_text="Intensity", showgrid=True, row=row, col=col_pos)
    fig.update_yaxes(title_text="Count", showgrid=True, row=row, col=col_pos)

# Layout
fig.update_layout(
    height=650,
    showlegend=False,
    title_text=f"<b>Intensity Distributions ({transform.upper()})</b>",
    plot_bgcolor=theme.get('bg_primary', 'white'),
    paper_bgcolor=theme.get('paper_bg', 'white'),
    font=dict(family="Arial", size=11)
)

st.plotly_chart(fig, use_container_width=True)

# Simple summary stats
st.subheader("📊 Summary Statistics")
summary_data = []
for col in numeric_cols[:6]:
    values = df[col][df[col] > 1.0].dropna()
    if len(values) > 0:
        summary_data.append({
            'Sample': col,
            'N': len(values),
            'Mean': values.mean(),
            'Std': values.std(),
            'Min': values.min(),
            'Max': values.max()
        })

if summary_data:
    summary_df = pd.DataFrame(summary_data).round(2)
    st.dataframe(summary_df, use_container_width=True)
