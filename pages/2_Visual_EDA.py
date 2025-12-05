"""
pages/2_Visual_EDA.py
Clean implementation using working Colab transformations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from helpers.transforms import apply_transformation, TRANSFORM_NAMES, TRANSFORM_DESCRIPTIONS
from helpers.constants import get_theme

st.set_page_config(page_title="Visual EDA", layout="wide")

st.title("📊 Visual Exploratory Data Analysis")

# Load data
protein_data = st.session_state.get("protein_data")
if not protein_data:
    st.error("❌ No data loaded. Please upload data first.")
    st.stop()

st.success(f"✅ Loaded: {len(protein_data.raw):,} proteins × {len(protein_data.numeric_cols)} samples")

# Controls
st.subheader("🎛️ Controls")
col1, col2 = st.columns(2)

with col1:
    transform_options = list(TRANSFORM_NAMES.keys())
    selected_transform = st.selectbox(
        "Transformation",
        options=transform_options,
        format_func=lambda x: TRANSFORM_NAMES[x],
        index=1  # Default to log2
    )
    st.caption(TRANSFORM_DESCRIPTIONS.get(selected_transform, ""))

with col2:
    max_plots = st.slider("Max samples to plot", 1, min(12, len(protein_data.numeric_cols)), 6)

st.divider()

# APPLY TRANSFORMATION (Colab method)
st.info(f"🔄 Applying **{TRANSFORM_NAMES[selected_transform]}** transformation...")
df_transformed, transformed_cols = apply_transformation(
    df=protein_data.raw,
    numeric_cols=protein_data.numeric_cols[:max_plots],
    method=selected_transform
)

st.success(f"✅ Transformed {len(transformed_cols)} samples")

# 6-PANEL PLOTS
st.subheader("📈 Sample Intensity Distributions")

theme = get_theme(st.session_state.get("theme", "light"))

fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=[f"<b>{col.replace('_transformed', '')}</b>" for col in transformed_cols[:6]],
    vertical_spacing=0.15,
    horizontal_spacing=0.10
)

# Plot each sample
for idx, col in enumerate(transformed_cols[:6]):
    row = (idx // 3) + 1
    col_pos = (idx % 3) + 1
    
    # Filter values > 1.0 (proteomics standard)
    values = df_transformed[col][df_transformed[col] > 1.0].dropna()
    
    if len(values) == 0:
        continue
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=40,
            opacity=0.75,
            marker_color=theme.get('primary', '#1f77b4'),
            showlegend=False
        ),
        row=row, col=col_pos
    )
    
    # Mean line
    mean_val = values.mean()
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"μ={mean_val:.2f}",
        row=row, col=col_pos
    )
    
    # Axes
    fig.update_xaxes(title_text="Intensity", showgrid=True, row=row, col=col_pos)
    fig.update_yaxes(title_text="Count", showgrid=True, row=row, col=col_pos)

fig.update_layout(
    height=650,
    showlegend=False,
    title=f"<b>{TRANSFORM_NAMES[selected_transform]} Distributions</b>",
    plot_bgcolor=theme.get('bg_primary', 'white'),
    paper_bgcolor=theme.get('paper_bg', 'white')
)

st.plotly_chart(fig, use_container_width=True)

# Summary stats
st.subheader("📊 Summary Statistics")
summary_data = []
for col in transformed_cols[:6]:
    values = df_transformed[col][df_transformed[col] > 1.0].dropna()
    if len(values) > 0:
        summary_data.append({
            'Sample': col.replace('_transformed', ''),
            'N': len(values),
            'Mean': values.mean(),
            'Std': values.std(),
            'Min': values.min(),
            'Max': values.max()
        })

summary_df = pd.DataFrame(summary_data).round(2)
st.dataframe(summary_df, use_container_width=True)
