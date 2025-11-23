import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from components.header import render_header
from config.colors import ThermoFisherColors

render_header()
st.title("Data Quality Assessment")

protein_uploaded = st.session_state.get('protein_uploaded', False)
peptide_uploaded = st.session_state.get('peptide_uploaded', False)

if not protein_uploaded:
    st.warning("⚠️ No data loaded. Please upload protein data first.")
    if st.button("Go to Protein Upload", type="primary", use_container_width=True):
        st.switch_page("pages/1_📊_Protein_Upload.py")
    st.stop()

# Data selection tabs - ONLY difference is data source
data_tab1, data_tab2 = st.tabs(["Protein Data (default)", "Peptide Data"])

with data_tab1:
    # Use protein data
    current_data = st.session_state.protein_data
    data_type = "Protein"
    st.success(f"✓ Analyzing protein data: {current_data.n_proteins:,} proteins")

with data_tab2:
    if peptide_uploaded:
        # Use peptide data
        current_data = st.session_state.peptide_data
        data_type = "Peptide"
        st.success(f"✓ Analyzing peptide data: {current_data.n_rows:,} peptides")
    else:
        st.info("ℹ️ Peptide data not loaded. Upload peptide data to enable this view.")
        st.stop()

# ============================================================
# PREPARE DATA FOR BOXPLOTS
# ============================================================

# Get condition data
condition_mapping = current_data.condition_mapping
quant_data = current_data.quant_data

# Create figure
fig = go.Figure()

# Add boxplot for each sample
for col in quant_data.columns:
    condition = condition_mapping.get(col, col)
    condition_letter = condition[0]  # 'A' or 'B'
    
    # Get log10 values for this sample
    values = quant_data[col].dropna()
    log10_values = np.log10(values[values > 0])
    
    # Determine color based on condition
    color = '#E71316' if condition_letter == 'A' else '#9BD3DD'  # Red for A, Sky for B
    
    fig.add_trace(go.Box(
        y=log10_values,
        name=condition,
        marker_color=color,
        boxmean='sd',  # Show mean and standard deviation
        hovertemplate='<b>%{fullData.name}</b><br>Log₁₀ Intensity: %{y:.2f}<extra></extra>'
    ))

# Update layout
fig.update_layout(
    title=dict(
        text=f'{data_type} Intensity Distribution by Sample',
        font=dict(size=16, color=ThermoFisherColors.PRIMARY_GRAY, family='Arial', weight=600)
    ),
    yaxis_title='Log₁₀ Intensity',
    showlegend=False,
    height=500,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Arial, sans-serif", color=ThermoFisherColors.PRIMARY_GRAY),
    xaxis=dict(
        tickangle=-45,
        showgrid=False
    ),
    yaxis=dict(
        gridcolor='rgba(0,0,0,0.1)',
        showgrid=True,
        zeroline=False
    )
)

# ============================================================
# DISPLAY CHART
# ============================================================

st.markdown("---")
st.markdown("### Intensity Distribution Analysis")

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# NAVIGATION
# ============================================================

st.markdown("---")
st.markdown("### Navigation")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    if st.button("← View Results", use_container_width=True):
        st.session_state.upload_stage = 'summary'
        st.switch_page("pages/1_📊_Protein_Upload.py")

with nav_col2:
    if st.button("Upload Peptide Data", use_container_width=True):
        st.switch_page("pages/2_🔬_Peptide_Upload.py")

with nav_col3:
    if st.button("🔄 Start Over", type="primary", use_container_width=True):
        keys_to_delete = list(st.session_state.keys())
        for key in keys_to_delete:
            del st.session_state[key]
        st.switch_page("pages/1_📊_Protein_Upload.py")
