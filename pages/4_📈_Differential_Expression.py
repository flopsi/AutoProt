import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from components.header import render_header
from config.colors import ThermoFisherColors
from utils.quality_plots import prepare_condition_data

render_header()
st.title("Differential Expression Analysis")

protein_uploaded = st.session_state.get('protein_uploaded', False)
peptide_uploaded = st.session_state.get('peptide_uploaded', False)

if not protein_uploaded and not peptide_uploaded:
    st.warning("⚠️ No data loaded. Please upload protein or peptide data first.")
    if st.button("Go to Protein Upload", type="primary", use_container_width=True):
        st.switch_page("pages/1_📊_Protein_Upload.py")
    st.stop()

# Data selection tabs
data_tab1, data_tab2 = st.tabs(["Protein Data", "Peptide Data"])

# ============================================================
# PROTEIN TAB
# ============================================================

with data_tab1:
    if protein_uploaded:
        current_data = st.session_state.protein_data
        data_type = "Protein"
        
        quant_data = current_data.quant_data
        condition_mapping = current_data.condition_mapping
        species_map = current_data.species_map
        
        # Prepare condition data
        a_data, b_data = prepare_condition_data(quant_data, condition_mapping)
        
        # Calculate log2 fold-change
        a_mean = a_data.mean(axis=1)
        b_mean = b_data.mean(axis=1)
        log2fc = a_mean - b_mean  # Assuming data is already log2-transformed
        
        # ============================================================
        # LOG2 FOLD-CHANGE DENSITY PLOT
        # ============================================================
        
        st.markdown("---")
        st.markdown("### Log₂ Fold-Change Distribution")
        
        # Remove NaN values
        valid_fc = log2fc.dropna()
        
        # Create density plot
        fig_density = go.Figure()
        
        # Add histogram (density)
        fig_density.add_trace(go.Histogram(
            x=valid_fc,
            histnorm='probability density',
            nbinsx=50,
            marker=dict(
                color='#9BD3DD',
                line=dict(color='#7570b2', width=1)
            ),
            opacity=0.7,
            name='Density',
            hovertemplate='Log₂ FC: %{x:.2f}<br>Density: %{y:.3f}<extra></extra>'
        ))
        
        # Add median line
        median_fc = valid_fc.median()
        fig_density.add_vline(
            x=median_fc,
            line_dash="dash",
            line_color='#E71316',
            line_width=2,
            annotation_text=f"Median: {median_fc:.2f}",
            annotation_position="top"
        )
        
        # Add zero line
        fig_density.add_vline(
            x=0,
            line_dash="solid",
            line_color='rgba(0,0,0,0.3)',
            line_width=1
        )
        
        fig_density.update_layout(
            title=f'{data_type} Log₂ Fold-Change Distribution (A - B)',
            xaxis_title='Log₂ Fold-Change (A - B)',
            yaxis_title='Density',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", color=ThermoFisherColors.PRIMARY_GRAY),
            xaxis=dict(gridcolor='rgba(0,0,0,0.1)', zeroline=True, zerolinecolor='rgba(0,0,0,0.3)'),
            yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
            showlegend=False
        )
        
        st.plotly_chart(fig_density, use_container_width=True)
        
        # Basic statistics
        st.markdown("#### Statistics")
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            st.metric("Median", f"{median_fc:.2f}")
        with stats_col2:
            st.metric("Mean", f"{valid_fc.mean():.2f}")
        with stats_col3:
            st.metric("Std Dev", f"{valid_fc.std():.2f}")
        with stats_col4:
            st.metric("Count", len(valid_fc))
    
    else:
        st.info("ℹ️ Protein data not loaded. Upload protein data to enable this analysis.")

# ============================================================
# PEPTIDE TAB
# ============================================================

with data_tab2:
    if peptide_uploaded:
        current_data = st.session_state.peptide_data
        data_type = "Peptide"
        
        quant_data = current_data.quant_data
        condition_mapping = current_data.condition_mapping
        
        # Prepare condition data
        a_data, b_data = prepare_condition_data(quant_data, condition_mapping)
        
        # Calculate log2 fold-change
        a_mean = a_data.mean(axis=1)
        b_mean = b_data.mean(axis=1)
        log2fc = a_mean - b_mean
        
        # ============================================================
        # LOG2 FOLD-CHANGE DENSITY PLOT
        # ============================================================
        
        st.markdown("---")
        st.markdown("### Log₂ Fold-Change Distribution")
        
        valid_fc = log2fc.dropna()
        
        fig_density = go.Figure()
        
        fig_density.add_trace(go.Histogram(
            x=valid_fc,
            histnorm='probability density',
            nbinsx=50,
            marker=dict(
                color='#9BD3DD',
                line=dict(color='#7570b2', width=1)
            ),
            opacity=0.7,
            name='Density',
            hovertemplate='Log₂ FC: %{x:.2f}<br>Density: %{y:.3f}<extra></extra>'
        ))
        
        median_fc = valid_fc.median()
        fig_density.add_vline(
            x=median_fc,
            line_dash="dash",
            line_color='#E71316',
            line_width=2,
            annotation_text=f"Median: {median_fc:.2f}",
            annotation_position="top"
        )
        
        fig_density.add_vline(
            x=0,
            line_dash="solid",
            line_color='rgba(0,0,0,0.3)',
            line_width=1
        )
        
        fig_density.update_layout(
            title=f'{data_type} Log₂ Fold-Change Distribution (A - B)',
            xaxis_title='Log₂ Fold-Change (A - B)',
            yaxis_title='Density',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", color=ThermoFisherColors.PRIMARY_GRAY),
            xaxis=dict(gridcolor='rgba(0,0,0,0.1)', zeroline=True, zerolinecolor='rgba(0,0,0,0.3)'),
            yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
            showlegend=False
        )
        
        st.plotly_chart(fig_density, use_container_width=True)
        
        st.markdown("#### Statistics")
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            st.metric("Median", f"{median_fc:.2f}")
        with stats_col2:
            st.metric("Mean", f"{valid_fc.mean():.2f}")
        with stats_col3:
            st.metric("Std Dev", f"{valid_fc.std():.2f}")
        with stats_col4:
            st.metric("Count", len(valid_fc))
    
    else:
        st.info("ℹ️ Peptide data not loaded. Upload peptide data to enable this analysis.")

# ============================================================
# NAVIGATION
# ============================================================

st.markdown("---")
st.markdown("### Navigation")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    if st.button("← Data Quality", use_container_width=True):
        st.switch_page("pages/3_✓_Data_Quality.py")

with nav_col2:
    if st.button("View Results Summary", use_container_width=True):
        st.session_state.upload_stage = 'summary'
        st.switch_page("pages/1_📊_Protein_Upload.py")

with nav_col3:
    if st.button("🔄 Start Over", type="primary", use_container_width=True):
        keys_to_delete = list(st.session_state.keys())
        for key in keys_to_delete:
            del st.session_state[key]
        
        st.session_state.protein_uploaded = False
        st.session_state.peptide_uploaded = False
        st.session_state.upload_stage = 'upload'
        
        st.switch_page("pages/1_📊_Protein_Upload.py")
