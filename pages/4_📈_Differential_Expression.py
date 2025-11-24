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

# Species colors (matching R script)
SPECIES_COLORS = {
    'human': '#199d76',      # Green
    'ecoli': '#7570b2',      # Purple
    'yeast': '#d85f02',      # Orange
    'celegans': '#8B0000',   # Dark red
    'other': '#7B7B7B'       # Gray
}

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
        
        # Calculate log2 fold-change for each protein
        a_mean = a_data.mean(axis=1)
        b_mean = b_data.mean(axis=1)
        log2fc = a_mean - b_mean  # Assuming data is already log2-transformed
        
        # Create dataframe with species annotation
        fc_data = pd.DataFrame({
            'log2fc': log2fc,
            'species': [species_map.get(idx, 'other') for idx in log2fc.index]
        }).dropna()
        
        # ============================================================
        # LOG2 FOLD-CHANGE DENSITY PLOT BY SPECIES
        # ============================================================
        
        st.markdown("---")
        st.markdown("### Log₂ Fold-Change Distribution by Species")
        
        fig_density = go.Figure()
        
        # Plot density for each species
        for species in ['human', 'ecoli', 'yeast', 'celegans']:
            species_data = fc_data[fc_data['species'] == species]['log2fc']
            
            if len(species_data) > 0:
                fig_density.add_trace(go.Histogram(
                    x=species_data,
                    name=species.capitalize(),
                    histnorm='probability density',
                    nbinsx=40,
                    marker=dict(
                        color=SPECIES_COLORS[species],
                        line=dict(color=SPECIES_COLORS[species], width=0.5)
                    ),
                    opacity=0.6,
                    hovertemplate=(
                        f'<b>{species.capitalize()}</b><br>' +
                        'Log₂ FC: %{x:.2f}<br>' +
                        'Density: %{y:.3f}<extra></extra>'
                    )
                ))
                
                # Add median line for each species
                median_fc = species_data.median()
                fig_density.add_vline(
                    x=median_fc,
                    line_dash="dash",
                    line_color=SPECIES_COLORS[species],
                    line_width=1.5,
                    opacity=0.7,
                    annotation=dict(
                        text=f"{species[:1].upper()}: {median_fc:.2f}",
                        yanchor="top",
                        font=dict(size=10, color=SPECIES_COLORS[species])
                    )
                )
        
        # Add zero reference line
        fig_density.add_vline(
            x=0,
            line_dash="solid",
            line_color='rgba(0,0,0,0.3)',
            line_width=1
        )
        
        fig_density.update_layout(
            title=f'{data_type} Log₂ Fold-Change Distribution (A - B) by Species',
            xaxis_title='Log₂ Fold-Change (A - B)',
            yaxis_title='Density',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", color=ThermoFisherColors.PRIMARY_GRAY),
            xaxis=dict(gridcolor='rgba(0,0,0,0.1)', zeroline=True, zerolinecolor='rgba(0,0,0,0.3)'),
            yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
            barmode='overlay',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            )
        )
        
        st.plotly_chart(fig_density, use_container_width=True)
        
        # Statistics table by species
        st.markdown("#### Statistics by Species")
        
        stats_data = []
        for species in ['human', 'ecoli', 'yeast', 'celegans']:
            species_fc = fc_data[fc_data['species'] == species]['log2fc']
            
            if len(species_fc) > 0:
                stats_data.append({
                    'Species': species.capitalize(),
                    'Count': len(species_fc),
                    'Median Log₂ FC': f"{species_fc.median():.2f}",
                    'Mean Log₂ FC': f"{species_fc.mean():.2f}",
                    'Std Dev': f"{species_fc.std():.2f}"
                })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
    
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
        species_map = current_data.species_map
        
        # Prepare condition data
        a_data, b_data = prepare_condition_data(quant_data, condition_mapping)
        
        # Calculate log2 fold-change
        a_mean = a_data.mean(axis=1)
        b_mean = b_data.mean(axis=1)
        log2fc = a_mean - b_mean
        
        # Create dataframe with species annotation
        fc_data = pd.DataFrame({
            'log2fc': log2fc,
            'species': [species_map.get(idx, 'other') for idx in log2fc.index]
        }).dropna()
        
        # ============================================================
        # LOG2 FOLD-CHANGE DENSITY PLOT BY SPECIES
        # ============================================================
        
        st.markdown("---")
        st.markdown("### Log₂ Fold-Change Distribution by Species")
        
        fig_density = go.Figure()
        
        for species in ['human', 'ecoli', 'yeast', 'celegans']:
            species_data = fc_data[fc_data['species'] == species]['log2fc']
            
            if len(species_data) > 0:
                fig_density.add_trace(go.Histogram(
                    x=species_data,
                    name=species.capitalize(),
                    histnorm='probability density',
                    nbinsx=40,
                    marker=dict(
                        color=SPECIES_COLORS[species],
                        line=dict(color=SPECIES_COLORS[species], width=0.5)
                    ),
                    opacity=0.6,
                    hovertemplate=(
                        f'<b>{species.capitalize()}</b><br>' +
                        'Log₂ FC: %{x:.2f}<br>' +
                        'Density: %{y:.3f}<extra></extra>'
                    )
                ))
                
                median_fc = species_data.median()
                fig_density.add_vline(
                    x=median_fc,
                    line_dash="dash",
                    line_color=SPECIES_COLORS[species],
                    line_width=1.5,
                    opacity=0.7,
                    annotation=dict(
                        text=f"{species[:1].upper()}: {median_fc:.2f}",
                        yanchor="top",
                        font=dict(size=10, color=SPECIES_COLORS[species])
                    )
                )
        
        fig_density.add_vline(
            x=0,
            line_dash="solid",
            line_color='rgba(0,0,0,0.3)',
            line_width=1
        )
        
        fig_density.update_layout(
            title=f'{data_type} Log₂ Fold-Change Distribution (A - B) by Species',
            xaxis_title='Log₂ Fold-Change (A - B)',
            yaxis_title='Density',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", color=ThermoFisherColors.PRIMARY_GRAY),
            xaxis=dict(gridcolor='rgba(0,0,0,0.1)', zeroline=True, zerolinecolor='rgba(0,0,0,0.3)'),
            yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
            barmode='overlay',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            )
        )
        
        st.plotly_chart(fig_density, use_container_width=True)
        
        st.markdown("#### Statistics by Species")
        
        stats_data = []
        for species in ['human', 'ecoli', 'yeast', 'celegans']:
            species_fc = fc_data[fc_data['species'] == species]['log2fc']
            
            if len(species_fc) > 0:
                stats_data.append({
                    'Species': species.capitalize(),
                    'Count': len(species_fc),
                    'Median Log₂ FC': f"{species_fc.median():.2f}",
                    'Mean Log₂ FC': f"{species_fc.mean():.2f}",
                    'Std Dev': f"{species_fc.std():.2f}"
                })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
    
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
