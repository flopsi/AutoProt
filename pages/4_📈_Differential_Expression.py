import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
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

# Species colors matching R script exactly
SPECIES_COLORS = {
    'human': '#199d76',      # colorhuman
    'ecoli': '#7570b2',      # colorecoli  
    'yeast': '#d85f02',      # coloryeast
    'celegans': '#8B0000'    # colorcelegans
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
        
        # Calculate log2 fold-change: log2(A/B) = log2(A) - log2(B)
        a_mean = a_data.mean(axis=1)
        b_mean = b_data.mean(axis=1)
        log2fc = a_mean - b_mean
        
        # Create dataframe with species annotation
        fc_data = pd.DataFrame({
            'log2fc': log2fc,
            'species': [species_map.get(idx, 'other') for idx in log2fc.index]
        }).dropna()
        
        # ============================================================
        # DENSITY PLOT (matching R f.p.density exactly)
        # ============================================================
        
        st.markdown("---")
        st.markdown("### Log₂ Fold-Change Distribution")
        
        fig = go.Figure()
        
        # Plot density curve for each species (matching R geom_density)
        for species in ['human', 'yeast', 'ecoli', 'celegans']:
            species_fc = fc_data[fc_data['species'] == species]['log2fc'].values
            
            if len(species_fc) > 10:  # Need enough points for KDE
                # Calculate KDE (kernel density estimation)
                kde = gaussian_kde(species_fc)
                x_range = np.linspace(species_fc.min(), species_fc.max(), 200)
                density = kde(x_range)
                
                # Add density curve
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=density,
                    mode='lines',
                    name=species.capitalize(),
                    line=dict(
                        color=SPECIES_COLORS[species],
                        width=2
                    ),
                    fill='tozeroy',
                    fillcolor=SPECIES_COLORS[species],
                    opacity=0.6,
                    hovertemplate=(
                        f'<b>{species.capitalize()}</b><br>' +
                        'Log₂ FC: %{x:.2f}<br>' +
                        'Density: %{y:.3f}<extra></extra>'
                    )
                ))
                
                # Add median line (matching R geom_vline)
                median_fc = np.median(species_fc)
                fig.add_vline(
                    x=median_fc,
                    line_dash="dash",
                    line_color=SPECIES_COLORS[species],
                    line_width=1.5,
                    opacity=0.9
                )
        
        # Styling to match R ggplot theme
        fig.update_layout(
            title='',
            xaxis_title='Log₂(A/B)',
            yaxis_title='Density',
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", color='black', size=11),
            xaxis=dict(
                showgrid=False,
                zeroline=True,
                zerolinecolor='rgba(0,0,0,0.3)',
                zerolinewidth=1,
                range=[-3.5, 3.5],  # FCmin to FCmax from R
                tickmode='linear',
                tick0=-3,
                dtick=1
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                range=[0, 7],  # Matching R limits
                tickmode='linear',
                tick0=0,
                dtick=1
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(255,255,255,0.8)'
            ),
            margin=dict(l=60, r=20, t=40, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics table by species
        st.markdown("#### Statistics by Species")
        
        stats_data = []
        for species in ['human', 'yeast', 'ecoli', 'celegans']:
            species_fc = fc_data[fc_data['species'] == species]['log2fc']
            
            if len(species_fc) > 0:
                stats_data.append({
                    'Species': species.capitalize(),
                    'Count': len(species_fc),
                    'Median': f"{species_fc.median():.2f}",
                    'Q1': f"{species_fc.quantile(0.25):.2f}",
                    'Q3': f"{species_fc.quantile(0.75):.2f}"
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
        st.info("🔬 Peptide differential expression analysis available after protein analysis.")
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
