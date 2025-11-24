import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from components.header import render_header
from config.colors import ThermoFisherColors
from utils.quality_plots import prepare_condition_data
from utils.statistical_analysis import limma_by_species

render_header()
st.title("Differential Expression Analysis")

protein_uploaded = st.session_state.get('protein_uploaded', False)
peptide_uploaded = st.session_state.get('peptide_uploaded', False)

if not protein_uploaded and not peptide_uploaded:
    st.warning("⚠️ No data loaded. Please upload protein or peptide data first.")
    if st.button("Go to Protein Upload", type="primary", use_container_width=True):
        st.switch_page("pages/1_📊_Protein_Upload.py")
    st.stop()

# Species colors matching R script
SPECIES_COLORS = {
    'human': '#199d76',      # Green/teal
    'yeast': '#d85f02',      # Orange
    'ecoli': '#7570b2',      # Purple
}

# Expected fold-changes
EXPECTED_FC = {
    'human': 0,
    'yeast': 1,
    'ecoli': -2
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
        
        st.markdown("---")
        st.markdown("### Step 1: Run limma Analysis per Species")
        
        # Run limma button
        if st.button("🔬 Run limma Analysis", type="primary", use_container_width=True):
            with st.spinner("Running limma for each species..."):
                try:
                    limma_results = limma_by_species(a_data, b_data, species_map)
                    st.session_state['limma_results'] = limma_results
                    st.success("✅ Analysis complete!")
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.stop()
        
        # Check if results exist
        if 'limma_results' not in st.session_state:
            st.info("👆 Click 'Run limma Analysis' to start")
            st.stop()
        
        limma_results = st.session_state['limma_results']
        
        # ============================================================
        # STEP 2: DENSITY PLOT
        # ============================================================
        
        st.markdown("---")
        st.markdown("### Step 2: Log₂ Fold-Change Distribution")
        
        fig = go.Figure()
        
        # Plot density curve for each species
        species_order = ['human', 'yeast', 'ecoli']
        
        for species in species_order:
            species_data = limma_results[limma_results['species'] == species]
            
            if len(species_data) > 10:
                log2fc = species_data['logFC'].values
                
                # Calculate KDE
                kde = gaussian_kde(log2fc, bw_method='scott')
                x_range = np.linspace(-3, 3, 300)
                density = kde(x_range)
                
                # Add density curve with fill
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=density,
                    mode='lines',
                    name=species.capitalize(),
                    line=dict(
                        color=SPECIES_COLORS[species],
                        width=0
                    ),
                    fill='tozeroy',
                    fillcolor=SPECIES_COLORS[species],
                    opacity=0.6,
                    hovertemplate=(
                        f'<b>{species.capitalize()}</b><br>' +
                        'Log₂(A/B): %{x:.2f}<br>' +
                        'Density: %{y:.2f}<extra></extra>'
                    )
                ))
                
                # Add expected fold-change line
                expected = EXPECTED_FC[species]
                fig.add_vline(
                    x=expected,
                    line_dash="dash",
                    line_color=SPECIES_COLORS[species],
                    line_width=2,
                    opacity=0.8
                )
        
        # Styling
        fig.update_layout(
            xaxis_title='Log₂(A/B)',
            yaxis_title='Density',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", color=ThermoFisherColors.PRIMARY_GRAY, size=12),
            xaxis=dict(
                showgrid=False,
                zeroline=True,
                zerolinecolor='rgba(0,0,0,0.2)',
                zerolinewidth=1,
                range=[-3, 3],
                tickmode='linear',
                dtick=1
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                range=[0, None]
            ),
            legend=dict(
                orientation='h',
                yanchor='top',
                y=0.98,
                xanchor='right',
                x=0.98,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1
            ),
            margin=dict(l=60, r=20, t=40, b=50),
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics table
        st.markdown("#### limma Results by Species")
        
        stats_data = []
        for species in species_order:
            species_data = limma_results[limma_results['species'] == species]
            
            if len(species_data) > 0:
                log2fc = species_data['logFC']
                expected = EXPECTED_FC[species]
                measured = log2fc.median()
                error = abs(measured - expected)
                
                # Count significant proteins (adj.P.Val < 0.05)
                sig_count = (species_data['adj.P.Val'] < 0.05).sum()
                
                stats_data.append({
                    'Species': species.capitalize(),
                    'Proteins': len(species_data),
                    'Significant (FDR<0.05)': sig_count,
                    'Expected Log₂ FC': f"{expected:.1f}",
                    'Measured Log₂ FC': f"{measured:.2f}",
                    'Error': f"{error:.2f}"
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
