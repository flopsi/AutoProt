import altair as alt
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict
from models.proteomics_data import ProteomicsDataset
from config.colors import ThermoFisherColors


def create_interactive_protein_analysis(protein_data: ProteomicsDataset):
    """
    Create interactive Altair chart with:
    - Top: Boxplots of log2 intensities for conditions A and B
    - Bottom: Stacked horizontal bar charts for protein counts by species
    """
    
    # ============================================================
    # PREPARE DATA FOR TOP PANEL (BOXPLOTS)
    # ============================================================
    
    # Get condition A and B data
    a_data = protein_data.get_condition_data('A')
    b_data = protein_data.get_condition_data('B')
    
    # Combine and create long-form DataFrame for boxplots
    intensity_data = []
    
    for idx in protein_data.raw_df.index:
        species = protein_data.species_map.get(idx, 'unknown')
        
        # Condition A values
        for col in a_data.columns:
            value = a_data.loc[idx, col]
            if pd.notna(value) and value > 0:
                intensity_data.append({
                    'Condition': 'Condition A',
                    'Log2_Intensity': np.log2(value),
                    'Species': species.capitalize(),
                    'Protein_ID': idx
                })
        
        # Condition B values
        for col in b_data.columns:
            value = b_data.loc[idx, col]
            if pd.notna(value) and value > 0:
                intensity_data.append({
                    'Condition': 'Condition B',
                    'Log2_Intensity': np.log2(value),
                    'Species': species.capitalize(),
                    'Protein_ID': idx
                })
    
    intensity_df = pd.DataFrame(intensity_data)
    
    # ============================================================
    # PREPARE DATA FOR BOTTOM PANEL (STACKED BAR CHARTS)
    # ============================================================
    
    # Count proteins by species and condition
    total_counts = protein_data.get_species_counts()
    
    a_detected = a_data.dropna(how='all').index
    a_counts = {sp: sum(1 for i in a_detected if protein_data.species_map.get(i) == sp)
               for sp in ['human', 'ecoli', 'yeast']}
    
    b_detected = b_data.dropna(how='all').index
    b_counts = {sp: sum(1 for i in b_detected if protein_data.species_map.get(i) == sp)
               for sp in ['human', 'ecoli', 'yeast']}
    
    # Create DataFrame for bar charts
    count_data = []
    for species in ['human', 'ecoli', 'yeast']:
        count_data.append({
            'Category': 'Total',
            'Species': species.capitalize(),
            'Count': total_counts.get(species, 0)
        })
        count_data.append({
            'Category': 'Condition A',
            'Species': species.capitalize(),
            'Count': a_counts[species]
        })
        count_data.append({
            'Category': 'Condition B',
            'Species': species.capitalize(),
            'Count': b_counts[species]
        })
    
    count_df = pd.DataFrame(count_data)
    
    # ============================================================
    # DEFINE COLOR SCALES
    # ============================================================
    
    species_color_scale = alt.Scale(
        domain=['Human', 'Ecoli', 'Yeast'],
        range=[
            ThermoFisherColors.NAVY,
            ThermoFisherColors.PURPLE,
            ThermoFisherColors.ORANGE
        ]
    )
    
    # ============================================================
    # CREATE SELECTIONS
    # ============================================================
    
    # Brush selection on top panel (boxplots)
    brush = alt.selection_interval(encodings=['x'])
    
    # Click selection on bottom panel (bar charts) - by species
    click = alt.selection_multi(fields=['Species'])
    
    # ============================================================
    # TOP PANEL: BOXPLOTS OF LOG2 INTENSITIES
    # ============================================================
    
    boxplots = alt.Chart(intensity_df).mark_boxplot(
        size=50,
        extent='min-max'
    ).encode(
        x=alt.X('Condition:N', title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Log2_Intensity:Q', title='Log2 Intensity', scale=alt.Scale(zero=False)),
        color=alt.condition(
            brush,
            alt.Color('Species:N', scale=species_color_scale, legend=None),
            alt.value('lightgray')
        ),
        tooltip=['Condition:N', 'Species:N', 'Log2_Intensity:Q']
    ).properties(
        width=600,
        height=300,
        title='Protein Intensity Distribution by Condition'
    ).add_selection(
        brush
    ).transform_filter(
        click
    )
    
    # ============================================================
    # BOTTOM PANEL: STACKED HORIZONTAL BAR CHARTS
    # ============================================================
    
    bars = alt.Chart(count_df).mark_bar().encode(
        x=alt.X('Count:Q', title='Protein Count'),
        y=alt.Y('Category:N', title=None, axis=alt.Axis(labelAngle=0)),
        color=alt.condition(
            click,
            alt.Color('Species:N', scale=species_color_scale, 
                     legend=alt.Legend(title='Species', orient='top', direction='horizontal')),
            alt.value('lightgray')
        ),
        order=alt.Order('Species:N', sort='ascending'),
        tooltip=['Category:N', 'Species:N', 'Count:Q']
    ).properties(
        width=600,
        height=150,
        title='Protein Counts by Species'
    ).add_selection(
        click
    ).transform_filter(
        brush
    )
    
    # ============================================================
    # COMBINE PANELS
    # ============================================================
    
    chart = alt.vconcat(
        boxplots,
        bars,
        title={
            "text": "Interactive Protein Analysis",
            "subtitle": "Select conditions (top) or species (bottom) to filter",
            "fontSize": 16,
            "fontWeight": 600
        }
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        labelFontSize=11,
        titleFontSize=12
    )
    
    return chart
