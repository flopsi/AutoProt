import plotly.graph_objects as go
from typing import Dict
from config.colors import ThermoFisherColors

def create_species_bar_chart(species_counts: Dict[str, int], title: str, show_percentages: bool = True) -> go.Figure:
    """
    Create a single stacked bar for one condition/total
    """
    species_order = ['human', 'ecoli', 'yeast']
    sorted_counts = {sp: species_counts.get(sp, 0) for sp in species_order}
    total = sum(sorted_counts.values())
    
    fig = go.Figure()
    
    # Create stacked bar with all species
    for species in species_order:
        count = sorted_counts[species]
        percentage = (count / total * 100) if total > 0 and count > 0 else 0
        label = f"{count} ({percentage:.1f}%)" if show_percentages and count > 0 else ""
        
        fig.add_trace(go.Bar(
            x=[title],
            y=[count],
            name=species.capitalize(),
            marker_color=ThermoFisherColors.SPECIES_COLORS[species],
            text=label,
            textposition='inside',
            textfont=dict(color='white', size=11, family='Arial'),
            hovertemplate=f"{species.capitalize()}: {count}<br>({percentage:.1f}%)<extra></extra>",
            showlegend=True
        ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=14, color=ThermoFisherColors.PRIMARY_GRAY, family='Arial'),
            x=0.5,
            xanchor='center'
        ),
        barmode='stack',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            title=None,
            font=dict(size=11, family='Arial')
        ),
        height=400,
        margin=dict(l=40, r=40, t=80, b=40),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent
        font=dict(family="Arial, sans-serif", color=ThermoFisherColors.PRIMARY_GRAY),
        xaxis_title=None,
        yaxis_title="Protein Count",
        yaxis=dict(
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=False
        ),
        xaxis=dict(
            showgrid=False,
            showticklabels=False
        )
    )
    
    return fig


def create_combined_species_chart(total_counts: Dict[str, int], 
                                  condition_a_counts: Dict[str, int],
                                  condition_b_counts: Dict[str, int],
                                  show_percentages: bool = True) -> go.Figure:
    """
    Create ONE chart with THREE stacked bars side-by-side (Total, Condition A, Condition B)
    """
    species_order = ['human', 'ecoli', 'yeast']
    categories = ['Total', 'Condition A', 'Condition B']
    
    # Prepare data for each species
    data_by_species = {
        'human': [
            total_counts.get('human', 0),
            condition_a_counts.get('human', 0),
            condition_b_counts.get('human', 0)
        ],
        'ecoli': [
            total_counts.get('ecoli', 0),
            condition_a_counts.get('ecoli', 0),
            condition_b_counts.get('ecoli', 0)
        ],
        'yeast': [
            total_counts.get('yeast', 0),
            condition_a_counts.get('yeast', 0),
            condition_b_counts.get('yeast', 0)
        ]
    }
    
    fig = go.Figure()
    
    # Add traces for each species
    for species in species_order:
        counts = data_by_species[species]
        
        # Calculate labels with percentages
        labels = []
        for i, cat in enumerate(categories):
            count = counts[i]
            if cat == 'Total':
                total = sum(data_by_species[sp][0] for sp in species_order)
            elif cat == 'Condition A':
                total = sum(data_by_species[sp][1] for sp in species_order)
            else:  # Condition B
                total = sum(data_by_species[sp][2] for sp in species_order)
            
            percentage = (count / total * 100) if total > 0 and count > 0 else 0
            label = f"{count}<br>({percentage:.1f}%)" if show_percentages and count > 0 else ""
            labels.append(label)
        
        fig.add_trace(go.Bar(
            name=species.capitalize(),
            x=categories,
            y=counts,
            marker_color=ThermoFisherColors.SPECIES_COLORS[species],
            text=labels,
            textposition='inside',
            textfont=dict(color='white', size=11, family='Arial'),
            hovertemplate='%{x}<br>' + species.capitalize() + ': %{y}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text="Protein Distribution by Species",
            font=dict(size=16, color=ThermoFisherColors.PRIMARY_GRAY, family='Arial', weight=600),
            x=0.5,
            xanchor='center'
        ),
        barmode='stack',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            title=None,
            font=dict(size=12, family='Arial'),
            bgcolor='rgba(0,0,0,0)'
        ),
        height=500,
        margin=dict(l=60, r=60, t=100, b=60),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        font=dict(family="Arial, sans-serif", color=ThermoFisherColors.PRIMARY_GRAY),
        xaxis=dict(
            title=None,
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(0,0,0,0.2)',
            tickfont=dict(size=12, family='Arial')
        ),
        yaxis=dict(
            title="Protein Count",
            titlefont=dict(size=13, family='Arial'),
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(0,0,0,0.2)'
        )
    )
    
    return fig
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
