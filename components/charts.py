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
