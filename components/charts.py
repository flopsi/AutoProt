def create_species_bar_chart(species_counts: Dict[str, int], title: str, show_percentages: bool = True) -> go.Figure:
    species_order = ['human', 'ecoli', 'yeast']
    sorted_counts = {sp: species_counts.get(sp, 0) for sp in species_order}
    total = sum(sorted_counts.values())
    
    fig = go.Figure()
    
    # Create stacked bar with all species
    for species in species_order:
        count = sorted_counts[species]
        if count > 0:
            percentage = (count / total * 100) if total > 0 else 0
            label = f"{count} ({percentage:.1f}%)" if show_percentages else str(count)
            
            fig.add_trace(go.Bar(
                x=[title],
                y=[count],
                name=species.capitalize(),
                marker_color=ThermoFisherColors.SPECIES_COLORS[species],
                text=label,
                textposition='inside',
                textfont=dict(color='white', size=12),
                hovertemplate=f"{species.capitalize()}: {count}<br>({percentage:.1f}%)<extra></extra>"
            ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=ThermoFisherColors.PRIMARY_GRAY)),
        barmode='stack',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            title=None
        ),
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", color=ThermoFisherColors.PRIMARY_GRAY),
        xaxis_title=None,
        yaxis_title="Protein Count"
    )
    
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(gridcolor=ThermoFisherColors.LIGHT_GRAY)
    
    return fig
