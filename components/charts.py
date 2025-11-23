import streamlit as st
import pandas as pd
import altair as alt
from typing import Dict
from config.colors import ThermoFisherColors

def create_species_bar_chart(species_counts: Dict[str, int], title: str, show_percentages: bool = True):
    """
    Create species bar chart using Altair with custom colors
    """
    species_order = ['human', 'ecoli', 'yeast']
    sorted_counts = {sp: species_counts.get(sp, 0) for sp in species_order}
    total = sum(sorted_counts.values())
    
    # Create DataFrame
    data = []
    for species in species_order:
        count = sorted_counts[species]
        percentage = (count / total * 100) if total > 0 and count > 0 else 0
        data.append({
            'Species': species.capitalize(),
            'Count': count,
            'Percentage': f"{percentage:.1f}%",
            'Label': f"{count} ({percentage:.1f}%)" if show_percentages else str(count)
        })
    
    df = pd.DataFrame(data)
    
    # Color mapping
    color_scale = alt.Scale(
        domain=['Human', 'Ecoli', 'Yeast'],
        range=[
            ThermoFisherColors.NAVY,
            ThermoFisherColors.PURPLE,
            ThermoFisherColors.ORANGE
        ]
    )
    
    # Create Altair chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Species:N', title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Count:Q', title='Protein Count'),
        color=alt.Color('Species:N', scale=color_scale, legend=alt.Legend(
            orient='top',
            direction='horizontal',
            title=None
        )),
        tooltip=['Species:N', 'Count:Q', 'Percentage:N']
    ).properties(
        title=title,
        height=400
    )
    
    # Add text labels on bars
    text = chart.mark_text(
        align='center',
        baseline='middle',
        dy=-10,
        color='black',
        fontSize=12
    ).encode(
        text='Label:N'
    )
    
    # Display chart
    st.altair_chart(chart + text, use_container_width=True)
