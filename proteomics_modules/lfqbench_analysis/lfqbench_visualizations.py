"""
LFQbench Visualization Module
Generates all plots for benchmark analysis following R script specifications.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
from scipy.stats import gaussian_kde


class LFQbenchVisualizer:
    """Generate all LFQbench visualizations"""
    
    def __init__(self, color_map: Optional[Dict[str, str]] = None):
        self.color_map = color_map or {
            'Human': '#199d76',
            'Yeast': '#d85f02',
            'E. coli': '#7570b2',
            'C.elegans': 'darkred'
        }
    
    def plot_density(self, df: pd.DataFrame, title: str = "Log2 Fold-Change Distribution") -> go.Figure:
        """
        Create SCIENTIFICALLY CORRECT density plot using KDE
        This matches R's geom_density() - NOT violin plots
        """
        fig = go.Figure()
        
        for species in df['Species'].unique():
            species_data = df[df['Species'] == species]['log2_fc'].dropna()
            
            if len(species_data) > 10:  # Need enough points for KDE
                # Calculate KDE (Kernel Density Estimation)
                kde = gaussian_kde(species_data, bw_method='scott')
                
                # Generate smooth x-axis
                x_range = np.linspace(species_data.min() - 1, species_data.max() + 1, 200)
                density = kde(x_range)
                
                # Plot density curve
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=density,
                    mode='lines',
                    name=species,
                    line=dict(
                        color=self.color_map.get(species, 'gray'),
                        width=2
                    ),
                    fill='tozeroy',
                    fillcolor=self.color_map.get(species, 'gray'),
                    opacity=0.3
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Log2 Fold-Change",
            yaxis_title="Density",
            template="plotly_white",
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_volcano(self, df: pd.DataFrame, 
                    fc_threshold: float = 0.5,
                    alpha: float = 0.01,
                    title: str = "Volcano Plot") -> go.Figure:
        """
        Create volcano plot (log2FC vs -log10 p-value)
        """
