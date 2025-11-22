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
import io
import zipfile


class LFQbenchVisualizer:
    """Generate all LFQbench visualizations"""
    
    def __init__(self, color_map: Optional[Dict[str, str]] = None):
        self.color_map = color_map or {
            'Human': '#199d76',
            'Yeast': '#d85f02',
            'E. coli': '#7570b2',
            'C.elegans': 'darkred'
        }
        # Store figures for export
        self.figures = {}
    
    def plot_density(self, df: pd.DataFrame, title: str = "Log2 Fold-Change Distribution") -> go.Figure:
        """
        Create SCIENTIFICALLY CORRECT density plot using KDE
        """
        fig = go.Figure()
        
        for species in df['Species'].unique():
            species_data = df[df['Species'] == species]['log2_fc'].dropna()
            
            if len(species_data) > 10:
                kde = gaussian_kde(species_data, bw_method='scott')
                x_range = np.linspace(species_data.min() - 1, species_data.max() + 1, 200)
                density = kde(x_range)
                
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
        
        self.figures['density'] = fig
        return fig
    
    def plot_volcano(self, df: pd.DataFrame, 
                    fc_threshold: float = 0.5,
                    alpha: float = 0.01,
                    title: str = "Volcano Plot") -> go.Figure:
        """
        Create volcano plot - FIXED
        """
        df = df.copy()
        
        # Handle p_adj values - FIXED
        df['p_adj'] = pd.to_numeric(df['p_adj'], errors='coerce')
        
        # Replace zeros and very small values
        df.loc[df['p_adj'] <= 0, 'p_adj'] = 1e-300
        df.loc[df['p_adj'].isna(), 'p_adj'] = 1.0
        
        # Calculate -log10(p_adj)
        df['-log10_pval'] = -np.log10(df['p_adj'])
        
        # Cap extremely high values for better visualization
        max_log_p = df['-log10_pval'].replace([np.inf, -np.inf], np.nan).max()
        if pd.isna(max_log_p) or max_log_p > 50:
            max_log_p = 50
        df.loc[df['-log10_pval'] > max_log_p, '-log10_pval'] = max_log_p
        
        fig = go.Figure()
        
        for species in df['Species'].unique():
            species_data = df[df['Species'] == species]
            
            # Separate significant and non-significant
            sig_data = species_data[species_data['is_significant'] == True]
            non_sig_data = species_data[species_data['is_significant'] == False]
            
            # Plot non-significant (smaller, transparent)
            if len(non_sig_data) > 0:
                fig.add_trace(go.Scatter(
                    x=non_sig_data['log2_fc'],
                    y=non_sig_data['-log10_pval'],
                    mode='markers',
                    name=f"{species} (NS)",
                    marker=dict(
                        color=self.color_map.get(species, 'gray'),
                        size=4,
                        opacity=0.3
                    ),
                    show
