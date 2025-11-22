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
        Create density plot of log2 fold-changes by species
        """
        fig = go.Figure()
        
        for species in df['Species'].unique():
            species_data = df[df['Species'] == species]['log2_fc'].dropna()
            
            if len(species_data) > 0:
                fig.add_trace(go.Violin(
                    x=species_data,
                    name=species,
                    line_color=self.color_map.get(species, 'gray'),
                    fillcolor=self.color_map.get(species, 'gray'),
                    opacity=0.6,
                    showlegend=True
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Log2 Fold-Change",
            yaxis_title="Density",
            template="plotly_white",
            height=500,
            showlegend=True
        )
        
        return fig
    
    def plot_volcano(self, df: pd.DataFrame, 
                    fc_threshold: float = 0.5,
                    alpha: float = 0.01,
                    title: str = "Volcano Plot") -> go.Figure:
        """
        Create volcano plot (log2FC vs -log10 p-value)
        """
        df = df.copy()
        
        # Handle p_adj values
        df['p_adj'] = pd.to_numeric(df['p_adj'], errors='coerce')
        df.loc[df['p_adj'] == 0, 'p_adj'] = 1e-300
        df['-log10_pval'] = -np.log10(df['p_adj'])
        
        fig = go.Figure()
        
        for species in df['Species'].unique():
            species_data = df[df['Species'] == species]
            
            fig.add_trace(go.Scatter(
                x=species_data['log2_fc'],
                y=species_data['-log10_pval'],
                mode='markers',
                name=species,
                marker=dict(
                    color=self.color_map.get(species, 'gray'),
                    size=6,
                    opacity=0.6
                ),
                text=species_data.index,
                hovertemplate='<b>%{text}</b><br>Log2FC: %{x:.2f}<br>-log10(padj): %{y:.2f}<extra></extra>'
            ))
        
        # Add threshold lines
        fig.add_hline(y=-np.log10(alpha), line_dash="dot", line_color="gray", opacity=0.7)
        fig.add_vline(x=fc_threshold, line_dash="dot", line_color="gray", opacity=0.7)
        fig.add_vline(x=-fc_threshold, line_dash="dot", line_color="gray", opacity=0.7)
        
        fig.update_layout(
            title=title,
            xaxis_title="Log2 Fold-Change",
            yaxis_title="-log10(adjusted p-value)",
            template="plotly_white",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def plot_fc_boxplot(self, df: pd.DataFrame, 
                       title: str = "Fold-Change Accuracy") -> go.Figure:
        """
        Boxplot of |measured - expected| fold-changes by species
        Shows accuracy metric - FIXED
        """
        fig = go.Figure()
        
        has_data = False
        
        for species in df['Species'].unique():
            species_data = df[df['Species'] == species]
            
            # Ensure fc_deviation is numeric
            deviations = pd.to_numeric(species_data['fc_deviation'], errors='coerce').dropna()
            
            if len(deviations) > 0:
                has_data = True
                fig.add_trace(go.Box(
                    y=deviations,
                    name=species,
                    marker_color=self.color_map.get(species, 'gray'),
                    boxmean='sd'
                ))
        
        if not has_data:
            # Show empty state message
            fig.add_annotation(
                text="No fold-change deviation data available",
                showarrow=False,
                font=dict(size=16)
            )
        
        fig.update_layout(
            title=title,
            xaxis_title="Species",
            yaxis_title="|Expected - Measured| Log2 FC",
            template="plotly_white",
            height=500,
            showlegend=False
        )
        
        return fig
    
    def plot_cv_violin(self, df: pd.DataFrame,
                      title: str = "Coefficient of Variation") -> go.Figure:
        """
        Violin plot of CV distribution by species
        """
        cv_data = []
        for _, row in df.iterrows():
            exp_cv = pd.to_numeric(row['exp_cv'], errors='coerce')
            ctr_cv = pd.to_numeric(row['ctr_cv'], errors='coerce')
            
            if not pd.isna(exp_cv):
                cv_data.append({'Species': row['Species'], 'CV': exp_cv, 'Condition': 'Experimental'})
            if not pd.isna(ctr_cv):
                cv_data.append({'Species': row['Species'], 'CV': ctr_cv, 'Condition': 'Control'})
        
        cv_df = pd.DataFrame(cv_data)
        
        if len(cv_df) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No CV data available",
                showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        fig = px.violin(
            cv_df,
            x='Species',
            y='CV',
            color='Species',
            color_discrete_map=self.color_map,
            box=True,
            points=False
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Species",
            yaxis_title="CV (%)",
            template="plotly_white",
            height=500,
            showlegend=False,
            yaxis_range=[0, 30]
        )
        
        return fig
    
    def plot_ma(self, df: pd.DataFrame,
                title: str = "MA Plot") -> go.Figure:
        """
        MA plot: log2(mean intensity) vs log2(fold-change)
        """
        df = df.copy()
        
        # Ensure numeric columns
        df['exp_mean'] = pd.to_numeric(df['exp_mean'], errors='coerce')
        df['ctr_mean'] = pd.to_numeric(df['ctr_mean'], errors='coerce')
        df['log2_fc'] = pd.to_numeric(df['log2_fc'], errors='coerce')
        
        # Calculate log2 mean
        df['log2_mean'] = np.log2((df['exp_mean'] + df['ctr_mean']) / 2)
        
        fig = go.Figure()
        
        for species in df['Species'].unique():
            species_data = df[df['Species'] == species]
            
            fig.add_trace(go.Scatter(
                x=species_data['log2_mean'],
                y=species_data['log2_fc'],
                mode='markers',
                name=species,
                marker=dict(
                    color=self.color_map.get(species, 'gray'),
                    size=5,
                    opacity=0.5
                ),
                text=species_data.index,
                hovertemplate='<b>%{text}</b><br>Mean: %{x:.2f}<br>Log2FC: %{y:.2f}<extra></extra>'
            ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        fig.update_layout(
            title=title,
            xaxis_title="Log2(Mean Intensity)",
            yaxis_title="Log2 Fold-Change",
            template="plotly_white",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def plot_facet_scatter(self, df: pd.DataFrame,
                          title: str = "Control vs Experimental") -> go.Figure:
        """
        Faceted scatter plot: control mean vs log2FC
        """
        species_list = df['Species'].unique()
        n_species = len(species_list)
        
        fig = make_subplots(
            rows=1, cols=n_species,
            subplot_titles=list(species_list),
            horizontal_spacing=0.1
        )
        
        for i, species in enumerate(species_list, 1):
            species_data = df[df['Species'] == species]
            
            # Ensure numeric
            ctr_mean = pd.to_numeric(species_data['ctr_mean'], errors='coerce')
            log2_fc = pd.to_numeric(species_data['log2_fc'], errors='coerce')
            
            fig.add_trace(
                go.Scatter(
                    x=np.log2(ctr_mean),
                    y=log2_fc,
                    mode='markers',
                    marker=dict(
                        color=self.color_map.get(species, 'gray'),
                        size=5,
                        opacity=0.5
                    ),
                    name=species,
                    showlegend=False
                ),
                row=1, col=i
            )
            
            median_fc = log2_fc.median()
            if not pd.isna(median_fc):
                fig.add_hline(
                    y=median_fc,
                    line_dash="dash",
                    line_color=self.color_map.get(species, 'gray'),
                    row=1, col=i
                )
        
        fig.update_layout(
            title=title,
            template="plotly_white",
            height=400,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Log2(Control Mean)")
        fig.update_yaxes(title_text="Log2 Fold-Change")
        
        return fig
    
    def plot_pca(self, pca_result: np.ndarray,
                variance_explained: np.ndarray,
                sample_names: List[str],
                title: str = "PCA - Sample Overview") -> go.Figure:
        """
        PCA plot of samples
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            mode='markers+text',
            text=sample_names,
            textposition='top center',
            marker=dict(size=12, color='steelblue'),
            textfont=dict(size=10)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=f"PC1 ({variance_explained[0]*100:.1f}%)",
            yaxis_title=f"PC2 ({variance_explained[1]*100:.1f}%)" if len(variance_explained) > 1 else "PC2",
            template="plotly_white",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def plot_asymmetry_table(self, asymmetry_df: pd.DataFrame) -> go.Figure:
        """
        Display asymmetry factors as interactive table
        """
        if len(asymmetry_df) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No asymmetry data available",
                showarrow=False,
                font=dict(size=14)
            )
            return fig
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(asymmetry_df.columns),
                fill_color='lightgray',
                align='left',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=[asymmetry_df[col] for col in asymmetry_df.columns],
                fill_color='white',
                align='left',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(
            title="Asymmetry Factors by Species",
            height=300
        )
        
        return fig
    
    def plot_confusion_matrix(self, metrics: Dict[str, float]) -> go.Figure:
        """
        Heatmap of confusion matrix
        """
        confusion = np.array([
            [int(metrics['tn']), int(metrics['fp'])],
            [int(metrics['fn']), int(metrics['tp'])]
        ])
        
        fig = go.Figure(data=go.Heatmap(
            z=confusion,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'],
            text=confusion,
            texttemplate='%{text}',
            textfont={"size": 16},
            colorscale='Blues',
            showscale=False
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400,
            template="plotly_white"
        )
        
        return fig
    
    def create_summary_metrics_table(self, metrics: Dict[str, float]) -> go.Figure:
        """
        Create summary metrics display table
        """
        metrics_display = {
            'Metric': [
                'Sensitivity (%)',
                'Specificity (%)',
                'Empirical FDR (%)',
                'Accuracy',
                'Trueness',
                'CV Mean (%)',
                'CV Median (%)',
                'True Positives',
                'False Positives',
                'True Negatives',
                'False Negatives',
                'Total Proteins'
            ],
            'Value': [
                f"{float(metrics['sensitivity']):.2f}",
                f"{float(metrics['specificity']):.2f}",
                f"{float(metrics['de_fdr']):.2f}",
                f"{float(metrics['accuracy']):.3f}",
                f"{float(metrics['trueness']):.3f}",
                f"{float(metrics['cv_mean']):.2f}",
                f"{float(metrics['cv_median']):.2f}",
                str(int(metrics['tp'])),
                str(int(metrics['fp'])),
                str(int(metrics['tn'])),
                str(int(metrics['fn'])),
                str(int(metrics['n_proteins']))
            ]
        }
        
        df_metrics = pd.DataFrame(metrics_display)
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Metric</b>', '<b>Value</b>'],
                fill_color='steelblue',
                align='left',
                font=dict(size=13, color='white')
            ),
            cells=dict(
                values=[df_metrics['Metric'], df_metrics['Value']],
                fill_color='white',
                align='left',
                font=dict(size=12)
            )
        )])
        
        fig.update_layout(
            title="Performance Metrics Summary",
            height=500
        )
        
        return fig


def get_lfqbench_visualizer(color_map: Optional[Dict[str, str]] = None) -> LFQbenchVisualizer:
    """Get LFQbench visualizer instance"""
    return LFQbenchVisualizer(color_map)
