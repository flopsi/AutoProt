"""
LFQbench Visualization Module
Generates all plots for benchmark analysis.
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
        self.figures = {}
    
    def plot_density(self, df: pd.DataFrame, title: str = "Log2 Fold-Change Distribution") -> go.Figure:
        """Create density plot using KDE"""
        fig = go.Figure()
        
        for species in df['Species'].unique():
            species_data = df[df['Species'] == species]['log2_fc'].dropna()
            
            if len(species_data) > 10:
                kde = gaussian_kde(species_data, bw_method='scott')
                x_range = np.linspace(species_data.min() - 1, species_data.max() + 1, 200)
                density = kde(x_range)
                
                fig.add_trace(go.Scatter(
                    x=x_range, y=density, mode='lines', name=species,
                    line=dict(color=self.color_map.get(species, 'gray'), width=2),
                    fill='tozeroy', fillcolor=self.color_map.get(species, 'gray'), opacity=0.3
                ))
        
        fig.update_layout(
            title=title, xaxis_title="Log2 Fold-Change", yaxis_title="Density",
            template="plotly_white", height=500, showlegend=True, hovermode='x unified'
        )
        self.figures['density'] = fig
        return fig
    
    def plot_volcano(self, df: pd.DataFrame, fc_threshold: float = 0.5, alpha: float = 0.01, title: str = "Volcano Plot") -> go.Figure:
        """Create volcano plot with proper p-value handling"""
        df = df.copy()
        
        if 'p_adj' not in df.columns or 'log2_fc' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="Missing data", showarrow=False)
            return fig
        
        df['p_adj'] = pd.to_numeric(df['p_adj'], errors='coerce')
        df['log2_fc'] = pd.to_numeric(df['log2_fc'], errors='coerce')
        df = df.dropna(subset=['p_adj', 'log2_fc'])
        
        if len(df) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No valid data", showarrow=False)
            return fig
        
        df.loc[df['p_adj'] <= 0, 'p_adj'] = 1e-300
        df['-log10_pval'] = -np.log10(df['p_adj'])
        df['-log10_pval'] = df['-log10_pval'].replace([np.inf, -np.inf], 300)
        df.loc[df['-log10_pval'] > 50, '-log10_pval'] = 50
        
        fig = go.Figure()
        has_sig = 'is_significant' in df.columns
        
        for species in sorted(df['Species'].unique()):
            species_data = df[df['Species'] == species].copy()
            
            if has_sig:
                sig_data = species_data[species_data['is_significant'] == True]
                non_sig_data = species_data[species_data['is_significant'] == False]
                
                if len(non_sig_data) > 0:
                    fig.add_trace(go.Scatter(
                        x=non_sig_data['log2_fc'], y=non_sig_data['-log10_pval'],
                        mode='markers', name=species, marker=dict(
                            color=self.color_map.get(species, 'gray'), size=5, opacity=0.3
                        ), text=non_sig_data.index,
                        hovertemplate='<b>%{text}</b><br>Log2FC: %{x:.2f}<br>-log10(p): %{y:.2f}<extra></extra>'
                    ))
                
                if len(sig_data) > 0:
                    fig.add_trace(go.Scatter(
                        x=sig_data['log2_fc'], y=sig_data['-log10_pval'],
                        mode='markers', name=f"{species} (Sig)", marker=dict(
                            color=self.color_map.get(species, 'gray'), size=10, opacity=0.9,
                            symbol='diamond', line=dict(width=1, color='white')
                        ), text=sig_data.index,
                        hovertemplate='<b>%{text}</b><br>Log2FC: %{x:.2f}<br>-log10(p): %{y:.2f}<extra></extra>'
                    ))
            else:
                fig.add_trace(go.Scatter(
                    x=species_data['log2_fc'], y=species_data['-log10_pval'],
                    mode='markers', name=species, marker=dict(
                        color=self.color_map.get(species, 'gray'), size=6, opacity=0.6
                    ), text=species_data.index,
                    hovertemplate='<b>%{text}</b><br>Log2FC: %{x:.2f}<br>-log10(p): %{y:.2f}<extra></extra>'
                ))
        
        fig.add_hline(y=-np.log10(alpha), line_dash="dash", line_color="red", opacity=0.7,
                     annotation_text=f"α={alpha}", annotation_position="right")
        fig.add_vline(x=fc_threshold, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=-fc_threshold, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title=title, xaxis_title="Log2 Fold-Change", yaxis_title="-log10(adjusted p-value)",
            template="plotly_white", height=700, showlegend=True, hovermode='closest'
        )
        self.figures['volcano'] = fig
        return fig
    
    def plot_fc_boxplot(self, df: pd.DataFrame, title: str = "Fold-Change Accuracy") -> go.Figure:
        """Boxplot of fold-change deviations"""
        fig = go.Figure()
        has_data = False
        
        for species in df['Species'].unique():
            species_data = df[df['Species'] == species]
            deviations = pd.to_numeric(species_data['fc_deviation'], errors='coerce').dropna()
            
            if len(deviations) > 0:
                has_data = True
                fig.add_trace(go.Box(
                    y=deviations, name=species, marker_color=self.color_map.get(species, 'gray'), boxmean='sd'
                ))
        
        if not has_data:
            fig.add_annotation(text="No data available", showarrow=False)
        
        fig.update_layout(
            title=title, xaxis_title="Species", yaxis_title="|Expected - Measured| Log2 FC",
            template="plotly_white", height=500, showlegend=False
        )
        self.figures['fc_boxplot'] = fig
        return fig
    
    def plot_cv_violin(self, df: pd.DataFrame, title: str = "Coefficient of Variation") -> go.Figure:
        """Violin plot of CV distribution"""
        cv_data = []
        for _, row in df.iterrows():
            exp_cv = pd.to_numeric(row['exp_cv'], errors='coerce')
            ctr_cv = pd.to_numeric(row['ctr_cv'], errors='coerce')
            
            if not pd.isna(exp_cv):
                cv_data.append({'Species': row['Species'], 'CV': exp_cv})
            if not pd.isna(ctr_cv):
                cv_data.append({'Species': row['Species'], 'CV': ctr_cv})
        
        cv_df = pd.DataFrame(cv_data)
        
        if len(cv_df) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No CV data", showarrow=False)
            return fig
        
        fig = px.violin(cv_df, x='Species', y='CV', color='Species', color_discrete_map=self.color_map, box=True, points=False)
        fig.update_layout(
            title=title, xaxis_title="Species", yaxis_title="CV (%)",
            template="plotly_white", height=500, showlegend=False, yaxis_range=[0, 30]
        )
        self.figures['cv_violin'] = fig
        return fig
    
    def plot_ma(self, df: pd.DataFrame, title: str = "MA Plot") -> go.Figure:
        """MA plot"""
        df = df.copy()
        df['exp_mean'] = pd.to_numeric(df['exp_mean'], errors='coerce')
        df['ctr_mean'] = pd.to_numeric(df['ctr_mean'], errors='coerce')
        df['log2_fc'] = pd.to_numeric(df['log2_fc'], errors='coerce')
        df['log2_mean'] = np.log2((df['exp_mean'] + df['ctr_mean']) / 2)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['log2_mean', 'log2_fc'])
        
        fig = go.Figure()
        
        for species in df['Species'].unique():
            species_data = df[df['Species'] == species]
            
            if len(species_data) > 0:
                fig.add_trace(go.Scatter(
                    x=species_data['log2_mean'], y=species_data['log2_fc'], mode='markers', name=species,
                    marker=dict(color=self.color_map.get(species, 'gray'), size=5, opacity=0.5),
                    text=species_data.index,
                    hovertemplate='<b>%{text}</b><br>A: %{x:.2f}<br>M: %{y:.2f}<extra></extra>'
                ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        fig.update_layout(
            title=title, xaxis_title="A = Log2(Mean Intensity)", yaxis_title="M = Log2 Fold-Change",
            template="plotly_white", height=600, showlegend=True
        )
        self.figures['ma_plot'] = fig
        return fig
    
    def plot_facet_scatter(self, df: pd.DataFrame, title: str = "Control vs Experimental") -> go.Figure:
        """Faceted scatter plot"""
        species_list = df['Species'].unique()
        n_species = len(species_list)
        
        fig = make_subplots(rows=1, cols=n_species, subplot_titles=list(species_list), horizontal_spacing=0.1)
        
        for i, species in enumerate(species_list, 1):
            species_data = df[df['Species'] == species]
            ctr_mean = pd.to_numeric(species_data['ctr_mean'], errors='coerce')
            log2_fc = pd.to_numeric(species_data['log2_fc'], errors='coerce')
            
            fig.add_trace(
                go.Scatter(x=np.log2(ctr_mean), y=log2_fc, mode='markers',
                          marker=dict(color=self.color_map.get(species, 'gray'), size=5, opacity=0.5),
                          name=species, showlegend=False),
                row=1, col=i
            )
            
            median_fc = log2_fc.median()
            if not pd.isna(median_fc):
                fig.add_hline(y=median_fc, line_dash="dash", line_color=self.color_map.get(species, 'gray'), row=1, col=i)
        
        fig.update_layout(title=title, template="plotly_white", height=400, showlegend=False)
        fig.update_xaxes(title_text="Log2(Control Mean)")
        fig.update_yaxes(title_text="Log2 Fold-Change")
        self.figures['facet_scatter'] = fig
        return fig
    
    def plot_pca(self, pca_result: np.ndarray, variance_explained: np.ndarray, sample_names: List[str], title: str = "PCA") -> go.Figure:
        """PCA plot"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pca_result[:, 0], y=pca_result[:, 1], mode='markers+text', text=sample_names,
            textposition='top center', marker=dict(size=12, color='steelblue'), textfont=dict(size=10)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=f"PC1 ({variance_explained[0]*100:.1f}%)",
            yaxis_title=f"PC2 ({variance_explained[1]*100:.1f}%)" if len(variance_explained) > 1 else "PC2",
            template="plotly_white", height=600, showlegend=False
        )
        self.figures['pca'] = fig
        return fig
    
    def plot_asymmetry_table(self, asymmetry_df: pd.DataFrame) -> go.Figure:
        """Display asymmetry factors"""
        if len(asymmetry_df) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No asymmetry data", showarrow=False, font=dict(size=14))
            return fig
        
        display_df = asymmetry_df.copy()
        if 'Asymmetry Factor' in display_df.columns:
            display_df['Asymmetry Factor'] = display_df['Asymmetry Factor'].apply(
                lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A"
            )
        
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(display_df.columns), fill_color='lightgray', align='left', font=dict(size=12)),
            cells=dict(values=[display_df[col] for col in display_df.columns], fill_color='white', align='left', font=dict(size=11))
        )])
        
        fig.update_layout(title="Asymmetry Factors by Species", height=300)
        self.figures['asymmetry'] = fig
        return fig
    
    def plot_confusion_matrix(self, metrics: Dict[str, float]) -> go.Figure:
        """Confusion matrix heatmap"""
        confusion = np.array([
            [int(metrics['tn']), int(metrics['fp'])],
            [int(metrics['fn']), int(metrics['tp'])]
        ])
        
        fig = go.Figure(data=go.Heatmap(
            z=confusion, x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'], text=confusion,
            texttemplate='%{text}', textfont={"size": 16}, colorscale='Blues', showscale=False
        ))
        
        fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual", height=400, template="plotly_white")
        self.figures['confusion'] = fig
        return fig
    
    def create_summary_metrics_table(self, metrics: Dict[str, float]) -> go.Figure:
        """Create summary metrics table"""
        metric_names = [
            'Sensitivity (%)', 'Specificity (%)', 'Empirical FDR (%)',
            'Accuracy', 'Trueness', 'CV Mean (%)', 'CV Median (%)',
            'True Positives', 'False Positives', 'True Negatives', 'False Negatives', 'Total Proteins'
        ]
        
        metric_keys = ['sensitivity', 'specificity', 'de_fdr', 'accuracy', 'trueness', 'cv_mean', 'cv_median', 'tp', 'fp', 'tn', 'fn', 'n_proteins']
        
        values = []
        for key in metric_keys:
            val = metrics.get(key, 0)
            if key in ['sensitivity', 'specificity', 'de_fdr', 'cv_mean', 'cv_median']:
                values.append(f"{float(val):.2f}")
            elif key in ['accuracy', 'trueness']:
                values.append(f"{float(val):.3f}")
            else:
                values.append(f"{int(val)}")
        
        fig = go.Figure(data=[go.Table(
            columnwidth=[200, 100],
            header=dict(values=['<b>Metric</b>', '<b>Value</b>'], fill_color='steelblue', align=['left', 'right'], font=dict(size=14, color='white'), height=40),
            cells=dict(values=[metric_names, values], fill_color='white', align=['left', 'right'], font=dict(size=13, color='black'), height=35)
        )])
        
        fig.update_layout(title="Performance Metrics Summary", height=550, margin=dict(l=20, r=20, t=60, b=20))
        self.figures['metrics'] = fig
        return fig
    
    def export_all_figures(self) -> bytes:
        """Export all figures as HTML in ZIP"""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for name, fig in self.figures.items():
                html_str = fig.to_html(include_plotlyjs='cdn')
                zip_file.writestr(f"{name}.html", html_str)
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()


def get_lfqbench_visualizer(color_map: Optional[Dict[str, str]] = None) -> LFQbenchVisualizer:
    """Get LFQbench visualizer instance"""
    return LFQbenchVisualizer(color_map)
