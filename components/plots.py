import plotly.graph_objects as go
import pandas as pd
import numpy as np
def create_volcano_plot(df: pd.DataFrame, p_val_cutoff: float, fc_cutoff: float):
    """    Create an interactive volcano plot    Args:        df: DataFrame with protein data including significance        p_val_cutoff: -log10 p-value threshold        fc_cutoff: log2 fold change threshold    Returns:        Plotly figure object    """    # Separate data by significance    df_up = df[df['significance'] == 'UP']
    df_down = df[df['significance'] == 'DOWN']
    df_ns = df[df['significance'] == 'NS']
    fig = go.Figure()
    # Non-significant points (gray)    fig.add_trace(go.Scatter(
        x=df_ns['log2FoldChange'],
        y=df_ns['negLog10PValue'],
        mode='markers',
        name='Not Significant',
        marker=dict(
            color='#94a3b8',
            size=6,
            opacity=0.5,
            line=dict(width=0)
        ),
        text=df_ns['gene'],
        hovertemplate='<b>%{text}</b><br>' +                     'Log2FC: %{x:.3f}<br>' +                     '-Log10P: %{y:.3f}<br>' +                     '<extra></extra>',
        customdata=df_ns.index
    ))
    # Downregulated points (blue)    fig.add_trace(go.Scatter(
        x=df_down['log2FoldChange'],
        y=df_down['negLog10PValue'],
        mode='markers',
        name='Downregulated',
        marker=dict(
            color='#3b82f6',
            size=8,
            opacity=0.8,
            line=dict(color='white', width=1)
        ),
        text=df_down['gene'],
        hovertemplate='<b>%{text}</b><br>' +                     'Log2FC: %{x:.3f}<br>' +                     '-Log10P: %{y:.3f}<br>' +                     '<extra></extra>',
        customdata=df_down.index
    ))
    # Upregulated points (red)    fig.add_trace(go.Scatter(
        x=df_up['log2FoldChange'],
        y=df_up['negLog10PValue'],
        mode='markers',
        name='Upregulated',
        marker=dict(
            color='#ef4444',
            size=8,
            opacity=0.8,
            line=dict(color='white', width=1)
        ),
        text=df_up['gene'],
        hovertemplate='<b>%{text}</b><br>' +                     'Log2FC: %{x:.3f}<br>' +                     '-Log10P: %{y:.3f}<br>' +                     '<extra></extra>',
        customdata=df_up.index
    ))
    # Add threshold lines    # Vertical lines for FC cutoff    fig.add_vline(x=fc_cutoff, line_dash="dash", line_color="rgba(100, 116, 139, 0.5)",
                  annotation_text=f"FC={fc_cutoff}")
    fig.add_vline(x=-fc_cutoff, line_dash="dash", line_color="rgba(100, 116, 139, 0.5)",
                  annotation_text=f"FC={-fc_cutoff}")
    # Horizontal line for p-value cutoff    fig.add_hline(y=p_val_cutoff, line_dash="dash", line_color="rgba(100, 116, 139, 0.5)",
                  annotation_text=f"p={p_val_cutoff:.1f}")
    # Update layout    fig.update_layout(
        title={
            'text': 'Volcano Plot - Differential Expression',
            'font': {'size': 18, 'color': '#1e293b', 'family': 'Arial, sans-serif'}
        },
        xaxis_title='Log2 Fold Change',
        yaxis_title='-Log10 P-value',
        template='plotly_white',
        hovermode='closest',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.1)",
            borderwidth=1        ),
        plot_bgcolor='#f8fafc',
        paper_bgcolor='white',
        margin=dict(l=60, r=40, t=60, b=60)
    )
    fig.update_xaxis(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(203, 213, 225, 0.5)',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='rgba(100, 116, 139, 0.3)'    )
    fig.update_yaxis(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(203, 213, 225, 0.5)'    )
    return fig
