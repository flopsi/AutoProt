# components/qc_plots.py
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def replicate_boxplots(df: pd.DataFrame, replicate_names: list, title="Replicate Boxplots") -> go.Figure:
    # df: rows=reps proteins, cols=reps
    melted = df[replicate_names].melt(var_name='Replicate', value_name='Intensity')
    fig = px.box(melted, x='Replicate', y='Intensity', points='outliers', title=title)
    fig.update_layout(showlegend=False)
    return fig

def cv_histogram(protein_df: pd.DataFrame, replicate_names: List[str], bins=20) -> go.Figure:
    cvs = []
    for idx, row in protein_df.iterrows():
        vals = [row[r] for r in replicate_names if pd.notnull(row[r])]
        if len(vals) > 1:
            cvs.append(np.std(vals, ddof=1) / np.mean(vals))
    fig = px.histogram(cvs, x=cvs, nbins=bins, title="CV Distribution across Proteins")
    fig.update_layout(xaxis_title="CV", yaxis_title="Frequency")
    return fig

def pca_scatter(replicate_df: pd.DataFrame, labels: List[str], color_by=None) -> go.Figure:
    # replicate_df: rows proteins, cols replicates
    pca = px.scatter_matrix(replicate_df, dimensions=replicate_df.columns, color=None)
    return pca
    # Note: A simpler PCA plot is shown in app.py for clarity; adapt as needed.

def missing_value_heatmap(df: pd.DataFrame) -> go.Figure:
    # df: rows proteins, cols replicates
    heat = (~df.isna()).astype(int)
    fig = px.imshow(heat, color_continuous_scale='Blues', origin='lower',
                    labels=dict(x="Replicate", y="Protein", color="Present"),
                    aspect="auto")
    fig.update_layout(title="Missing Value Heatmap")
    return fig

def rank_plot(df: pd.DataFrame, by_col: str, replicate_names: List[str]) -> go.Figure:
    # Simple rank plot: rank proteins by mean intensity across replicates
    means = df[replicate_names].mean(axis=1)
    ranks = means.rank(ascending=False)
    rank_df = pd.DataFrame({'Protein': df.index, 'MeanIntensity': means, 'Rank': ranks})
    fig = px.line(rank_df, x='Rank', y='MeanIntensity', hover_data=['Protein'], title="Dynamic Range / Rank Plot")
    fig.update_traces(mode='markers+lines')
    return fig
