# helpers/evaluation.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from typing import Dict, List
import streamlit as st

def evaluate_transformation_metrics(
    df_raw: pd.DataFrame,
    df_transformed: pd.DataFrame,
    raw_cols: List[str],
    trans_cols: List[str],
) -> Dict[str, float]:
    raw_vals = df_raw[raw_cols].to_numpy().ravel()
    trans_vals = df_transformed[trans_cols].to_numpy().ravel()

    raw_vals = raw_vals[np.isfinite(raw_vals)]
    trans_vals = trans_vals[np.isfinite(trans_vals)]

    n_raw = min(5000, len(raw_vals))
    n_trans = min(5000, len(trans_vals))
    _, p_raw = stats.shapiro(raw_vals[:n_raw]) if n_raw >= 3 else (None, np.nan)
    _, p_trans = stats.shapiro(trans_vals[:n_trans]) if n_trans >= 3 else (None, np.nan)

    means_raw = df_raw[raw_cols].mean(axis=1)
    vars_raw = df_raw[raw_cols].var(axis=1)
    means_trans = df_transformed[trans_cols].mean(axis=1)
    vars_trans = df_transformed[trans_cols].var(axis=1)

    def safe_corr(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 3:
            return np.nan
        return float(np.corrcoef(a[mask], b[mask])[0, 1])

    corr_raw = safe_corr(means_raw, vars_raw)
    corr_trans = safe_corr(means_trans, vars_trans)

    return {
        "shapiro_raw": float(p_raw) if np.isfinite(p_raw) else np.nan,
        "shapiro_trans": float(p_trans) if np.isfinite(p_trans) else np.nan,
        "mean_var_corr_raw": corr_raw,
        "mean_var_corr_trans": corr_trans,
    }


@st.cache_data(show_spinner=False)
def cached_evaluate_transformation_metrics(
    df_raw: pd.DataFrame,
    df_transformed: pd.DataFrame,
    raw_cols: List[str],
    trans_cols: List[str],
    method: str,
    file_hash: str,
) -> Dict[str, float]:
    return evaluate_transformation_metrics(df_raw, df_transformed, raw_cols, trans_cols)


@st.cache_data(show_spinner=False)
def create_raw_row_figure(
    df_raw: pd.DataFrame,
    raw_cols: List[str],
    title: str,
    file_hash: str,
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Raw Intensities", "Q-Q Plot (Raw)", "Mean-Variance (Raw)"],
        horizontal_spacing=0.08,
    )

    raw_vals, means_raw, vars_raw = cached_raw_values(df_raw, raw_cols, file_hash)

def create_raw_row_figure(
    df_raw: pd.DataFrame,
    raw_cols: List[str],
    title: str = "Raw Data Diagnostics",
) -> go.Figure:
    """
    Single row (1×3) for raw:
    col1: raw intensities (+ mean + ±2σ region)
    col2: QQ raw
    col3: mean–variance raw
    """
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Raw Intensities",
            "Q-Q Plot (Raw)",
            "Mean-Variance (Raw)",
        ],
        horizontal_spacing=0.08,
    )

    # ---------- Col 1: raw distribution + mean + ±2σ ----------
    raw_vals = df_raw[raw_cols].to_numpy().ravel()
    raw_vals = raw_vals[np.isfinite(raw_vals)]

    if len(raw_vals) > 0:
        mu = float(np.mean(raw_vals))
        sigma = float(np.std(raw_vals))
        x0, x1 = mu - 2 * sigma, mu + 2 * sigma

        # Histogram
        hist, bin_edges = np.histogram(raw_vals, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]

        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=hist,
                width=bin_width * 0.9,
                marker=dict(color="#1f77b4", opacity=0.7),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Shaded ±2σ region
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor="#ffffff",
            opacity=0.15,
            line_width=0,
            row=1,
            col=1,
        )

        # Mean line
        fig.add_vline(
            x=mu,
            line_color="white",
            line_width=2,
            line_dash="dash",
            row=1,
            col=1,
        )

        # Text annotations
        fig.add_annotation(
            x=mu,
            y=max(hist) * 0.82,
            xref="x1",
            yref="y1",
            text=f"μ={mu:.2f}",
            showarrow=False,
            font=dict(color="white", size=10),
        )
        fig.add_annotation(
            x=x1,
            y=max(hist) * 0.1,
            xref="x1",
            yref="y1",
            text="±2σ",
            showarrow=False,
            font=dict(color="white", size=9),
        )

    fig.update_xaxes(title_text="Intensity", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)

    # ---------- Col 2: QQ raw ----------
    if len(raw_vals) >= 10:
        osm_raw, osr_raw = stats.probplot(raw_vals, dist="norm")[:2]
        theo_q_raw = osm_raw[0]
        ordered_raw = osm_raw[1]
        fig.add_trace(
            go.Scatter(
                x=theo_q_raw,
                y=ordered_raw,
                mode="markers",
                marker=dict(color="#1f77b4", size=3),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        min_q, max_q = theo_q_raw.min(), theo_q_raw.max()
        fig.add_trace(
            go.Scatter(
                x=[min_q, max_q],
                y=[min_q, max_q],
                mode="lines",
                line=dict(color="red", dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)

    # ---------- Col 3: mean–variance raw ----------
    means_raw = df_raw[raw_cols].mean(axis=1)
    vars_raw = df_raw[raw_cols].var(axis=1)
    fig.add_trace(
        go.Scatter(
            x=means_raw,
            y=vars_raw,
            mode="markers",
            marker=dict(color="#1f77b4", size=4, opacity=0.4),
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig.update_xaxes(title_text="Mean", row=1, col=3)
    fig.update_yaxes(title_text="Variance", row=1, col=3)

    fig.update_layout(
        height=350,
        title=title,
        font=dict(family="Arial", size=11),
    )
    return fig


def create_transformed_row_figure(
    df_transformed: pd.DataFrame,
    trans_cols: List[str],
    title: str,
) -> go.Figure:
    """
    Single row (1×3) for transformed:
    col1: transformed intensities (+ mean + ±2σ)
    col2: QQ transformed
    col3: mean–variance transformed
    """
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            f"{title} Intensities",
            "Q-Q Plot (Transformed)",
            "Mean-Variance (Transformed)",
        ],
        horizontal_spacing=0.08,
    )

    # ---------- Col 1: transformed distribution + mean + ±2σ ----------
    trans_vals = df_transformed[trans_cols].to_numpy().ravel()
    trans_vals = trans_vals[np.isfinite(trans_vals)]

    if len(trans_vals) > 0:
        mu = float(np.mean(trans_vals))
        sigma = float(np.std(trans_vals))
        x0, x1 = mu - 2 * sigma, mu + 2 * sigma

        hist, bin_edges = np.histogram(trans_vals, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]

        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=hist,
                width=bin_width * 0.9,
                marker=dict(color="#ff7f0e", opacity=0.7),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Shaded ±2σ
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor="#ffffff",
            opacity=0.15,
            line_width=0,
            row=1,
            col=1,
        )

        # Mean line
        fig.add_vline(
            x=mu,
            line_color="white",
            line_width=2,
            line_dash="dash",
            row=1,
            col=1,
        )

        # Annotations
        fig.add_annotation(
            x=mu,
            y=max(hist) * 1.05,
            xref="x1",
            yref="y1",
            text=f"μ={mu:.2f}",
            showarrow=False,
            font=dict(color="white", size=10),
        )
        fig.add_annotation(
            x=x1,
            y=max(hist) * 0.1,
            xref="x1",
            yref="y1",
            text="±2σ",
            showarrow=False,
            font=dict(color="white", size=9),
        )

    fig.update_xaxes(title_text="Transformed Intensity", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)

    # ---------- Col 2: QQ transformed ----------
    if len(trans_vals) >= 10:
        osm_t, osr_t = stats.probplot(trans_vals, dist="norm")[:2]
        theo_q_t = osm_t[0]
        ordered_t = osm_t[1]
        fig.add_trace(
            go.Scatter(
                x=theo_q_t,
                y=ordered_t,
                mode="markers",
                marker=dict(color="#ff7f0e", size=3),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        min_qt, max_qt = theo_q_t.min(), theo_q_t.max()
        fig.add_trace(
            go.Scatter(
                x=[min_qt, max_qt],
                y=[min_qt, max_qt],
                mode="lines",
                line=dict(color="red", dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)

    # ---------- Col 3: mean–variance transformed ----------
    means_trans = df_transformed[trans_cols].mean(axis=1)
    vars_trans = df_transformed[trans_cols].var(axis=1)
    fig.add_trace(
        go.Scatter(
            x=means_trans,
            y=vars_trans,
            mode="markers",
            marker=dict(color="#ffb74d", size=4, opacity=0.4),
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig.update_xaxes(title_text="Mean", row=1, col=3)
    fig.update_yaxes(title_text="Variance", row=1, col=3)

    fig.update_layout(
        height=350,
        title=f"Transformation: {title}",
        font=dict(family="Arial", size=11),
    )
    return fig


