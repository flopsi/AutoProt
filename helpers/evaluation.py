# helpers/evaluation.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from typing import Dict, List


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


def create_evaluation_figure(
    df_raw: pd.DataFrame,
    df_transformed: pd.DataFrame,
    raw_cols: List[str],
    trans_cols: List[str],
    title: str,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            "Raw Data Distribution",
            "Transformed Data Distribution",
            "Q-Q Plot (Raw)",
            "Q-Q Plot (Transformed)",
            "Mean-Variance (Raw)",
            "Mean-Variance (Transformed)",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # 1. Raw distribution
    raw_vals = df_raw[raw_cols].to_numpy().ravel()
    raw_vals = raw_vals[np.isfinite(raw_vals)]
    fig.add_trace(
        go.Histogram(x=raw_vals, nbinsx=50, marker_color="#1f77b4", opacity=0.7, showlegend=False),
        row=1, col=1,
    )

    # 2. Transformed distribution
    trans_vals = df_transformed[trans_cols].to_numpy().ravel()
    trans_vals = trans_vals[np.isfinite(trans_vals)]
    fig.add_trace(
        go.Histogram(x=trans_vals, nbinsx=50, marker_color="#ff7f0e", opacity=0.7, showlegend=False),
        row=1, col=2,
    )

    # 3. Q-Q raw
    if len(raw_vals) >= 10:
        osm, osr = stats.probplot(raw_vals, dist="norm")[:2]
        theo_q = osm[0]
        ordered_vals = osm[1]
        fig.add_trace(
            go.Scatter(x=theo_q, y=ordered_vals, mode="markers",
                       marker=dict(color="#1f77b4", size=3), showlegend=False),
            row=1, col=3,
        )
        min_q, max_q = theo_q.min(), theo_q.max()
        fig.add_trace(
            go.Scatter(x=[min_q, max_q], y=[min_q, max_q],
                       mode="lines", line=dict(color="red", dash="dash"), showlegend=False),
            row=1, col=3,
        )

    # 4. Q-Q transformed
    if len(trans_vals) >= 10:
        osm_t, osr_t = stats.probplot(trans_vals, dist="norm")[:2]
        theo_q_t = osm_t[0]
        ord_t = osm_t[1]
        fig.add_trace(
            go.Scatter(x=theo_q_t, y=ord_t, mode="markers",
                       marker=dict(color="#ff7f0e", size=3), showlegend=False),
            row=2, col=1,
        )
        min_qt, max_qt = theo_q_t.min(), theo_q_t.max()
        fig.add_trace(
            go.Scatter(x=[min_qt, max_qt], y=[min_qt, max_qt],
                       mode="lines", line=dict(color="red", dash="dash"), showlegend=False),
            row=2, col=1,
        )

    # 5. Mean–variance raw
    means_raw = df_raw[raw_cols].mean(axis=1)
    vars_raw = df_raw[raw_cols].var(axis=1)
    fig.add_trace(
        go.Scatter(x=means_raw, y=vars_raw, mode="markers",
                   marker=dict(color="#1f77b4", size=4, opacity=0.4), showlegend=False),
        row=2, col=2,
    )
    fig.update_xaxes(type="log", title_text="Mean", row=2, col=2)
    fig.update_yaxes(type="log", title_text="Variance", row=2, col=2)

    # 6. Mean–variance transformed
    means_trans = df_transformed[trans_cols].mean(axis=1)
    vars_trans = df_transformed[trans_cols].var(axis=1)
    fig.add_trace(
        go.Scatter(x=means_trans, y=vars_trans, mode="markers",
                   marker=dict(color="#ffb74d", size=4, opacity=0.4), showlegend=False),
        row=2, col=3,
    )
    fig.update_xaxes(title_text="Mean", row=2, col=3)
    fig.update_yaxes(title_text="Variance", row=2, col=3)

    fig.update_layout(
        height=800,
        title=f"Transformation Evaluation: {title}",
        font=dict(family="Arial", size=11),
    )
    return fig
