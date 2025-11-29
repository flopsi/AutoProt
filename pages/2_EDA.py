import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from dataclasses import dataclass
from typing import List

from components import inject_custom_css, render_header, render_navigation, render_footer, COLORS

st.set_page_config(
    page_title="EDA | Thermo Fisher Scientific",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
render_header()


# -----------------------
# Data model
# -----------------------
@dataclass
class MSData:
    raw: pd.DataFrame
    raw_filled: pd.DataFrame
    missing_count: int
    numeric_cols: List[str]
    transforms: object  # TransformsCache


TF_CHART_COLORS = ["#262262", "#A6192E", "#EA7600", "#F1B434", "#B5BD00", "#9BD3DD"]


# -----------------------
# Utilities
# -----------------------
def parse_protein_group(pg_str: str) -> str:
    if pd.isna(pg_str):
        return "Unknown"
    return str(pg_str).split(";")[0].strip()


def extract_conditions(cols: list[str]) -> tuple[list[str], dict]:
    """Extract condition letters and build color map."""
    conditions = [c[0] if c and c[0].isalpha() else "X" for c in cols]
    cond_order = sorted(set(conditions))
    color_map = {cond: TF_CHART_COLORS[i % len(TF_CHART_COLORS)] for i, cond in enumerate(cond_order)}
    return conditions, color_map


def sort_columns_by_condition(cols: list[str]) -> list[str]:
    """Sort columns: A1, A2, A3, B1, B2, B3, ..."""
    def sort_key(col: str):
        if col and col[0].isalpha():
            head, tail = col[0], col[1:]
            return (head, int(tail) if tail.isdigit() else 0)
        return (col, 0)
    return sorted(cols, key=sort_key)


# -----------------------
# Chart creation (cached)
# -----------------------
@st.cache_data
def create_intensity_heatmap(df_json: str, index_col: str | None, numeric_cols: list[str]) -> go.Figure:
    df = pd.read_json(df_json)
    labels = [parse_protein_group(df[index_col].iloc[i]) for i in range(len(df))] if index_col and index_col in df.columns else [f"Row {i}" for i in range(len(df))]
    sorted_cols = sort_columns_by_condition(numeric_cols)
    z = df[sorted_cols].values

    fig = go.Figure(
        data=go.Heatmap(
            z=z, x=sorted_cols, y=labels, colorscale="Viridis", showscale=True,
            colorbar=dict(title="log2"),
            hovertemplate="Protein: %{y}<br>Sample: %{x}<br>log2: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Intensity distribution (log2 filled)", xaxis_title="Samples", yaxis_title="",
        height=500, yaxis=dict(tickfont=dict(size=8)), xaxis=dict(tickangle=45),
        plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Arial", color="#54585A"),
    )
    return fig


@st.cache_data
def create_missing_distribution_chart(mask_json: str, label: str) -> go.Figure:
    mask = pd.read_json(mask_json)
    missing_per_row = mask.sum(axis=1)
    total_missing = mask.sum().sum()
    max_missing = mask.shape[1]
    counts = [(missing_per_row == i).sum() / len(mask) * 100 for i in range(max_missing + 1)]

    fig = go.Figure(
        data=go.Bar(
            x=[str(i) for i in range(max_missing + 1)], y=counts, marker_color="#262262",
            hovertemplate="Missing values in row: %{x}<br>Percent: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Missing values per {label} (row-wise)", xaxis_title="Number of missing values",
        yaxis_title="% of total", height=350, plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#54585A"), bargap=0.2,
        annotations=[dict(text=f"Total missing entries: {total_missing:,}", xref="paper", yref="paper", x=1, y=1.1, showarrow=False, font=dict(size=10), xanchor="right")],
    )
    return fig


@st.cache_data
def create_violin_plot(df_json: str, numeric_cols: list[str]) -> go.Figure:
    df = pd.read_json(df_json)
    long_df = df[numeric_cols].melt(var_name="Sample", value_name="log2_value")
    long_df["Condition"] = long_df["Sample"].str.extract(r"^([A-Z])")
    long_df["Replicate"] = long_df["Sample"].str.extract(r"(\d+)$")
    long_df["CondRep"] = long_df["Condition"] + long_df["Replicate"]

    cond_order = sorted(long_df["Condition"].dropna().unique())
    x_order = []
    for cond in cond_order:
        reps = sorted(long_df.loc[long_df["Condition"] == cond, "Replicate"].dropna().unique(), key=lambda r: int(r))
        x_order.extend([f"{cond}{r}" for r in reps])

    fig = px.violin(long_df, x="CondRep", y="log2_value", color="Condition", box=True, points=False,
                    category_orders={"CondRep": x_order}, color_discrete_sequence=TF_CHART_COLORS)
    fig.update_traces(width=0.7, scalemode="count", jitter=0.2)
    fig.update_layout(
        title="Replicate distributions for all conditions (log2 filled)", xaxis_title="Condition & replicate",
        yaxis_title="log2(intensity)", violinmode="group", violingroupgap=0.15, violingap=0.05,
        plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Arial", color="#54585A"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


@st.cache_data
def create_pca_plot(df_json: str, numeric_cols: list[str]) -> go.Figure:
    df = pd.read_json(df_json)
    data = df[numeric_cols].T.values
    valid_cols = np.std(data, axis=0) > 0
    data_clean = data[:, valid_cols]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_clean)
    pca = PCA(n_components=min(3, len(numeric_cols)))
    pca_result = pca.fit_transform(data_scaled)

    sorted_cols = sort_columns_by_condition(numeric_cols)
    conditions, color_map = extract_conditions(sorted_cols)

    fig = go.Figure()
    for cond in sorted(set(conditions)):
        idx = [i for i, c in enumerate(conditions) if c == cond]
        fig.add_trace(
            go.Scatter(
                x=pca_result[idx, 0], y=pca_result[idx, 1], mode="markers+text",
                marker=dict(size=12, color=color_map[cond]), text=[sorted_cols[i] for i in idx],
                textposition="top center", name=f"Condition {cond}",
                hovertemplate="Sample: %{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="PCA - Sample clustering",
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)" if len(pca.explained_variance_ratio_) > 1 else "PC2",
        height=450, plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#54585A"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


@st.cache_data
def compute_normality_stats(values: np.ndarray) -> dict:
    clean = values[np.isfinite(values)]
    if len(clean) < 20:
        return {"kurtosis": np.nan, "skewness": np.nan, "W": np.nan, "p": np.nan}
    sample = np.random.choice(clean, min(5000, len(clean)), replace=False)
    try:
        W, p = stats.shapiro(sample)
    except Exception:
        W, p = np.nan, np.nan
    return {"kurtosis": stats.kurtosis(clean), "skewness": stats.skew(clean), "W": W, "p": p}


@st.cache_data
def analyze_transformations(df_json: str, numeric_cols: list[str]) -> pd.DataFrame:
    df = pd.read_json(df_json)
    all_vals = df[numeric_cols].values.flatten()
    all_vals = all_vals[np.isfinite(all_vals)]

    transforms = {
        "Raw (log2 filled)": all_vals,
        "Log10": np.log10(np.maximum(all_vals, 1)),
        "Square root": np.sqrt(all_vals),
        "Cube root": np.cbrt(all_vals),
    }

    results = [{"Transformation": name, **compute_normality_stats(vals)} for name, vals in transforms.items()]
    return pd.DataFrame(results)


@st.cache_data
def compute_permanova(df_json: str, numeric_cols: list[str], n_perm: int = 999) -> dict:
    df = pd.read_json(df_json)
    data = df[numeric_cols].T.values
    sorted_cols = sort_columns_by_condition(numeric_cols)
    conditions = np.array([c[0] if c and c[0].isalpha() else "X" for c in sorted_cols])

    if len(np.unique(conditions)) < 2:
        return {"F": np.nan, "p": np.nan, "R2": np.nan}

    dist_matrix = squareform(pdist(data, metric="euclidean"))
    n = len(conditions)

    def f_stat(dist_mat, groups):
        ss_total = np.sum(dist_mat**2) / (2 * n)
        ss_within = 0
        for g in np.unique(groups):
            mask = groups == g
            if np.sum(mask) > 1:
                ss_within += np.sum(dist_mat[np.ix_(mask, mask)]**2) / (2 * np.sum(mask))
        ss_between = ss_total - ss_within
        df_between = len(np.unique(groups)) - 1
        df_within = n - len(np.unique(groups))
        if df_within == 0 or ss_within == 0:
            return np.nan, np.nan
        F = (ss_between / df_between) / (ss_within / df_within)
        R2 = ss_between / ss_total if ss_total > 0 else 0
        return F, R2

    F_obs, R2
