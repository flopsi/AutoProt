import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler
from sklearn.decomposition import PCA
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
    original: pd.DataFrame        # after filtering + renaming
    filled: pd.DataFrame          # numeric 0/NaN/1 -> 1
    log2_filled: pd.DataFrame     # log2(filled)
    numeric_cols: List[str]       # renamed intensity columns (A1,A2,...)


# Thermo Fisher chart palette (style guide)
TF_CHART_COLORS = ["#262262", "#A6192E", "#EA7600", "#F1B434", "#B5BD00", "#9BD3DD"]


def parse_protein_group(pg_str: str) -> str:
    if pd.isna(pg_str):
        return "Unknown"
    return str(pg_str).split(";")[0].strip()


def sort_columns_by_condition(cols: list[str]) -> list[str]:
    def sort_key(col: str):
        if len(col) >= 1 and col[0].isalpha():
            head, tail = col[0], col[1:]
            return (head, int(tail) if tail.isdigit() else 0)
        return (col, 0)
    return sorted(cols, key=sort_key)


# ----------------------
# Transformations
# ----------------------
TRANSFORMATIONS = {
    "Raw (log2 filled)": lambda x: x,
    "Log10": lambda x: np.log10(np.maximum(x, 1)),
    "Square root": lambda x: np.sqrt(x),
    "Cube root": lambda x: np.cbrt(x),
    "Yeo-Johnson": lambda x: PowerTransformer(method="yeo-johnson", standardize=False)
    .fit_transform(x.reshape(-1, 1))
    .flatten()
    if len(x) > 1
    else x,
    "Quantile": lambda x: QuantileTransformer(output_distribution="normal", random_state=42)
    .fit_transform(x.reshape(-1, 1))
    .flatten()
    if len(x) > 1
    else x,
}


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
    return {
        "kurtosis": stats.kurtosis(clean),
        "skewness": stats.skew(clean),
        "W": W,
        "p": p,
    }


@st.cache_data
def analyze_transformations(df_json: str, numeric_cols: list[str]) -> pd.DataFrame:
    df = pd.read_json(df_json)
    all_values = df[numeric_cols].values.flatten()
    all_values = all_values[np.isfinite(all_values)]
    results = []
    for name, func in TRANSFORMATIONS.items():
        try:
            transformed = func(all_values.copy())
            stats_dict = compute_normality_stats(transformed)
        except Exception:
            stats_dict = {"kurtosis": np.nan, "skewness": np.nan, "W": np.nan, "p": np.nan}
        results.append(
            {
                "Transformation": name,
                "Kurtosis": stats_dict["kurtosis"],
                "Skewness": stats_dict["skewness"],
                "Shapiro W": stats_dict["W"],
                "Shapiro p": stats_dict["p"],
            }
        )
    return pd.DataFrame(results)


@st.cache_data
def create_intensity_heatmap(df_json: str, index_col: str | None, numeric_cols: list[str]) -> go.Figure:
    df = pd.read_json(df_json)
    if index_col and index_col in df.columns:
        labels = df[index_col].apply(parse_protein_group).tolist()
    else:
        labels = [f"Row {i}" for i in range(len(df))]

    sorted_cols = sort_columns_by_condition(numeric_cols)
    z = df[sorted_cols].values

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=sorted_cols,
            y=labels,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="log2"),
            hovertemplate="Protein: %{y}<br>Sample: %{x}<br>log2: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Intensity distribution (log2 filled)",
        xaxis_title="Samples",
        yaxis_title="",
        height=500,
        yaxis=dict(tickfont=dict(size=8)),
        xaxis=dict(tickangle=45),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#54585A"),
    )
    return fig


@st.cache_data
def create_missing_distribution_chart(mask_json: str, label: str) -> go.Figure:
    mask = pd.read_json(mask_json)
    missing_per_row = mask.sum(axis=1)
    total = mask.size
    total_missing = mask.sum().sum()

    max_missing = mask.shape[1]
    counts = [(missing_per_row == i).sum() / len(mask) * 100 for i in range(max_missing + 1)]

    fig = go.Figure(
        data=go.Bar(
            x=[str(i) for i in range(max_missing + 1)],
            y=counts,
            marker_color="#262262",  # NAVY
            hovertemplate="Missing values in row: %{x}<br>Percent: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Missing values per {label} (row-wise)",
        xaxis_title="Number of missing values",
        yaxis_title="% of total",
        height=350,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#54585A"),
        bargap=0.2,
        annotations=[
            dict(
                text=f"Total missing entries: {total_missing:,}",
                xref="paper",
                yref="paper",
                x=1,
                y=1.1,
                showarrow=False,
                font=dict(size=10),
                xanchor="right",
            )
        ],
    )
    return fig


@st.cache_data
def create_all_conditions_replicate_violin(model_json: str, numeric_cols: list[str]) -> go.Figure:
    df = pd.read_json(model_json)
    df_log2 = df[numeric_cols]

    long_df = df_log2.melt(var_name="Sample", value_name="log2_value")
    long_df["Condition"] = long_df["Sample"].str.extract(r"^([A-Z])")
    long_df["Replicate"] = long_df["Sample"].str.extract(r"^.*?(\d+)$")
    long_df["CondRep"] = long_df["Condition"] + long_df["Replicate"]

    cond_order = long_df["Condition"].dropna().unique()
    x_order = []
    for cond in cond_order:
        reps = (
            long_df.loc[long_df["Condition"] == cond, "Replicate"]
            .dropna()
            .unique()
        )
        reps = sorted(reps, key=lambda r: int(r))
        x_order.extend([f"{cond}{r}" for r in reps])

    fig = px.violin(
        long_df,
        x="CondRep",
        y="log2_value",
        color="Condition",
        box=True,
        points="outliers",
        category_orders={"CondRep": x_order},
        color_discrete_sequence=TF_CHART_COLORS,
    )

    fig.update_traces(width=0.7, scalemode="count", jitter=0.2)
    fig.update_layout(
        title="Replicate distributions for all conditions (log2 filled)",
        xaxis_title="Condition & replicate",
        yaxis_title="log2(intensity)",
        violinmode="group",
        violingroupgap=0.15,
        violingap=0.05,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#54585A"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


@st.cache_data
def compute_pca(df_json: str, numeric_cols: list[str]) -> tuple:
    df = pd.read_json(df_json)
    data = df[numeric_cols].values
    data_T = data.T
    valid_cols = np.std(data_T, axis=0) > 0
    data_clean = data_T[:, valid_cols]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_clean)
    pca = PCA(n_components=min(3, len(numeric_cols)))
    pca_result = pca.fit_transform(data_scaled)
    return pca_result, pca.explained_variance_ratio_, numeric_cols


@st.cache_data
def create_pca_plot(df_json: str, numeric_cols: list[str]) -> go.Figure:
    pca_result, var_explained, cols = compute_pca(df_json, numeric_cols)
    sorted_cols = sort_columns_by_condition(cols)
    conditions = [c[0] if len(c) >= 1 and c[0].isalpha() else "X" for c in sorted_cols]
    cond_order = sorted(set(conditions))
    color_map = {cond: TF_CHART_COLORS[i % len(TF_CHART_COLORS)] for i, cond in enumerate(cond_order)}

    fig = go.Figure()
    for cond in cond_order:
        idx = [i for i, c in enumerate(conditions) if c == cond]
        fig.add_trace(
            go.Scatter(
                x=pca_result[idx, 0],
                y=pca_result[idx, 1],
                mode="markers+text",
                marker=dict(size=12, color=color_map[cond]),
                text=[sorted_cols[i] for i in idx],
                textposition="top center",
                name=f"Condition {cond}",
                hovertemplate="Sample: %{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="PCA - Sample clustering",
        xaxis_title=f"PC1 ({var_explained[0]*100:.1f}% variance)",
        yaxis_title=f"PC2 ({var_explained[1]*100:.1f}% variance)" if len(var_explained) > 1 else "PC2",
        height=450,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#54585A"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


@st.cache_data
def compute_permanova(df_json: str, numeric_cols: list[str], n_perm: int = 999) -> dict:
    df = pd.read_json(df_json)
    data = df[numeric_cols].values
    data_T = data.T
    sorted_cols = sort_columns_by_condition(numeric_cols)
    conditions = np.array([c[0] if len(c) >= 1 and c[0].isalpha() else "X" for c in sorted_cols])
    unique_conds = np.unique(conditions)
    if len(unique_conds) < 2:
        return {"F": np.nan, "p": np.nan, "R2": np.nan}

    from scipy.spatial.distance import pdist, squareform

    dist_matrix = squareform(pdist(data_T, metric="euclidean"))
    n = len(conditions)

    def f_stat(dist_mat, groups):
        ss_total = np.sum(dist_mat**2) / (2 * n)
        ss_within = 0
        for g in np.unique(groups):
            mask = groups == g
            n_g = np.sum(mask)
            if n_g > 1:
                within = dist_mat[np.ix_(mask, mask)]
                ss_within += np.sum(within**2) / (2 * n_g)
        ss_between = ss_total - ss_within
        df_between = len(np.unique(groups)) - 1
        df_within = n - len(np.unique(groups))
        if df_within == 0 or ss_within == 0:
            return np.nan, np.nan
        F = (ss_between / df_between) / (ss_within / df_within)
        R2 = ss_between / ss_total
        return F, R2

    F_obs, R2 = f_stat(dist_matrix, conditions)
    f_perms = []
    for _ in range(n_perm):
        perm = np.random.permutation(conditions)
        F_perm, _ = f_stat(dist_matrix, perm)
        f_perms.append(F_perm)
    p_val = (np.sum(np.array(f_perms) >= F_obs) + 1) / (n_perm + 1)
    return {"F": F_obs, "p": p_val, "R2": R2}


st.markdown("## Exploratory data analysis")

protein_model: MSData | None = st.session_state.get("protein_model")
peptide_model: MSData | None = st.session_state.get("peptide_model")
protein_idx = st.session_state.get("protein_index_col")
peptide_idx = st.session_state.get("peptide_index_col")

if protein_model is None and peptide_model is None:
    st.warning("No data cached. Please upload data on the Data Upload page first.")
    render_navigation(back_page="pages/1_Data_Upload.py", next_page=None)
    render_footer()
    st.stop()

tab_protein, tab_peptide = st.tabs(["Protein data", "Peptide data"])


def render_eda(model: MSData | None, index_col: str | None, label: str):
    if model is None:
        st.info(f"No {label} data uploaded yet")
        return

    df_log2 = model.log2_filled
    numeric_cols = model.numeric_cols
    df_json = df_log2.to_json()
    mask = st.session_state.get(f"{label}_missing_mask")
    if mask is None:
        mask = pd.DataFrame(False, index=df_log2.index, columns=numeric_cols)
    mask_json = mask.to_json()

    st.caption(f"**{len(df_log2):,} {label}s** × **{len(numeric_cols)} samples**")

    # Heatmap + missing
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_heat = create_intensity_heatmap(df_json, index_col, numeric_cols)
        st.plotly_chart(fig_heat, use_container_width=True)
    with col2:
        fig_bar = create_missing_distribution_chart(mask_json, label)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # Violin: all conditions & replicates
    st.markdown("### Sample distributions (all conditions)")
    fig_violin = create_all_conditions_replicate_violin(df_json, numeric_cols)
    st.plotly_chart(fig_violin, use_container_width=True)

    st.markdown("---")

    # PCA + PERMANOVA
    st.markdown("### Variance analysis")
    st.caption("PCA clustering and PERMANOVA test for biological variance (conditions = first letter of sample).")

    col1, col2 = st.columns([2, 1])
    with col1:
        fig_pca = create_pca_plot(df_json, numeric_cols)
        st.plotly_chart(fig_pca, use_container_width=True)
    with col2:
        st.markdown("#### PERMANOVA results")
        permanova = compute_permanova(df_json, numeric_cols)
        st.metric("Pseudo-F statistic", f"{permanova['F']:.2f}" if not np.isnan(permanova["F"]) else "N/A")
        st.metric(
            "R² (variance explained)",
            f"{permanova['R2']*100:.1f}%" if not np.isnan(permanova["R2"]) else "N/A",
        )
        st.metric("p-value", f"{permanova['p']:.4f}" if not np.isnan(permanova["p"]) else "N/A")
        if permanova["p"] < 0.05:
            st.success("✓ Significant biological variance (p < 0.05)")
        elif not np.isnan(permanova["p"]):
            st.warning("No significant biological variance detected")

    st.markdown("---")

    # Normality
    st.markdown("### Normality analysis")
    st.caption("Testing which transformation best normalizes intensity distributions (on log2 filled baseline).")

    stats_df = analyze_transformations(df_json, numeric_cols)
    best_idx = stats_df["Shapiro W"].idxmax()
    best_transform = stats_df.loc[best_idx, "Transformation"]

    def highlight_best(row):
        if row["Transformation"] == best_transform:
            return ["background-color: #B5BD00; color: white"] * len(row)
        return [""] * len(row)

    styled_df = stats_df.style.apply(highlight_best, axis=1).format(
        {
            "Kurtosis": "{:.3f}",
            "Skewness": "{:.3f}",
            "Shapiro W": "{:.4f}",
            "Shapiro p": "{:.2e}",
        }
    )
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    st.success(
        f"Recommended transformation for downstream parametric tests: **{best_transform}** "
        f"(highest Shapiro W = {stats_df.loc[best_idx, 'Shapiro W']:.4f})"
    )


with tab_protein:
    render_eda(protein_model, protein_idx, "protein")

with tab_peptide:
    render_eda(peptide_model, peptide_idx, "peptide")

render_navigation(back_page="pages/1_Data_Upload.py", next_page=None)
render_footer()
