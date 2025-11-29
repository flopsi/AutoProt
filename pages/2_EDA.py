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


@dataclass
class TransformsCache:
    log2: pd.DataFrame
    log10: pd.DataFrame
    sqrt: pd.DataFrame
    cbrt: pd.DataFrame
    yeo_johnson: pd.DataFrame
    quantile: pd.DataFrame
    condition_wise_cvs: pd.DataFrame


@dataclass
class MSData:
    raw: pd.DataFrame
    raw_filled: pd.DataFrame
    missing_count: int
    numeric_cols: List[str]
    transforms: TransformsCache


TF_CHART_COLORS = ["#262262", "#A6192E", "#EA7600", "#F1B434", "#B5BD00", "#9BD3DD"]


def parse_protein_group(pg_str: str) -> str:
    if pd.isna(pg_str):
        return "Unknown"
    return str(pg_str).split(";")[0].strip()


def extract_conditions(cols: list[str]) -> tuple[list[str], dict]:
    conditions = [c[0] if c and c[0].isalpha() else "X" for c in cols]
    cond_order = sorted(set(conditions))
    color_map = {cond: TF_CHART_COLORS[i % len(TF_CHART_COLORS)] for i, cond in enumerate(cond_order)}
    return conditions, color_map


def sort_columns_by_condition(cols: list[str]) -> list[str]:
    def sort_key(col: str):
        if col and col[0].isalpha():
            head, tail = col[0], col[1:]
            return (head, int(tail) if tail.isdigit() else 0)
        return (col, 0)
    return sorted(cols, key=sort_key)


@st.cache_data
def create_intensity_heatmap(df_log2: pd.DataFrame, index_col: str | None, numeric_cols: list[str]) -> go.Figure:
    df = df_log2.copy()
    if index_col and index_col in df.columns:
        labels = df[index_col].apply(parse_protein_group).tolist()
    else:
        labels = [f"Row {i}" for i in range(len(df))]

    sorted_cols = sort_columns_by_condition(numeric_cols)
    
    # Limit to top 100 by variance
    if len(df) > 100:
        variance = df[sorted_cols].var(axis=1)
        top_idx = variance.nlargest(100).index
        df = df.loc[top_idx]
        labels = [labels[i] for i in range(len(top_idx))]
    
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
        title="Intensity distribution (log2 filled, top 100 by variance)",
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
def create_missing_distribution_chart(mask: pd.DataFrame, label: str) -> go.Figure:
    missing_per_row = mask.sum(axis=1)
    total_missing = mask.sum().sum()
    max_missing = mask.shape[1]
    counts = [(missing_per_row == i).sum() / len(mask) * 100 for i in range(max_missing + 1)]

    fig = go.Figure(
        data=go.Bar(
            x=[str(i) for i in range(max_missing + 1)],
            y=counts,
            marker_color="#262262",
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
def create_violin_plot(df_log2: pd.DataFrame, numeric_cols: list[str]) -> go.Figure:
    long_df = df_log2[numeric_cols].melt(var_name="Sample", value_name="log2_value")
    long_df["Condition"] = long_df["Sample"].str.extract(r"^([A-Z])")
    long_df["Replicate"] = long_df["Sample"].str.extract(r"(\d+)$")
    long_df["CondRep"] = long_df["Condition"] + long_df["Replicate"]

    cond_order = sorted(long_df["Condition"].dropna().unique())
    x_order = []
    for cond in cond_order:
        reps = sorted(
            long_df.loc[long_df["Condition"] == cond, "Replicate"].dropna().unique(),
            key=lambda r: int(r),
        )
        x_order.extend([f"{cond}{r}" for r in reps])

    fig = px.violin(
        long_df,
        x="CondRep",
        y="log2_value",
        color="Condition",
        box=True,
        points=False,
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


def create_pca_plot(df_log2: pd.DataFrame, numeric_cols: list[str], title_suffix: str = "") -> go.Figure | None:
    df = df_log2[numeric_cols]
    
    if len(df) < 3:
        return None
    
    data = df.T.values  # samples × features
    valid_cols = np.std(data, axis=0) > 0
    data_clean = data[:, valid_cols]

    if data_clean.shape[1] < 2:
        return None

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_clean)
    pca = PCA(n_components=min(3, data_scaled.shape[0]))
    pca_result = pca.fit_transform(data_scaled)

    sorted_cols = sort_columns_by_condition(numeric_cols)
    conditions, color_map = extract_conditions(sorted_cols)

    fig = go.Figure()
    for cond in sorted(set(conditions)):
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
        title=f"PCA{title_suffix}",
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)" if len(pca.explained_variance_ratio_) > 1 else "PC2",
        height=400,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#54585A"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def permanova_core(df_log2: pd.DataFrame, numeric_cols: list[str]) -> dict:
    if len(df_log2) < 3:
        return {"F": np.nan, "p": np.nan, "R2": np.nan}
    
    data = df_log2[numeric_cols].T.values
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

    F_obs, R2 = f_stat(dist_matrix, conditions)
    f_perms = [f_stat(dist_matrix, np.random.permutation(conditions))[0] for _ in range(999)]
    p_val = (np.sum(np.array(f_perms) >= F_obs) + 1) / (999 + 1)
    return {"F": F_obs, "p": p_val, "R2": R2}


@st.cache_data
def analyze_transformations(df_log2: pd.DataFrame, numeric_cols: list[str], transforms: TransformsCache) -> pd.DataFrame:
    transforms_dict = {
        "Raw (log2 filled)": df_log2[numeric_cols].values.flatten(),
        "Log10": transforms.log10[numeric_cols].values.flatten(),
        "Square root": transforms.sqrt[numeric_cols].values.flatten(),
        "Cube root": transforms.cbrt[numeric_cols].values.flatten(),
        "Yeo-Johnson": transforms.yeo_johnson[numeric_cols].values.flatten(),
        "Quantile": transforms.quantile[numeric_cols].values.flatten(),
    }

    results = []
    for name, vals in transforms_dict.items():
        clean = vals[np.isfinite(vals)]
        if len(clean) >= 20:
            sample = np.random.choice(clean, min(5000, len(clean)), replace=False)
            try:
                W, p = stats.shapiro(sample)
            except Exception:
                W, p = np.nan, np.nan
            results.append({
                "Transformation": name,
                "Kurtosis": stats.kurtosis(clean),
                "Skewness": stats.skew(clean),
                "Shapiro W": W,
                "Shapiro p": p,
            })

    return pd.DataFrame(results)


st.markdown("## Exploratory data analysis")

protein_model: MSData | None = st.session_state.get("protein_model")
peptide_model: MSData | None = st.session_state.get("peptide_model")
protein_idx = st.session_state.get("protein_index_col")
peptide_idx = st.session_state.get("peptide_index_col")
protein_species_col = st.session_state.get("protein_species_col")
peptide_species_col = st.session_state.get("peptide_species_col")

if protein_model is None and peptide_model is None:
    st.warning("No data cached. Please upload data on the Data Upload page first.")
    render_navigation(back_page="pages/1_Data_Upload.py", next_page=None)
    render_footer()
    st.stop()

tab_protein, tab_peptide = st.tabs(["Protein data", "Peptide data"])


def render_eda(model: MSData | None, index_col: str | None, species_col: str | None, label: str):
    if model is None:
        st.info(f"No {label} data uploaded yet")
        return

    numeric_cols = model.numeric_cols
    df_log2 = model.transforms.log2[numeric_cols]

    mask = st.session_state.get(f"{label}_missing_mask")
    if mask is None:
        mask = pd.DataFrame(False, index=df_log2.index, columns=numeric_cols)

    st.caption(
        f"**{len(df_log2):,} {label}s** × **{len(numeric_cols)} samples** | "
        f"**{model.missing_count:,} missing cells** (NaN/0/1)"
    )

    # Row 1: Heatmap + missing
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_heat = create_intensity_heatmap(df_log2, index_col, numeric_cols)
        st.plotly_chart(fig_heat, width="stretch", key=f"heatmap_{label}")
    with col2:
        fig_bar = create_missing_distribution_chart(mask, label)
        st.plotly_chart(fig_bar, width="stretch", key=f"missing_{label}")

    st.markdown("---")

    # Row 2: Violin
    st.markdown("### Sample distributions (all conditions)")
    fig_violin = create_violin_plot(df_log2, numeric_cols)
    st.plotly_chart(fig_violin, width="stretch", key=f"violin_{label}")

    st.markdown("---")

    # Row 3: Variance analysis with multi-species PCA
    st.markdown("### Variance analysis")

    species_dist = None
    if species_col and species_col in model.raw.columns:
        species_counts = model.raw[species_col].value_counts()
        if len(species_counts) >= 2:
            species_dist = {
                "species": species_counts.index.tolist(),
                "most_frequent": species_counts.index[0],
                "others": species_counts.index[1:].tolist(),
            }

    if species_dist:
        st.caption(
            f"Species detected: {', '.join(species_dist['species'])}. "
            "Showing PCA for: all | dominant | others"
        )
        col1, col2, col3 = st.columns(3)

        # Plot 1: All
        with col1:
            st.subheader("All", divider=True)
            df_all = df_log2
            fig_pca_all = create_pca_plot(df_all, numeric_cols, " (all species)")
            if fig_pca_all:
                st.plotly_chart(fig_pca_all, width="stretch", key=f"pca_all_{label}")
            perm_all = permanova_core(df_all, numeric_cols)
            st.metric("F-stat", f"{perm_all['F']:.2f}" if not np.isnan(perm_all["F"]) else "N/A")
            st.metric("p-value", f"{perm_all['p']:.4f}" if not np.isnan(perm_all["p"]) else "N/A")

        # Plot 2: Dominant only
        with col2:
            dom = species_dist["most_frequent"]
            st.subheader(dom, divider=True)
            mask_dom = model.raw[species_col] == dom
            df_dom = df_log2[mask_dom]
            fig_pca_dom = create_pca_plot(df_dom, numeric_cols, f" ({dom} only)")
            if fig_pca_dom:
                st.plotly_chart(fig_pca_dom, width="stretch", key=f"pca_dom_{label}")
            perm_dom = permanova_core(df_dom, numeric_cols)
            st.metric("F-stat", f"{perm_dom['F']:.2f}" if not np.isnan(perm_dom["F"]) else "N/A")
            st.metric("p-value", f"{perm_dom['p']:.4f}" if not np.isnan(perm_dom["p"]) else "N/A")

        # Plot 3: Others only
        with col3:
            other_label = "+".join(species_dist["others"])
            st.subheader(other_label, divider=True)
            mask_oth = model.raw[species_col].isin(species_dist["others"])
            df_oth = df_log2[mask_oth]
            fig_pca_oth = create_pca_plot(df_oth, numeric_cols, f" ({other_label})")
            if fig_pca_oth:
                st.plotly_chart(fig_pca_oth, width="stretch", key=f"pca_oth_{label}")
            perm_oth = permanova_core(df_oth, numeric_cols)
            st.metric("F-stat", f"{perm_oth['F']:.2f}" if not np.isnan(perm_oth["F"]) else "N/A")
            st.metric("p-value", f"{perm_oth['p']:.4f}" if not np.isnan(perm_oth["p"]) else "N/A")

        st.markdown("---")

        ps = [perm_all["p"], perm_dom["p"], perm_oth["p"]]
        ps_clean = [p for p in ps if not np.isnan(p)]

        if ps_clean and all(p >= 0.05 for p in ps_clean):
            st.error(
                "🔴 **STRONG: Preprocessing REQUIRED**\n\n"
                "All tests non-significant. Start with Phase 1-2."
            )
        elif not np.isnan(perm_oth["p"]) and perm_oth["p"] < 0.05 and (np.isnan(perm_all["p"]) or perm_all["p"] >= 0.05):
            st.warning(
                "🟡 **MEDIUM: Preprocessing SUGGESTED**\n\n"
                "Signal in non-dominant species masked in combined analysis."
            )
        else:
            st.info("🟢 **Signal detected. Optional: Phase 1 filtering.**")

    else:
        st.caption("Single-species dataset.")
        col1, col2 = st.columns([2, 1])
        with col1:
            fig_pca = create_pca_plot(df_log2, numeric_cols)
            st.plotly_chart(fig_pca, width="stretch", key=f"pca_{label}")
        with col2:
            perm = permanova_core(df_log2, numeric_cols)
            st.metric("Pseudo-F", f"{perm['F']:.2f}" if not np.isnan(perm["F"]) else "N/A")
            st.metric("p-value", f"{perm['p']:.4f}" if not np.isnan(perm["p"]) else "N/A")

    st.markdown("---")

    st.markdown("### Normality analysis")
    stats_df = analyze_transformations(df_log2, numeric_cols, model.transforms)
    if not stats_df.empty:
        best_idx = stats_df["Shapiro W"].idxmax()
        best_transform = stats_df.loc[best_idx, "Transformation"]
        styled_df = stats_df.style.apply(
            lambda row: ["background-color: #B5BD00; color: white"] * len(row) if row["Transformation"] == best_transform else [""] * len(row),
            axis=1
        ).format({
            "Kurtosis": "{:.3f}",
            "Skewness": "{:.3f}",
            "Shapiro W": "{:.4f}",
            "Shapiro p": "{:.2e}",
        })
        st.dataframe(styled_df, width="stretch", hide_index=True)
        st.success(f"Recommended: **{best_transform}**")


with tab_protein:
    render_eda(protein_model, protein_idx, protein_species_col, "protein")

with tab_peptide:
    render_eda(peptide_model, peptide_idx, peptide_species_col, "peptide")

render_navigation(back_page="pages/1_Data_Upload.py", next_page=None)
render_footer()
