import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler
from sklearn.decomposition import PCA
from components import inject_custom_css, render_header, render_navigation, render_footer, COLORS
import plotly.express as px

st.set_page_config(
    page_title="EDA | Thermo Fisher Scientific",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

inject_custom_css()
render_header()


def parse_protein_group(pg_str: str) -> str:
    if pd.isna(pg_str):
        return "Unknown"
    return str(pg_str).split(";")[0].strip()


def get_numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def sort_columns_by_condition(cols: list[str]) -> list[str]:
    def sort_key(col):
        if len(col) >= 1 and col[0].isalpha():
            return (col[0], int(col[1:]) if col[1:].isdigit() else 0)
        return (col, 0)
    return sorted(cols, key=sort_key)


def get_condition_groups(cols: list[str]) -> dict:
    """Group columns by condition prefix (e.g., A1,A2,A3 -> A)."""
    groups = {}
    for col in cols:
        if len(col) >= 1 and col[0].isalpha():
            prefix = col[0]
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append(col)
    return groups


CONDITION_COLORS = [
    '#E71316', '#262262', '#EA7600', '#B5BD00', '#54585A', '#00A9E0',
    '#7B2D8E', '#00B388', '#F7941D', '#0072CE', '#DA1884', '#78BE20'
]


def get_condition_color_map(conditions: list[str]) -> dict:
    return {c: CONDITION_COLORS[i % len(CONDITION_COLORS)] for i, c in enumerate(sorted(conditions))}


TRANSFORMATIONS = {
    "Raw": lambda x: x,
    "Log2": lambda x: np.log2(np.maximum(x, 1)),
    "Log10": lambda x: np.log10(np.maximum(x, 1)),
    "Square root": lambda x: np.sqrt(x),
    "Cube root": lambda x: np.cbrt(x),
    "Yeo-Johnson": lambda x: PowerTransformer(method='yeo-johnson', standardize=False).fit_transform(x.reshape(-1, 1)).flatten() if len(x) > 1 else x,
    "Quantile": lambda x: QuantileTransformer(output_distribution='normal', random_state=42).fit_transform(x.reshape(-1, 1)).flatten() if len(x) > 1 else x,
}


@st.cache_data
def compute_normality_stats(values: np.ndarray) -> dict:
    clean = values[np.isfinite(values)]
    if len(clean) < 20:
        return {"kurtosis": np.nan, "skewness": np.nan, "W": np.nan, "p": np.nan}
    sample = np.random.choice(clean, min(5000, len(clean)), replace=False)
    try:
        W, p = stats.shapiro(sample)
    except:
        W, p = np.nan, np.nan
    return {"kurtosis": stats.kurtosis(clean), "skewness": stats.skew(clean), "W": W, "p": p}


@st.cache_data
def analyze_transformations(df_json: str, numeric_cols: list[str]) -> pd.DataFrame:
    df = pd.read_json(df_json)
    all_values = df[numeric_cols].values.flatten()
    all_values = all_values[np.isfinite(all_values) & (all_values > 0)]

    results = []
    for name, func in TRANSFORMATIONS.items():
        try:
            transformed = func(all_values.copy())
            stats_dict = compute_normality_stats(transformed)
        except:
            stats_dict = {"kurtosis": np.nan, "skewness": np.nan, "W": np.nan, "p": np.nan}
        results.append({
            "Transformation": name,
            "Kurtosis": stats_dict["kurtosis"],
            "Skewness": stats_dict["skewness"],
            "Shapiro W": stats_dict["W"],
            "Shapiro p": stats_dict["p"]
        })
    return pd.DataFrame(results)


@st.cache_data
def create_intensity_heatmap(df_json: str, index_col: str, numeric_cols: list[str]) -> go.Figure:
    df = pd.read_json(df_json)

    if index_col and index_col in df.columns:
        labels = df[index_col].apply(parse_protein_group).tolist()
    else:
        labels = [f"Row {i}" for i in range(len(df))]

    sorted_cols = sort_columns_by_condition(numeric_cols)
    intensity_log2 = np.log2(np.maximum(df[sorted_cols].values, 1))

    fig = go.Figure(data=go.Heatmap(
        z=intensity_log2, x=sorted_cols, y=labels,
        colorscale="Viridis", showscale=True,
        colorbar=dict(title="log2"),
        hovertemplate="Protein: %{y}<br>Sample: %{x}<br>log2: %{z:.2f}<extra></extra>"
    ))

    fig.update_layout(
        title="Intensity distribution (log2)",
        xaxis_title="Samples", yaxis_title="",
        height=500, yaxis=dict(tickfont=dict(size=8)),
        xaxis=dict(tickangle=45), plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Arial", color="#54585A")
    )
    return fig


@st.cache_data
def create_missing_distribution_chart(df_json: str, numeric_cols: list[str], label: str) -> go.Figure:
    df = pd.read_json(df_json)
    missing_per_row = (df[numeric_cols] == 1).sum(axis=1)
    total_rows = len(df)
    max_missing = len(numeric_cols)
    counts = [(missing_per_row == i).sum() / total_rows * 100 for i in range(max_missing + 1)]

    fig = go.Figure(data=go.Bar(
        x=[str(i) for i in range(max_missing + 1)], y=counts,
        marker_color="#262262",
        hovertemplate="Missing: %{x}<br>Percent: %{y:.1f}%<extra></extra>"
    ))
    fig.update_layout(
        title=f"Missing values per {label}",
        xaxis_title="Number of missing values", yaxis_title="% of total",
        height=350, plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#54585A"), bargap=0.2
    )
    return fig


import plotly.express as px

@st.cache_data
def create_condition_violin_plot(df_json: str, numeric_cols: list[str]) -> go.Figure:
    """Violin plot with px.violin: y=log2, x=replicate, color=condition."""
    df_wide = pd.read_json(df_json)
    groups = get_condition_groups(numeric_cols)  # {"A": ["A1","A2","A3"], ...}

    # Build long-form table
    records = []
    for condition, cols in groups.items():
        cols = sorted(cols)
        for rep_idx, col in enumerate(cols, start=1):
            vals = np.log2(np.maximum(df_wide[col].values, 1))
            for v in vals:
                records.append(
                    {"log2_value": v,
                     "replicate": f"R{rep_idx}",
                     "condition": condition,
                     "sample": col}
                )

    df_long = pd.DataFrame(records)

    fig = px.violin(
        df_long,
        y="log2_value",
        x="replicate",
        color="condition",
        box=True,
        points=False,
        hover_data=["sample", "condition", "replicate"],
    )
    return fig



@st.cache_data
def compute_pca(df_json: str, numeric_cols: list[str]) -> tuple:
    df = pd.read_json(df_json)
    data = np.log2(np.maximum(df[numeric_cols].values, 1))
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
    groups = get_condition_groups(cols)
    color_map = get_condition_color_map(list(groups.keys()))
    sorted_cols = sort_columns_by_condition(cols)
    conditions = [c[0] if len(c) >= 1 and c[0].isalpha() else "X" for c in sorted_cols]

    fig = go.Figure()

    for cond in sorted(groups.keys()):
        mask = [c == cond for c in conditions]
        indices = [i for i, m in enumerate(mask) if m]

        fig.add_trace(go.Scatter(
            x=pca_result[indices, 0],
            y=pca_result[indices, 1],
            mode='markers+text',
            marker=dict(size=12, color=color_map[cond]),
            text=[sorted_cols[i] for i in indices],
            textposition="top center",
            name=f"Condition {cond}",
            hovertemplate="Sample: %{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>"
        ))

    fig.update_layout(
        title="PCA - Sample clustering",
        xaxis_title=f"PC1 ({var_explained[0]*100:.1f}% variance)",
        yaxis_title=f"PC2 ({var_explained[1]*100:.1f}% variance)" if len(var_explained) > 1 else "PC2",
        height=450, plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#54585A"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


@st.cache_data
def compute_permanova(df_json: str, numeric_cols: list[str], n_perm: int = 999) -> dict:
    df = pd.read_json(df_json)
    data = np.log2(np.maximum(df[numeric_cols].values, 1))
    data_T = data.T
    sorted_cols = sort_columns_by_condition(numeric_cols)
    conditions = np.array([c[0] if len(c) >= 1 and c[0].isalpha() else "X" for c in sorted_cols])
    unique_conds = np.unique(conditions)

    if len(unique_conds) < 2:
        return {"F": np.nan, "p": np.nan, "R2": np.nan}

    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(data_T, metric='euclidean'))
    n = len(conditions)

    def compute_f_stat(dist_mat, groups):
        ss_total = np.sum(dist_mat ** 2) / (2 * n)
        ss_within = 0
        for g in np.unique(groups):
            mask = groups == g
            n_g = np.sum(mask)
            if n_g > 1:
                within_dist = dist_mat[np.ix_(mask, mask)]
                ss_within += np.sum(within_dist ** 2) / (2 * n_g)
        ss_between = ss_total - ss_within
        df_between = len(np.unique(groups)) - 1
        df_within = n - len(np.unique(groups))
        if df_within == 0 or ss_within == 0:
            return np.nan, np.nan
        f_stat = (ss_between / df_between) / (ss_within / df_within)
        r2 = ss_between / ss_total
        return f_stat, r2

    f_obs, r2 = compute_f_stat(dist_matrix, conditions)
    f_perms = []
    for _ in range(n_perm):
        perm_cond = np.random.permutation(conditions)
        f_perm, _ = compute_f_stat(dist_matrix, perm_cond)
        f_perms.append(f_perm)
    p_value = (np.sum(np.array(f_perms) >= f_obs) + 1) / (n_perm + 1)
    return {"F": f_obs, "p": p_value, "R2": r2}


st.markdown("## Exploratory data analysis")

protein_data = st.session_state.get("protein_data")
peptide_data = st.session_state.get("peptide_data")
protein_idx = st.session_state.get("protein_index_col")
peptide_idx = st.session_state.get("peptide_index_col")

if protein_data is None and peptide_data is None:
    st.warning("No data cached. Please upload data on the Data Upload page first.")
    render_navigation(back_page="pages/1_Data_Upload.py", next_page=None)
    render_footer()
    st.stop()

tab_protein, tab_peptide = st.tabs(["Protein data", "Peptide data"])

with tab_protein:
    if protein_data is not None:
        numeric_cols = get_numeric_cols(protein_data)
        st.caption(f"**{len(protein_data):,} proteins** × **{len(numeric_cols)} samples**")
        df_json = protein_data.to_json()

        col1, col2 = st.columns([2, 1])
        with col1:
            fig_heat = create_intensity_heatmap(df_json, protein_idx, numeric_cols)
            st.plotly_chart(fig_heat, use_container_width=True)
        with col2:
            fig_bar = create_missing_distribution_chart(df_json, numeric_cols, "protein groups")
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        st.markdown("### Sample distributions")
        fig_violin = create_condition_violin_plot(df_json, numeric_cols)
        st.plotly_chart(fig_violin, use_container_width=True)

        st.markdown("---")

        st.markdown("### Variance analysis")
        st.caption("PCA clustering and PERMANOVA test for biological variance (all conditions)")

        col1, col2 = st.columns([2, 1])
        with col1:
            fig_pca = create_pca_plot(df_json, numeric_cols)
            st.plotly_chart(fig_pca, use_container_width=True)

        with col2:
            st.markdown("#### PERMANOVA results")
            permanova = compute_permanova(df_json, numeric_cols)
            st.metric("Pseudo-F statistic", f"{permanova['F']:.2f}" if not np.isnan(permanova['F']) else "N/A")
            st.metric("R² (variance explained)", f"{permanova['R2']*100:.1f}%" if not np.isnan(permanova['R2']) else "N/A")
            st.metric("p-value", f"{permanova['p']:.4f}" if not np.isnan(permanova['p']) else "N/A")
            if permanova['p'] < 0.05:
                st.success("✓ Significant biological variance (p < 0.05)")
            elif not np.isnan(permanova['p']):
                st.warning("⚠ No significant biological variance detected")

        st.markdown("---")

        st.markdown("### Normality analysis")
        st.caption("Testing which transformation best normalizes intensity distributions")
        stats_df = analyze_transformations(df_json, numeric_cols)
        best_idx = stats_df["Shapiro W"].idxmax()
        best_transform = stats_df.loc[best_idx, "Transformation"]

        def highlight_best(row):
            if row["Transformation"] == best_transform:
                return ["background-color: #B5BD00; color: white"] * len(row)
            return [""] * len(row)

        styled_df = stats_df.style.apply(highlight_best, axis=1).format({
            "Kurtosis": "{:.3f}", "Skewness": "{:.3f}",
            "Shapiro W": "{:.4f}", "Shapiro p": "{:.2e}"
        })
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        st.success(f"Recommended: **{best_transform}** (Shapiro W = {stats_df.loc[best_idx, 'Shapiro W']:.4f})")
    else:
        st.info("No protein data uploaded yet")

with tab_peptide:
    if peptide_data is not None:
        numeric_cols = get_numeric_cols(peptide_data)
        st.caption(f"**{len(peptide_data):,} peptides** × **{len(numeric_cols)} samples**")
        df_json = peptide_data.to_json()

        col1, col2 = st.columns([2, 1])
        with col1:
            fig_heat = create_intensity_heatmap(df_json, peptide_idx, numeric_cols)
            st.plotly_chart(fig_heat, use_container_width=True)
        with col2:
            fig_bar = create_missing_distribution_chart(df_json, numeric_cols, "peptides")
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        st.markdown("### Sample distributions")
        fig_violin = create_condition_violin_plot(df_json, numeric_cols)
        st.plotly_chart(fig_violin, use_container_width=True)

        st.markdown("---")

        st.markdown("### Variance analysis")
        st.caption("PCA clustering and PERMANOVA test for biological variance (all conditions)")

        col1, col2 = st.columns([2, 1])
        with col1:
            fig_pca = create_pca_plot(df_json, numeric_cols)
            st.plotly_chart(fig_pca, use_container_width=True)

        with col2:
            st.markdown("#### PERMANOVA results")
            permanova = compute_permanova(df_json, numeric_cols)
            st.metric("Pseudo-F statistic", f"{permanova['F']:.2f}" if not np.isnan(permanova['F']) else "N/A")
            st.metric("R² (variance explained)", f"{permanova['R2']*100:.1f}%" if not np.isnan(permanova['R2']) else "N/A")
            st.metric("p-value", f"{permanova['p']:.4f}" if not np.isnan(permanova['p']) else "N/A")
            if permanova['p'] < 0.05:
                st.success("✓ Significant biological variance (p < 0.05)")
            elif not np.isnan(permanova['p']):
                st.warning("⚠ No significant biological variance detected")

        st.markdown("---")

        st.markdown("### Normality analysis")
        st.caption("Testing which transformation best normalizes intensity distributions")
        stats_df = analyze_transformations(df_json, numeric_cols)
        best_idx = stats_df["Shapiro W"].idxmax()
        best_transform = stats_df.loc[best_idx, "Transformation"]

        def highlight_best(row):
            if row["Transformation"] == best_transform:
                return ["background-color: #B5BD00; color: white"] * len(row)
            return [""] * len(row)

        styled_df = stats_df.style.apply(highlight_best, axis=1).format({
            "Kurtosis": "{:.3f}", "Skewness": "{:.3f}",
            "Shapiro W": "{:.4f}", "Shapiro p": "{:.2e}"
        })
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        st.success(f"Recommended: **{best_transform}** (Shapiro W = {stats_df.loc[best_idx, 'Shapiro W']:.4f})")
    else:
        st.info("No peptide data uploaded yet")

render_navigation(back_page="pages/1_Data_Upload.py", next_page=None)
render_footer()
