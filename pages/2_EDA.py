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
    transforms: object


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
def detect_species_distribution(df_json: str, species_col: str | None) -> dict | None:
    """Detect species composition if species_col exists."""
    if not species_col:
        return None
    df = pd.read_json(df_json)
    if species_col not in df.columns:
        return None
    
    species_counts = df[species_col].value_counts()
    if len(species_counts) < 2:
        return None
    
    return {
        "species": species_counts.index.tolist(),
        "counts": species_counts.values.tolist(),
        "most_frequent": species_counts.index[0],
        "others": species_counts.index[1:].tolist(),
    }


@st.cache_data
def create_filtered_pca_plot(df_json: str, numeric_cols: list[str], species_col: str | None, filter_species: list[str] | None, label_suffix: str = "") -> go.Figure:
    """Create PCA for filtered species subset."""
    df = pd.read_json(df_json)
    
    # Filter by species if specified
    if filter_species and species_col:
        if species_col in df.columns:
            mask = df[species_col].isin(filter_species)
            df = df[mask]
    
    if len(df) < 3:
        return None
    
    numeric_cols_filtered = [c for c in numeric_cols if c in df.columns]
    data = df[numeric_cols_filtered].T.values
    valid_cols = np.std(data, axis=0) > 0
    data_clean = data[:, valid_cols]
    
    if data_clean.shape[1] < 2:
        return None
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_clean)
    pca = PCA(n_components=min(3, data_scaled.shape[0]))
    pca_result = pca.fit_transform(data_scaled)

    sorted_cols = sort_columns_by_condition(numeric_cols_filtered)
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
        title=f"PCA - Sample clustering{label_suffix}",
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)" if len(pca.explained_variance_ratio_) > 1 else "PC2",
        height=400, plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#54585A"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


@st.cache_data
def compute_permanova(df_json: str, numeric_cols: list[str], n_perm: int = 999) -> dict:
    """PERMANOVA for all data."""
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

    F_obs, R2 = f_stat(dist_matrix, conditions)
    f_perms = [f_stat(dist_matrix, np.random.permutation(conditions))[0] for _ in range(n_perm)]
    p_val = (np.sum(np.array(f_perms) >= F_obs) + 1) / (n_perm + 1)

    return {"F": F_obs, "p": p_val, "R2": R2}


@st.cache_data
def compute_permanova_filtered(df_json: str, numeric_cols: list[str], species_col: str | None, filter_species: list[str] | None, n_perm: int = 999) -> dict:
    """PERMANOVA for filtered species subset."""
    df = pd.read_json(df_json)
    
    if filter_species and species_col and species_col in df.columns:
        mask = df[species_col].isin(filter_species)
        df = df[mask]
    
    if len(df) < 3:
        return {"F": np.nan, "p": np.nan, "R2": np.nan}
    
    numeric_cols_filtered = [c for c in numeric_cols if c in df.columns]
    data = df[numeric_cols_filtered].T.values
    sorted_cols = sort_columns_by_condition(numeric_cols_filtered)
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
    f_perms = [f_stat(dist_matrix, np.random.permutation(conditions))[0] for _ in range(n_perm)]
    p_val = (np.sum(np.array(f_perms) >= F_obs) + 1) / (n_perm + 1)

    return {"F": F_obs, "p": p_val, "R2": R2}


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
def analyze_transformations(df_json: str, numeric_cols: list[str], transforms_obj) -> pd.DataFrame:
    """Analyze all precomputed transformations."""
    df = pd.read_json(df_json)
    
    transforms_dict = {
        "Raw (log2 filled)": df[numeric_cols].values.flatten(),
        "Log10": transforms_obj.log10[numeric_cols].values.flatten(),
        "Square root": transforms_obj.sqrt[numeric_cols].values.flatten(),
        "Cube root": transforms_obj.cbrt[numeric_cols].values.flatten(),
        "Yeo-Johnson": transforms_obj.yeo_johnson[numeric_cols].values.flatten(),
        "Quantile": transforms_obj.quantile[numeric_cols].values.flatten(),
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


# -----------------------
# Page logic
# -----------------------
st.markdown("## Exploratory data analysis")

protein_model = st.session_state.get("protein_model")
peptide_model = st.session_state.get("peptide_model")
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

    numeric_cols = model.numeric_cols
    df_json = model.transforms.log2[numeric_cols].to_json()
    mask = st.session_state.get(f"{label}_missing_mask")
    mask_json = (mask if mask is not None else pd.DataFrame(False, index=model.transforms.log2.index, columns=numeric_cols)).to_json()

    st.caption(f"**{len(model.transforms.log2):,} {label}s** × **{len(numeric_cols)} samples** | **{model.missing_count:,} missing cells** (NaN/0/1)")

    # Row 1: Heatmap + missing
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_heat = create_intensity_heatmap(df_json, index_col, numeric_cols)
        st.plotly_chart(fig_heat, width='stretch', key=f"heatmap_{label}")
    with col2:
        fig_bar = create_missing_distribution_chart(mask_json, label)
        st.plotly_chart(fig_bar, width='stretch', key=f"missing_{label}")

    st.markdown("---")

    # Row 2: Violin
    st.markdown("### Sample distributions (all conditions)")
    fig_violin = create_violin_plot(df_json, numeric_cols)
    st.plotly_chart(fig_violin, width='stretch', key=f"violin_{label}")

    st.markdown("---")

    # Row 3: Variance analysis with multi-species PCA
    st.markdown("### Variance analysis")
    
    # Detect species composition
    species_dist = detect_species_distribution(df_json, index_col)
    
    if species_dist and len(species_dist["species"]) >= 2:
        # Multi-species layout
        st.caption(f"Species detected: {', '.join(species_dist['species'])}. Showing PCA for: All | {species_dist['most_frequent']} only | {'+'.join(species_dist['others'])}")
        
        col1, col2, col3 = st.columns(3)
        
        # PCA 1: All species
        with col1:
            st.subheader("All species", divider=True)
            fig_pca_all = create_filtered_pca_plot(df_json, numeric_cols, index_col, None, " (all)")
            if fig_pca_all:
                st.plotly_chart(fig_pca_all, width='stretch', key=f"pca_all_{label}")
                permanova_all = compute_permanova_filtered(df_json, numeric_cols, index_col, None)
                st.metric("F-stat", f"{permanova_all['F']:.2f}" if not np.isnan(permanova_all['F']) else "N/A")
                st.metric("p-value", f"{permanova_all['p']:.4f}" if not np.isnan(permanova_all['p']) else "N/A")
        
        # PCA 2: Most frequent species only
        with col2:
            st.subheader(species_dist['most_frequent'], divider=True)
            fig_pca_most = create_filtered_pca_plot(df_json, numeric_cols, index_col, [species_dist['most_frequent']], f" ({species_dist['most_frequent']})")
            if fig_pca_most:
                st.plotly_chart(fig_pca_most, width='stretch', key=f"pca_most_{label}")
                permanova_most = compute_permanova_filtered(df_json, numeric_cols, index_col, [species_dist['most_frequent']])
                st.metric("F-stat", f"{permanova_most['F']:.2f}" if not np.isnan(permanova_most['F']) else "N/A")
                st.metric("p-value", f"{permanova_most['p']:.4f}" if not np.isnan(permanova_most['p']) else "N/A")
        
        # PCA 3: Other species combined
        with col3:
            other_label = "+".join(species_dist['others'])
            st.subheader(other_label, divider=True)
            fig_pca_others = create_filtered_pca_plot(df_json, numeric_cols, index_col, species_dist['others'], f" ({other_label})")
            if fig_pca_others:
                st.plotly_chart(fig_pca_others, width='stretch', key=f"pca_others_{label}")
                permanova_others = compute_permanova_filtered(df_json, numeric_cols, index_col, species_dist['others'])
                st.metric("F-stat", f"{permanova_others['F']:.2f}" if not np.isnan(permanova_others['F']) else "N/A")
                st.metric("p-value", f"{permanova_others['p']:.4f}" if not np.isnan(permanova_others['p']) else "N/A")
        
        # Preprocessing recommendation
        st.markdown("---")
        all_sig = [permanova_all['p'], permanova_most['p'], permanova_others['p']]
        all_sig_clean = [p for p in all_sig if not np.isnan(p)]
        
        if all(p >= 0.05 for p in all_sig_clean):
            st.error("🔴 **STRONG recommendation: Preprocessing REQUIRED**\n\nAll PCA/PERMANOVA tests are non-significant. "
                    "This indicates technical noise dominates biological signal. Start with:\n"
                    "1. **Phase 1:** Data quality filtering (remove low-abundance, high-missing)\n"
                    "2. **Phase 2:** Batch-aware missing value imputation\n"
                    "3. **Re-analyze** before proceeding to statistics.")
        elif not np.isnan(permanova_others['p']) and permanova_others['p'] < 0.05 and permanova_all['p'] >= 0.05:
            st.warning("🟡 **MEDIUM recommendation: Preprocessing SUGGESTED**\n\nSignal detected in non-dominant species but masked in combined analysis. "
                      "Try:\n"
                      "1. **Phase 3:** Batch effect correction (ComBat/SVA)\n"
                      "2. **Phase 4:** Within-species normalization")
        else:
            st.info("🟢 **Signal detected. Optional: Phase 1 filtering to reduce noise.**")
    
    else:
        # Single species - use original single PCA
        st.caption("PCA clustering and PERMANOVA test for biological variance (conditions = first letter of sample).")
        col1, col2 = st.columns([2, 1])
        with col1:
            fig_pca = create_filtered_pca_plot(df_json, numeric_cols, None, None)
            st.plotly_chart(fig_pca, width='stretch', key=f"pca_{label}")
        with col2:
            st.markdown("#### PERMANOVA results")
            permanova = compute_permanova(df_json, numeric_cols)
            st.metric("Pseudo-F", f"{permanova['F']:.2f}" if not np.isnan(permanova['F']) else "N/A")
            st.metric("R² (var explained)", f"{permanova['R2']*100:.1f}%" if not np.isnan(permanova['R2']) else "N/A")
            st.metric("p-value", f"{permanova['p']:.4f}" if not np.isnan(permanova['p']) else "N/A")
            if permanova['p'] < 0.05:
                st.success("✓ Significant biological variance (p < 0.05)")
            elif not np.isnan(permanova['p']):
                st.warning("No significant biological variance detected")

    st.markdown("---")

    # Row 4: Normality
    st.markdown("### Normality analysis")
    st.caption("Testing which transformation best normalizes intensity distributions.")

    stats_df = analyze_transformations(df_json, numeric_cols, model.transforms)
    best_idx = stats_df["Shapiro W"].idxmax()
    best_transform = stats_df.loc[best_idx, "Transformation"]

    styled_df = stats_df.style.apply(
        lambda row: ["background-color: #B5BD00; color: white"] * len(row) if row["Transformation"] == best_transform else [""] * len(row),
        axis=1
    ).format({"Kurtosis": "{:.3f}", "Skewness": "{:.3f}", "Shapiro W": "{:.4f}", "Shapiro p": "{:.2e}"})

    st.dataframe(styled_df, width='stretch', hide_index=True)
    st.success(f"Recommended: **{best_transform}** (Shapiro W = {stats_df.loc[best_idx, 'Shapiro W']:.4f})")


with tab_protein:
    render_eda(protein_model, protein_idx, "protein")

with tab_peptide:
    render_eda(peptide_model, peptide_idx, "peptide")

render_navigation(back_page="pages/1_Data_Upload.py", next_page=None)
render_footer()
