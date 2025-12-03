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
from typing import List, Dict, Tuple

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
    """Extract first protein ID from protein group string."""
    if pd.isna(pg_str):
        return "Unknown"
    return str(pg_str).split(";")[0].strip()


def extract_conditions(cols: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Extract condition letters and create color map."""
    conditions = [c[0] if c and c[0].isalpha() else "X" for c in cols]
    cond_order = sorted(set(conditions))
    color_map = {
        cond: TF_CHART_COLORS[i % len(TF_CHART_COLORS)]
        for i, cond in enumerate(cond_order)
    }
    return conditions, color_map


def sort_columns_by_condition(cols: List[str]) -> List[str]:
    """Sort columns by condition letter then numeric suffix (A1, A2, B1, B2, etc.)."""
    def sort_key(col: str):
        if col and col[0].isalpha():
            head, tail = col[0], col[1:]
            return (head, int(tail) if tail.isdigit() else 0)
        return (col, 0)
    return sorted(cols, key=sort_key)


@st.cache_data
def create_intensity_heatmap(
    df_log2: pd.DataFrame, index_col: str | None, numeric_cols: List[str]
) -> go.Figure:
    """Create heatmap of intensity values (top 100 by variance)."""
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
        labels = [labels[list(df_log2.index).index(i)] for i in top_idx]
    
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
        margin=dict(l=60, r=40, t=60, b=60),
    )
    
    return fig


@st.cache_data
def create_missing_distribution_chart(mask: pd.DataFrame, label: str) -> go.Figure:
    """Create bar chart showing distribution of missing values per row."""
    missing_per_row = mask.sum(axis=1)
    total_missing = mask.sum().sum()
    max_missing = mask.shape[1]
    
    counts = [
        (missing_per_row == i).sum() / len(mask) * 100
        for i in range(max_missing + 1)
    ]

    fig = go.Figure(
        data=go.Bar(
            x=[str(i) for i in range(max_missing + 1)],
            y=counts,
            marker_color="#262262",
            hovertemplate="Missing values: %{x}<br>Percent: %{y:.1f}%<extra></extra>",
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
        margin=dict(l=60, r=40, t=60, b=60),
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
def create_violin_plot(df_log2: pd.DataFrame, numeric_cols: List[str]) -> go.Figure:
    """Create violin plot showing replicate distributions per condition."""
    long_df = df_log2[numeric_cols].melt(var_name="Sample", value_name="log2_value")
    long_df["Condition"] = long_df["Sample"].str.extract(r"^([A-Z])")
    long_df["Replicate"] = long_df["Sample"].str.extract(r"(\d+)$")
    long_df["CondRep"] = long_df["Condition"] + long_df["Replicate"]

    # Sort order: A1, A2, ..., B1, B2, ...
    cond_order = sorted(long_df["Condition"].dropna().unique())
    x_order = []
    for cond in cond_order:
        reps = sorted(
            long_df.loc[long_df["Condition"] == cond, "Replicate"].dropna().unique(),
            key=int,
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

    fig.update_traces(width=0.7, scalemode="count")
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
        margin=dict(l=60, r=40, t=80, b=60),
    )
    
    return fig


def create_pca_plot(
    df_log2: pd.DataFrame, numeric_cols: List[str], title_suffix: str = ""
) -> go.Figure | None:
    """Create PCA scatter plot colored by condition."""
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
    
    n_components = min(2, data_scaled.shape[0], data_scaled.shape[1])
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data_scaled)

    sorted_cols = sort_columns_by_condition(numeric_cols)
    conditions, color_map = extract_conditions(sorted_cols)

    fig = go.Figure()
    for cond in sorted(set(conditions)):
        idx = [i for i, c in enumerate(conditions) if c == cond]
        fig.add_trace(
            go.Scatter(
                x=pca_result[idx, 0],
                y=pca_result[idx, 1] if n_components > 1 else np.zeros(len(idx)),
                mode="markers+text",
                marker=dict(size=12, color=color_map[cond]),
                text=[sorted_cols[i] for i in idx],
                textposition="top center",
                name=f"Condition {cond}",
                hovertemplate="Sample: %{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
            )
        )

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100 if n_components > 1 else 0

    fig.update_layout(
        title=f"PCA{title_suffix}",
        xaxis_title=f"PC1 ({var1:.1f}% var)",
        yaxis_title=f"PC2 ({var2:.1f}% var)",
        height=400,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#54585A"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=60, b=60),
    )
    
    return fig


def compute_permanova(df_log2: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, float]:
    """Compute PERMANOVA (permutational MANOVA) statistics - CORRECTED VERSION."""
    if len(df_log2) < 3:
        return {"F": np.nan, "p": np.nan, "R2": np.nan}
    
    # Transpose: samples as rows, proteins as columns
    data = df_log2[numeric_cols].T.values
    
    # Get conditions from column names
    sorted_cols = sort_columns_by_condition(numeric_cols)
    conditions = np.array([c[0] if c and c[0].isalpha() else "X" for c in sorted_cols])

    unique_conds = np.unique(conditions)
    if len(unique_conds) < 2:
        return {"F": np.nan, "p": np.nan, "R2": np.nan}

    # ============ CRITICAL FIX: Remove proteins with NaN in ANY sample ============
    # This ensures all samples have the same set of proteins
    valid_proteins = ~np.isnan(data).any(axis=0)
    data_clean = data[:, valid_proteins]
    
    # Ensure we have enough data
    if data_clean.shape[1] < 3 or data_clean.shape[0] < 3:
        return {"F": np.nan, "p": np.nan, "R2": np.nan}
    # ==============================================================================

    # Calculate distance matrix on cleaned data
    dist_matrix = squareform(pdist(data_clean, metric="euclidean"))
    n = len(conditions)

    def calc_f_statistic(dist_mat, groups):
        ss_total = np.sum(dist_mat**2) / (2 * n)
        ss_within = 0.0
        
        for g in np.unique(groups):
            mask = groups == g
            if mask.sum() > 1:
                ss_within += np.sum(dist_mat[np.ix_(mask, mask)]**2) / (2 * mask.sum())
        
        ss_between = ss_total - ss_within
        df_between = len(np.unique(groups)) - 1
        df_within = n - len(np.unique(groups))
        
        if df_within == 0 or ss_within == 0:
            return np.nan, np.nan
        
        F = (ss_between / df_between) / (ss_within / df_within)
        R2 = ss_between / ss_total if ss_total > 0 else 0
        return F, R2

    F_obs, R2 = calc_f_statistic(dist_matrix, conditions)
    
    if np.isnan(F_obs):
        return {"F": F_obs, "p": np.nan, "R2": R2}
    
    # Permutation test
    np.random.seed(42)  # Make deterministic
    f_perms = [
        calc_f_statistic(dist_matrix, np.random.permutation(conditions))[0]
        for _ in range(999)
    ]
    f_perms = np.array([f for f in f_perms if not np.isnan(f)])
    
    p_val = (np.sum(f_perms >= F_obs) + 1) / (len(f_perms) + 1) if len(f_perms) > 0 else np.nan
    
    return {"F": F_obs, "p": p_val, "R2": R2}


@st.cache_data
def analyze_transformations(
    df_log2: pd.DataFrame, numeric_cols: List[str], transforms: TransformsCache
) -> pd.DataFrame:
    """Compute normality statistics for different transformations."""
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
            np.random.seed(42)  # Make deterministic
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


def render_pca_with_stats(
    df: pd.DataFrame, numeric_cols: List[str], title: str, label: str, key_suffix: str
):
    """Render PCA plot with PERMANOVA statistics."""
    fig_pca = create_pca_plot(df, numeric_cols, f" ({title})")
    if fig_pca:
        st.plotly_chart(fig_pca, use_container_width=True, key=f"pca_{key_suffix}_{label}")
    
    perm = compute_permanova(df, numeric_cols)
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("F-stat", f"{perm['F']:.2f}" if not np.isnan(perm["F"]) else "N/A")
    with col_m2:
        st.metric("p-value", f"{perm['p']:.4f}" if not np.isnan(perm["p"]) else "N/A")
    
    if not np.isnan(perm["R2"]):
        st.caption(f"R² = {perm['R2']:.3f}")
    
    return perm


st.markdown("## Exploratory Data Analysis")

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

# Info box about workflow
st.info(
    "📊 **EDA Workflow**: This page provides qualitative overview and initial statistics. "
    "Use **Preprocessing** page for transformations and **Filtering** page for detailed quality control with interactive filters."
)

tab_protein, tab_peptide = st.tabs(["Protein data", "Peptide data"])


def render_eda(
    model: MSData | None, index_col: str | None, species_col: str | None, label: str
):
    """Render EDA visualizations and statistics."""
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

    # Row 1: Heatmap + Missing distribution
    st.markdown("### Data Quality Overview")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_heat = create_intensity_heatmap(df_log2, index_col, numeric_cols)
        st.plotly_chart(fig_heat, use_container_width=True, key=f"heatmap_{label}")
    with col2:
        fig_bar = create_missing_distribution_chart(mask, label)
        st.plotly_chart(fig_bar, use_container_width=True, key=f"missing_{label}")

    st.markdown("---")

    # Row 2: Violin plot
    st.markdown("### Sample Distributions")
    fig_violin = create_violin_plot(df_log2, numeric_cols)
    st.plotly_chart(fig_violin, use_container_width=True, key=f"violin_{label}")

    st.markdown("---")

    # Row 3: Variance analysis with species-stratified PCA
    st.markdown("### Variance Analysis & Group Separation")

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
            f"**Species detected:** {', '.join(species_dist['species'])}. "
            "Showing PCA for: All | Dominant | Others"
        )
        
        col1, col2, col3 = st.columns(3, vertical_alignment="top", border=True)

        # All species
        with col1:
            st.subheader("All Species", divider=True)
            perm_all = render_pca_with_stats(df_log2, numeric_cols, "all species", label, "all")

        # Dominant species only
        with col2:
            dom = species_dist["most_frequent"]
            st.subheader(dom, divider=True)
            mask_dom = model.raw[species_col] == dom
            df_dom = df_log2[mask_dom]
            perm_dom = render_pca_with_stats(df_dom, numeric_cols, f"{dom} only", label, "dom")

        # Other species combined
        with col3:
            other_label = "+".join(species_dist["others"])
            st.subheader(other_label, divider=True)
            mask_oth = model.raw[species_col].isin(species_dist["others"])
            df_oth = df_log2[mask_oth]
            perm_oth = render_pca_with_stats(df_oth, numeric_cols, other_label, label, "oth")

        st.markdown("---")

        # Interpretation
        st.markdown("#### 🔍 Interpretation")
        
        ps = [perm_all["p"], perm_dom["p"], perm_oth["p"]]
        ps_clean = [p for p in ps if not np.isnan(p)]

        if ps_clean and all(p >= 0.05 for p in ps_clean):
            st.error(
                "🔴 **STRONG: Filtering REQUIRED**\n\n"
                "All PERMANOVA tests non-significant (p ≥ 0.05). No separation between conditions detected. "
                "**Action**: Apply filters on Filtering page to remove noisy proteins/peptides."
            )
        elif (not np.isnan(perm_oth["p"]) and perm_oth["p"] < 0.05 
              and (np.isnan(perm_all["p"]) or perm_all["p"] >= 0.05)):
            st.warning(
                "🟡 **MEDIUM: Filtering SUGGESTED**\n\n"
                "Signal detected in non-dominant species (p < 0.05) but masked in combined analysis. "
                "**Action**: Consider species-specific filtering or separate analysis."
            )
        elif ps_clean and any(p < 0.05 for p in ps_clean):
            st.success(
                "🟢 **Signal Detected**\n\n"
                f"Condition separation detected (PERMANOVA p < 0.05). "
                "**Optional**: Use Filtering page to further improve signal quality."
            )
        else:
            st.info("ℹ️ Insufficient data for statistical testing.")

    else:
        st.caption("Single-species dataset or species column not available.")
        col1, col2 = st.columns([2, 1])
        with col1:
            fig_pca = create_pca_plot(df_log2, numeric_cols)
            if fig_pca:
                st.plotly_chart(fig_pca, use_container_width=True, key=f"pca_{label}")
        with col2:
            perm = compute_permanova(df_log2, numeric_cols)
            st.metric("Pseudo-F", f"{perm['F']:.2f}" if not np.isnan(perm["F"]) else "N/A")
            st.metric("p-value", f"{perm['p']:.4f}" if not np.isnan(perm["p"]) else "N/A")
            if not np.isnan(perm["R2"]):
                st.caption(f"R² = {perm['R2']:.3f}")

    st.markdown("---")

    # Row 4: Normality analysis
    st.markdown("### Normality Analysis & Transformation Comparison")
    st.caption("Comparing different transformations. Best transformation highlighted in yellow.")
    
    stats_df = analyze_transformations(df_log2, numeric_cols, model.transforms)
    
    if not stats_df.empty:
        best_idx = stats_df["Shapiro W"].idxmax()
        best_transform = stats_df.loc[best_idx, "Transformation"]
        
        styled_df = stats_df.style.apply(
            lambda row: (
                ["background-color: #B5BD00; color: white"] * len(row)
                if row["Transformation"] == best_transform
                else [""] * len(row)
            ),
            axis=1,
        ).format({
            "Kurtosis": "{:.3f}",
            "Skewness": "{:.3f}",
            "Shapiro W": "{:.4f}",
            "Shapiro p": "{:.2e}",
        })
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        st.success(f"✓ Recommended transformation: **{best_transform}**")
        st.caption(
            "**Shapiro-Wilk test**: W closer to 1.0 indicates better normality. "
            "Select transformation on Preprocessing page."
        )


with tab_protein:
    render_eda(protein_model, protein_idx, protein_species_col, "protein")

with tab_peptide:
    render_eda(peptide_model, peptide_idx, peptide_species_col, "peptide")

render_navigation(back_page="pages/1_Data_Upload.py", next_page="pages/3_Preprocessing.py")
render_footer()
