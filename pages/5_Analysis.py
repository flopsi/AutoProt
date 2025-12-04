import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from scipy import stats
from scipy.stats import ttest_ind
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

from components import inject_custom_css, render_header, render_navigation, render_footer, COLORS

st.set_page_config(
    page_title="Differential Expression | Thermo Fisher Scientific",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()
render_header()


# ========== DATA CLASSES & HELPERS ==========

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
class FilteredSubgroups:
    human: pd.DataFrame
    yeast: pd.DataFrame
    ecoli: pd.DataFrame
    mouse: pd.DataFrame
    all_species: pd.DataFrame

    def get(self, species_list: List[str]) -> pd.DataFrame:
        if not species_list:
            return self.all_species
        if set(species_list) == {"HUMAN", "YEAST", "ECOLI", "MOUSE"}:
            return self.all_species
        dfs = []
        for sp in species_list:
            if sp == "HUMAN" and not self.human.empty:
                dfs.append(self.human)
            elif sp == "YEAST" and not self.yeast.empty:
                dfs.append(self.yeast)
            elif sp == "ECOLI" and not self.ecoli.empty:
                dfs.append(self.ecoli)
            elif sp == "MOUSE" and not self.mouse.empty:
                dfs.append(self.mouse)
        return pd.concat(dfs, axis=0) if dfs else pd.DataFrame()


@dataclass
class MSData:
    raw: pd.DataFrame
    raw_filled: pd.DataFrame
    missing_count: int
    numeric_cols: List[str]
    transforms: TransformsCache
    species_subgroups: FilteredSubgroups
    species_col: str


TF_CHART_COLORS = ["#262262", "#EA7600", "#B5BD00", "#A6192E"]


def extract_conditions(cols: List[str]) -> Dict[str, str]:
    return {col: (col[0] if col and col[0].isalpha() else "X") for col in cols}


def build_condition_groups(numeric_cols: List[str]) -> Dict[str, List[str]]:
    condition_map = extract_conditions(numeric_cols)
    groups: Dict[str, List[str]] = {}
    for col in numeric_cols:
        groups.setdefault(condition_map[col], []).append(col)
    return groups


def get_transform_data(model: MSData, transform_key: str) -> pd.DataFrame:
    return getattr(model.transforms, transform_key, model.transforms.log2)


def compute_cv_per_condition(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    condition_groups = build_condition_groups(numeric_cols)
    cv_results = {}
    for cond, cols in condition_groups.items():
        if len(cols) < 2:
            continue
        mean_vals = df[cols].mean(axis=1)
        std_vals = df[cols].std(axis=1)
        cv = (std_vals / mean_vals * 100).replace([np.inf, -np.inf], np.nan)
        cv_results[f"CV_{cond}"] = cv
    return pd.DataFrame(cv_results, index=df.index)


def perform_ttest_analysis(
    df: pd.DataFrame,
    group1_cols: List[str],
    group2_cols: List[str],
    min_valid: int = 2,
) -> pd.DataFrame:
    results = []
    
    for idx, row in df.iterrows():
        g1_vals = row[group1_cols].dropna()
        g2_vals = row[group2_cols].dropna()
        
        if len(g1_vals) < min_valid or len(g2_vals) < min_valid:
            results.append({
                "protein_id": idx,
                "log2fc": np.nan,
                "pvalue": np.nan,
                "mean_group1": np.nan,
                "mean_group2": np.nan,
                "n_group1": len(g1_vals),
                "n_group2": len(g2_vals),
            })
            continue
        
        mean1 = g1_vals.mean()
        mean2 = g2_vals.mean()
        
        # Limma convention: log2(A) - log2(B) = log2(A/B)
        log2fc = mean1 - mean2  # CHANGED: was mean2 - mean1
        
        try:
            t_stat, pval = ttest_ind(g1_vals, g2_vals, equal_var=False)
        except Exception:
            t_stat, pval = np.nan, np.nan
        
        results.append({
            "protein_id": idx,
            "log2fc": log2fc,
            "pvalue": pval,
            "mean_group1": mean1,
            "mean_group2": mean2,
            "n_group1": len(g1_vals),
            "n_group2": len(g2_vals),
        })
    
    results_df = pd.DataFrame(results)
    results_df.set_index("protein_id", inplace=True)
    
    results_df["neg_log10_pval"] = -np.log10(results_df["pvalue"].replace(0, 1e-300))
    
    pvals = results_df["pvalue"].dropna().sort_values()
    n = len(pvals)
    if n > 0:
        ranks = np.arange(1, n + 1)
        fdr_vals = pvals.values * n / ranks
        fdr_vals = np.minimum.accumulate(fdr_vals[::-1])[::-1]
        fdr_dict = dict(zip(pvals.index, fdr_vals))
        results_df["fdr"] = results_df.index.map(fdr_dict)
    else:
        results_df["fdr"] = np.nan
    
    return results_df



def classify_regulation(row: pd.Series, fc_threshold: float, pval_threshold: float) -> str:
    if pd.isna(row["log2fc"]) or pd.isna(row["pvalue"]):
        return "Not tested"
    
    if row["pvalue"] > pval_threshold:
        return "Not significant"
    
    if row["log2fc"] > fc_threshold:
        return "Up-regulated"
    elif row["log2fc"] < -fc_threshold:
        return "Down-regulated"
    else:
        return "Not significant"


def calculate_error_rates(
    results_df: pd.DataFrame,
    true_fc_dict: Dict[str, float],
    fc_threshold: float,
    pval_threshold: float,
) -> Dict[str, float]:
    results_df["true_log2fc"] = results_df.index.map(
        lambda x: true_fc_dict.get(x, 0.0)
    )
    
    results_df["true_regulated"] = results_df["true_log2fc"].apply(
        lambda x: abs(x) > fc_threshold
    )
    
    results_df["observed_regulated"] = results_df["regulation"].isin(
        ["Up-regulated", "Down-regulated"]
    )
    
    TP = ((results_df["true_regulated"]) & (results_df["observed_regulated"])).sum()
    FP = ((~results_df["true_regulated"]) & (results_df["observed_regulated"])).sum()
    TN = ((~results_df["true_regulated"]) & (~results_df["observed_regulated"])).sum()
    FN = ((results_df["true_regulated"]) & (~results_df["observed_regulated"])).sum()
    
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    return {
        "TP": int(TP),
        "FP": int(FP),
        "TN": int(TN),
        "FN": int(FN),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "FPR": fpr,
        "FNR": fnr,
        "precision": precision,
    }


def calculate_species_rmse(
    results_df: pd.DataFrame,
    true_fc_dict: Dict[str, float],
    species_col_data: Dict,
    fc_threshold: float,
    pval_threshold: float,
) -> pd.DataFrame:
    """Calculate RMSE and other metrics per species."""
    species_metrics = []
    
    for species in ["HUMAN", "YEAST", "ECOLI"]:
        # Get proteins for this species
        species_proteins = [pid for pid, sp in species_col_data.items() if sp == species]
        
        if not species_proteins:
            continue
        
        # Get results for this species
        species_results = results_df.loc[results_df.index.intersection(species_proteins)].copy()
        
        if len(species_results) == 0:
            continue
        
        # Get theoretical FC for this species
        theo_fc = true_fc_dict.get(species_proteins[0], 0.0)
        
        # Calculate prediction error
        species_results["error"] = species_results["log2fc"] - theo_fc
        
        # Metrics
        n_proteins = len(species_results)
        rmse = np.sqrt((species_results["error"] ** 2).mean())
        mae = species_results["error"].abs().mean()
        bias = species_results["error"].mean()
        
        # Classification accuracy
        species_results["regulation"] = species_results.apply(
            lambda row: classify_regulation(row, fc_threshold, pval_threshold),
            axis=1
        )
        
        n_detected = (species_results["regulation"].isin(["Up-regulated", "Down-regulated"])).sum()
        detection_rate = n_detected / n_proteins if n_proteins > 0 else 0
        
        species_metrics.append({
            "Species": species,
            "N": n_proteins,
            "Theo FC": f"{theo_fc:.2f}",
            "RMSE": f"{rmse:.3f}",
            "MAE": f"{mae:.3f}",
            "Detection": f"{detection_rate:.1%}",
        })
    
    return pd.DataFrame(species_metrics)


def create_volcano_plot(
    results_df: pd.DataFrame,
    fc_threshold: float,
    pval_threshold: float,
    title: str = "Volcano Plot",
) -> go.Figure:
    results_df["regulation"] = results_df.apply(
        lambda row: classify_regulation(row, fc_threshold, pval_threshold),
        axis=1
    )
    
    color_map = {
        "Up-regulated": "#EA7600",
        "Down-regulated": "#262262",
        "Not significant": "#CCCCCC",
        "Not tested": "#999999",
    }
    
    fig = go.Figure()
    
    for reg_type in ["Not significant", "Not tested", "Down-regulated", "Up-regulated"]:
        subset = results_df[results_df["regulation"] == reg_type]
        if not subset.empty:
            fig.add_trace(go.Scatter(
                x=subset["log2fc"],
                y=subset["neg_log10_pval"],
                mode="markers",
                name=reg_type,
                marker=dict(
                    color=color_map[reg_type],
                    size=6,
                    opacity=0.7 if reg_type in ["Up-regulated", "Down-regulated"] else 0.3,
                ),
                text=subset.index,
                hovertemplate="<b>%{text}</b><br>" +
                             "log2FC: %{x:.2f}<br>" +
                             "-log10(p): %{y:.2f}<br>" +
                             "<extra></extra>",
            ))
    
    fig.add_hline(
        y=-np.log10(pval_threshold),
        line_dash="dash",
        line_color="red",
        annotation_text=f"p={pval_threshold}",
    )
    fig.add_vline(x=fc_threshold, line_dash="dash", line_color="red")
    fig.add_vline(x=-fc_threshold, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title=title,
        xaxis_title="log2 Fold Change",
        yaxis_title="-log10(p-value)",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#54585A"),
        height=600,
        hovermode="closest",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )
    
    return fig


def create_ma_plot_with_theoretical(
    results_df: pd.DataFrame,
    fc_threshold: float,
    pval_threshold: float,
    theo_fc_species: Dict[str, float],
) -> go.Figure:
    """Create MA plot with theoretical FC lines for each species."""
    
    results_df["mean_abundance"] = (results_df["mean_group1"] + results_df["mean_group2"]) / 2
    results_df["regulation"] = results_df.apply(
        lambda row: classify_regulation(row, fc_threshold, pval_threshold),
        axis=1
    )
    
    color_map = {
        "Up-regulated": "#EA7600",
        "Down-regulated": "#262262",
        "Not significant": "#CCCCCC",
        "Not tested": "#999999",
    }
    
    fig = go.Figure()
    
    # Add data points
    for reg_type in ["Not significant", "Not tested", "Down-regulated", "Up-regulated"]:
        subset = results_df[results_df["regulation"] == reg_type]
        if not subset.empty:
            fig.add_trace(go.Scatter(
                x=subset["mean_abundance"],
                y=subset["log2fc"],
                mode="markers",
                name=reg_type,
                marker=dict(
                    color=color_map[reg_type],
                    size=6,
                    opacity=0.7 if reg_type in ["Up-regulated", "Down-regulated"] else 0.3,
                ),
                text=subset.index,
                hovertemplate="<b>%{text}</b><br>" +
                             "Mean: %{x:.2f}<br>" +
                             "log2FC: %{y:.2f}<br>" +
                             "<extra></extra>",
            ))
    
    # Add theoretical FC lines for each species
    theo_colors = {"HUMAN": "#0066CC", "YEAST": "#FF6600", "ECOLI": "#00CC66"}
    
    for species, theo_fc in theo_fc_species.items():
        fig.add_hline(
            y=theo_fc,
            line_dash="dash",
            line_color=theo_colors.get(species, "#666666"),
            line_width=2,
            annotation_text=f"{species} (theo: {theo_fc:.2f})",
            annotation_position="right",
        )
    
    # Add FC threshold lines
    fig.add_hline(y=fc_threshold, line_dash="dash", line_color="red", line_width=1.5, opacity=0.5)
    fig.add_hline(y=-fc_threshold, line_dash="dash", line_color="red", line_width=1.5, opacity=0.5)
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
    
    fig.update_layout(
        title="MA Plot with Theoretical Fold Changes",
        xaxis_title="Mean log2 Abundance",
        yaxis_title="log2 Fold Change",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#54585A"),
        height=600,
        hovermode="closest",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )
    
    return fig


def create_combined_distplot_boxplot(
    results_df: pd.DataFrame,
    fc_threshold: float,
    pval_threshold: float,
) -> go.Figure:
    from plotly.subplots import make_subplots
    
    results_df["regulation"] = results_df.apply(
        lambda row: classify_regulation(row, fc_threshold, pval_threshold),
        axis=1
    )
    
    regulation_groups = [
        ("Down-regulated", "Group 1", TF_CHART_COLORS[0]),
        ("Not significant", "Group 2", TF_CHART_COLORS[1]),
        ("Up-regulated", "Group 3", TF_CHART_COLORS[2]),
        ("Not tested", "Group 4", TF_CHART_COLORS[3]),
    ]
    
    hist_data = []
    group_labels = []
    colors = []
    
    for reg_type, label, color in regulation_groups:
        data = results_df[results_df["regulation"] == reg_type]["log2fc"].dropna()
        if len(data) > 0:
            hist_data.append(data.values)
            group_labels.append(label)
            colors.append(color)
    
    if not hist_data:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        return fig
    
    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.1, 0.9],
        column_widths=[0.7, 0.3],
        specs=[
            [{"type": "scatter"}, {"type": "box"}],
            [{"type": "histogram"}, {"type": "box"}]
        ],
        horizontal_spacing=0.05,
        vertical_spacing=0.02,
    )
    
    for i, (data, label, color) in enumerate(zip(hist_data, group_labels, colors)):
        fig.add_trace(
            go.Histogram(
                x=data,
                name=label,
                marker_color=color,
                opacity=0.6,
                nbinsx=30,
                legendgroup=label,
                showlegend=True,
            ),
            row=2, col=1
        )
        
        hist, bin_edges = np.histogram(data, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(hist, sigma=2)
        
        fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=smoothed,
                name=label,
                line=dict(color=color, width=2),
                legendgroup=label,
                showlegend=False,
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data,
                y=[i] * len(data),
                mode="markers",
                marker=dict(
                    color=color,
                    symbol="line-ns-open",
                    size=10,
                    line=dict(width=1),
                ),
                name=label,
                legendgroup=label,
                showlegend=False,
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Box(
                y=data,
                name=label,
                marker_color=color,
                legendgroup=label,
                showlegend=False,
                boxmean='sd'
            ),
            row=2, col=2
        )
    
    fig.add_vline(x=fc_threshold, line_dash="dash", line_color="red", line_width=2, row=2, col=1)
    fig.add_vline(x=-fc_threshold, line_dash="dash", line_color="red", line_width=2, row=2, col=1)
    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1, row=2, col=1)
    
    fig.add_hline(y=fc_threshold, line_dash="dash", line_color="red", line_width=2, row=2, col=2)
    fig.add_hline(y=-fc_threshold, line_dash="dash", line_color="red", line_width=2, row=2, col=2)
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, row=2, col=2)
    
    fig.update_xaxes(title_text="log2 Fold Change", row=2, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=1)
    fig.update_yaxes(title_text="log2 Fold Change", row=2, col=2)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    
    fig.update_layout(
        title="Distribution of log2 Fold Changes",
        height=700,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#54585A"),
        showlegend=True,
        legend=dict(x=1.05, y=1),
        barmode='overlay'
    )
    
    return fig


# ========== START OF PAGE ==========

st.markdown("## Differential Expression Analysis")

protein_model: MSData | None = st.session_state.get("protein_model")
protein_idx = st.session_state.get("protein_index_col")

if protein_model is None:
    st.warning("No protein data cached. Please upload data on the Data Upload page first.")
    render_navigation(back_page="pages/4_Filtering.py", next_page=None)
    render_footer()
    st.stop()

numeric_cols = protein_model.numeric_cols
condition_groups = build_condition_groups(numeric_cols)

# ========== USE FILTERED DATA ==========
filtered_data = st.session_state.get("last_filtered_data")
filter_params = st.session_state.get("last_filtered_params")

if filtered_data is not None and not filtered_data.empty:
    use_filtered = True
    working_data = filtered_data.copy()
    st.info(f"✅ Using filtered dataset: {len(working_data):,} proteins")
else:
    use_filtered = False
    working_data = protein_model.raw_filled[numeric_cols].copy()
    st.warning("⚠️ Using full dataset (no filtered data from Filtering page). Go to Filtering and click 'Store for Analysis' first.")

# ========== TOP: Dataset Overview ==========
st.markdown("### 📊 Dataset Overview")

cv_data = compute_cv_per_condition(working_data, numeric_cols)

col_o1, col_o2, col_o3, col_o4 = st.columns(4)

with col_o1:
    st.metric("Total Proteins", f"{len(working_data):,}")

with col_o2:
    if not cv_data.empty:
        cv_mean = cv_data.to_numpy().ravel()
        cv_mean = cv_mean[~np.isnan(cv_mean)].mean()
        st.metric("Mean CV%", f"{cv_mean:.1f}")
    else:
        st.metric("Mean CV%", "N/A")

with col_o3:
    if not cv_data.empty:
        cv_median = np.median(cv_data.to_numpy().ravel()[~np.isnan(cv_data.to_numpy().ravel())])
        st.metric("Median CV%", f"{cv_median:.1f}")
    else:
        st.metric("Median CV%", "N/A")

with col_o4:
    st.metric("Conditions", len(condition_groups))

if filter_params and use_filtered:
    st.caption(f"Filter info: {', '.join(filter_params.get('active_filters', []))}")

st.markdown("---")

# ========== SIDEBAR: Analysis Settings ==========
with st.sidebar:
    st.markdown("## 🎛️ Analysis Settings")
    
    st.markdown("### Condition Comparison")
    
    available_conditions = sorted(condition_groups.keys())
    
    if len(available_conditions) < 2:
        st.error("Need at least 2 conditions for differential expression analysis")
        st.stop()
    
    group1 = st.selectbox(
        "Group 1 (control/reference)",
        options=available_conditions,
        index=0,
        key="de_group1"
    )
    
    group2 = st.selectbox(
        "Group 2 (treatment/test)",
        options=[c for c in available_conditions if c != group1],
        index=0,
        key="de_group2"
    )
    
    st.caption(f"**Comparison:** {group2} vs {group1}")
    st.caption(f"**Interpretation:** Positive log2FC = higher in {group1} (reference)")  # CHANGED
    
    st.markdown("### Transformation")
    transform_key = st.selectbox(
        "Select transformation",
        options=["log2", "log10", "sqrt", "cbrt", "yeo_johnson", "quantile"],
        format_func=lambda x: {
            "log2": "log2",
            "log10": "log10",
            "sqrt": "Square root",
            "cbrt": "Cube root",
            "yeo_johnson": "Yeo-Johnson",
            "quantile": "Quantile Norm",
        }[x],
        index=0,
        key="de_transform"
    )
    
    st.info("ℹ️ **Note:** log2 is recommended for interpretable fold changes.")
    
    st.markdown("---")
    
    st.markdown("### Statistical Thresholds")
    
    fc_threshold = st.number_input(
        "log2 Fold Change threshold",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        key="de_fc_threshold",
        help="Absolute log2FC threshold (1.0 = 2-fold change)"
    )
    
    st.caption(f"Fold change: {2**fc_threshold:.2f}x")
    
    pval_threshold = st.number_input(
        "P-value threshold",
        min_value=0.001,
        max_value=0.1,
        value=0.05,
        step=0.01,
        format="%.3f",
        key="de_pval_threshold",
    )
    
    use_fdr = st.checkbox(
        "Use FDR instead of p-value",
        value=False,
        key="de_use_fdr",
    )
    
    st.markdown("---")
    
    st.markdown("### Data Requirements")
    
    min_valid = st.number_input(
        "Min valid values per group",
        min_value=2,
        max_value=10,
        value=2,
        step=1,
        key="de_min_valid",
    )

# ========== MAIN ANALYSIS ==========

# ========== MAIN ANALYSIS ==========

transform_data_selected = get_transform_data(protein_model, transform_key)

if use_filtered:
    # Use intersection of indices to avoid KeyError
    common_idx = transform_data_selected.index.intersection(working_data.index)
    transform_data_selected = transform_data_selected.loc[common_idx]
    working_data = working_data.loc[common_idx]

group1_cols = condition_groups[group1]
group2_cols = condition_groups[group2]


st.markdown(f"### Comparison: {group2} vs {group1}")

col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("Group 1 samples", len(group1_cols))
    st.caption(", ".join(group1_cols))
with col_info2:
    st.metric("Group 2 samples", len(group2_cols))
    st.caption(", ".join(group2_cols))
with col_info3:
    st.metric("Total proteins", len(transform_data_selected))

st.markdown("---")

# ========== RUN/RESTART BUTTON ==========
col_btn1, col_btn2 = st.columns([1, 3])

with col_btn1:
    run_analysis = st.button("🔬 Run DE Analysis", type="primary", key="run_de")

with col_btn2:
    if "de_results" in st.session_state:
        st.caption("✅ Analysis cached. Click button to re-run with new parameters.")

if run_analysis:
    with st.spinner("Performing t-test analysis..."):
        results_df = perform_ttest_analysis(
            transform_data_selected,
            group1_cols,
            group2_cols,
            min_valid=min_valid
        )
        
        st.session_state.de_results = results_df
        st.session_state.de_params = {
            "group1": group1,
            "group2": group2,
            "group1_cols": group1_cols,
            "group2_cols": group2_cols,
            "transform_key": transform_key,
            "fc_threshold": fc_threshold,
            "pval_threshold": pval_threshold,
            "use_fdr": use_fdr,
        }
        st.success("✅ Analysis complete!")

# Display results if available
if "de_results" in st.session_state:
    results_df = st.session_state.de_results.copy()
    params = st.session_state.de_params
    
    pval_col = "fdr" if use_fdr else "pvalue"
    results_df["regulation"] = results_df.apply(
        lambda row: classify_regulation(
            pd.Series({"log2fc": row["log2fc"], "pvalue": row[pval_col]}),
            fc_threshold,
            pval_threshold
        ),
        axis=1
    )
    
    # Summary statistics
    st.markdown("### Summary Statistics")
    
    n_up = (results_df["regulation"] == "Up-regulated").sum()
    n_down = (results_df["regulation"] == "Down-regulated").sum()
    n_ns = (results_df["regulation"] == "Not significant").sum()
    n_not_tested = (results_df["regulation"] == "Not tested").sum()
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Up-regulated", f"{n_up:,}", delta=f"{n_up/len(results_df)*100:.1f}%")
    col_s2.metric("Down-regulated", f"{n_down:,}", delta=f"{n_down/len(results_df)*100:.1f}%")
    col_s3.metric("Not significant", f"{n_ns:,}")
    col_s4.metric("Not tested", f"{n_not_tested:,}")
    
    st.markdown("---")
    
    # ========== THEORETICAL FOLD CHANGES - CONDITION-BASED COMPOSITION ==========
    st.markdown("### 🎯 Theoretical Fold Changes (Optional)")
    st.caption("Enter species composition (%) for each condition. Percentages must sum to 100.")
    
    if "de_composition" not in st.session_state:
        st.session_state.de_composition = {
            "A": {"HUMAN": 0, "YEAST": 0, "ECOLI": 0},
            "B": {"HUMAN": 0, "YEAST": 0, "ECOLI": 0},
        }
    
    # Input section
    st.markdown("#### Condition A (Reference)")
    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        a_human = st.number_input("Human %", min_value=0, max_value=100, value=st.session_state.de_composition["A"]["HUMAN"], step=1, key="a_human")
    with col_a2:
        a_yeast = st.number_input("Yeast %", min_value=0, max_value=100, value=st.session_state.de_composition["A"]["YEAST"], step=1, key="a_yeast")
    with col_a3:
        a_ecoli = st.number_input("E.coli %", min_value=0, max_value=100, value=st.session_state.de_composition["A"]["ECOLI"], step=1, key="a_ecoli")
    
    a_sum = a_human + a_yeast + a_ecoli
    if a_sum != 100:
        st.warning(f"⚠️ Condition A sum: {a_sum}% (must be 100%)")
    
    st.markdown("#### Condition B (Treatment)")
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        b_human = st.number_input("Human %", min_value=0, max_value=100, value=st.session_state.de_composition["B"]["HUMAN"], step=1, key="b_human")
    with col_b2:
        b_yeast = st.number_input("Yeast %", min_value=0, max_value=100, value=st.session_state.de_composition["B"]["YEAST"], step=1, key="b_yeast")
    with col_b3:
        b_ecoli = st.number_input("E.coli %", min_value=0, max_value=100, value=st.session_state.de_composition["B"]["ECOLI"], step=1, key="b_ecoli")
    
    b_sum = b_human + b_yeast + b_ecoli
    if b_sum != 100:
        st.warning(f"⚠️ Condition B sum: {b_sum}% (must be 100%)")
    
    # Calculate button
    col_calc1, col_calc2 = st.columns([1, 3])
    
    with col_calc1:
        calc_theo = st.button("📊 Calculate Theoretical FC", key="calc_theo")
    
    with col_calc2:
        if a_sum == 100 and b_sum == 100:
            st.caption("✅ Both conditions sum to 100%")
        else:
            st.caption("❌ Fix percentages before calculating")
    
    if calc_theo and a_sum == 100 and b_sum == 100:
        st.session_state.de_composition["A"] = {"HUMAN": a_human, "YEAST": a_yeast, "ECOLI": a_ecoli}
        st.session_state.de_composition["B"] = {"HUMAN": b_human, "YEAST": b_yeast, "ECOLI": b_ecoli}
        
        theo_fc_species = {}
        for species, pct_a in [("HUMAN", a_human), ("YEAST", a_yeast), ("ECOLI", a_ecoli)]:
            pct_b = st.session_state.de_composition["B"][species]
            if pct_a > 0 and pct_b > 0:
                log2fc = np.log2(pct_b / pct_a)
            elif pct_b > 0:
                log2fc = 10.0
            elif pct_a > 0:
                log2fc = -10.0
            else:
                log2fc = 0.0
            theo_fc_species[species] = log2fc
        
        st.session_state.de_theo_fc_species = theo_fc_species
        st.success("✅ Theoretical FC calculated!")
    
    # Display calculated theoretical FC
    if "de_theo_fc_species" in st.session_state:
        st.markdown("#### Calculated Theoretical log2 Fold Changes")
        
        theo_display = []
        for species in ["HUMAN", "YEAST", "ECOLI"]:
            fc = st.session_state.de_theo_fc_species.get(species, 0.0)
            ratio = 2 ** fc
            theo_display.append({
                "Species": species,
                "log2FC": f"{fc:.3f}",
                "Fold Change": f"{ratio:.2f}x"
            })
        
        theo_df = pd.DataFrame(theo_display)
        st.dataframe(theo_df, use_container_width=True, hide_index=True)
        
        # Calculate error rates
        if st.button("📈 Calculate Error Rates from Composition", key="calc_error_rates"):
            protein_model = st.session_state.get("protein_model")
            if protein_model and hasattr(protein_model, 'raw'):
                species_col = protein_model.species_col
                if species_col and species_col in protein_model.raw.columns:
                    true_fc_dict = {}
                    for protein_id, species in protein_model.raw[species_col].items():
                        true_fc_dict[protein_id] = st.session_state.de_theo_fc_species.get(species, 0.0)
                    
                    error_metrics = calculate_error_rates(
                        results_df.copy(),
                        true_fc_dict,
                        fc_threshold,
                        pval_threshold
                    )
                    st.session_state.de_error_metrics = error_metrics
                    st.success("✅ Error rates calculated!")
                else:
                    st.error("Species column not found in protein data")
            else:
                st.error("Protein model not available")
    
    # Display error rates if calculated
    if "de_error_metrics" in st.session_state:
        st.markdown("---")
        st.markdown("### 📈 Error Rate Analysis")
        
        metrics = st.session_state.de_error_metrics
        
        protein_model = st.session_state.get("protein_model")
        if protein_model and hasattr(protein_model, 'raw'):
            species_col = protein_model.species_col
            if species_col and species_col in protein_model.raw.columns:
                species_col_data = protein_model.raw[species_col].to_dict()
                
                true_fc_dict = {}
                for protein_id, species in species_col_data.items():
                    true_fc_dict[protein_id] = st.session_state.de_theo_fc_species.get(species, 0.0)
                
                species_metrics_df = calculate_species_rmse(
                    results_df.copy(),
                    true_fc_dict,
                    species_col_data,
                    fc_threshold,
                    pval_threshold
                )
        
        # Compact layout: 3 columns
        col_comp, col_theo, col_perf = st.columns(3)
        
        with col_comp:
            st.markdown("**Input Composition**")
            comp_data = []
            for condition in ["A", "B"]:
                comp = st.session_state.de_composition[condition]
                comp_data.append([
                    condition,
                    comp["HUMAN"],
                    comp["YEAST"],
                    comp["ECOLI"]
                ])
            comp_df = pd.DataFrame(comp_data, columns=["Cond", "H%", "Y%", "E%"])
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        with col_theo:
            st.markdown("**Theoretical FC by Species**")
            theo_data = []
            for species in ["HUMAN", "YEAST", "ECOLI"]:
                fc = st.session_state.de_theo_fc_species.get(species, 0.0)
                direction = "↑" if fc > 0.2 else "↓" if fc < -0.2 else "→"
                theo_data.append([species, f"{fc:.2f}", direction])
            theo_df = pd.DataFrame(theo_data, columns=["Species", "log2FC", "Dir"])
            st.dataframe(theo_df, use_container_width=True, hide_index=True)
        
        with col_perf:
            st.markdown("**Overall Performance**")
            perf_data = [
                ["Sensitivity", f"{metrics['sensitivity']:.1%}"],
                ["Specificity", f"{metrics['specificity']:.1%}"],
                ["Precision", f"{metrics['precision']:.1%}"],
                ["FPR", f"{metrics['FPR']:.1%}"],
            ]
            perf_df = pd.DataFrame(perf_data, columns=["Metric", "Value"])
            st.dataframe(perf_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Confusion matrix and per-species metrics
        col_cm1, col_cm2, col_cm3 = st.columns([1, 1.5, 2])
        
        with col_cm1:
            st.markdown("**Confusion Matrix**")
            cm_data = [
                ["TP", metrics["TP"]],
                ["FP", metrics["FP"]],
                ["TN", metrics["TN"]],
                ["FN", metrics["FN"]],
            ]
            cm_df = pd.DataFrame(cm_data, columns=["", "Count"])
            st.dataframe(cm_df, use_container_width=True, hide_index=True)
        
        with col_cm2:
            st.markdown("**Per-Species RMSE**")
            st.dataframe(species_metrics_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Visualization tabs
    tab_volcano, tab_ma, tab_dist, tab_table = st.tabs(["Volcano Plot", "MA Plot", "Distribution + Boxplot", "Results Table"])
    
    with tab_volcano:
        st.markdown("### Volcano Plot")
        fig_volcano = create_volcano_plot(
            results_df.copy(),
            fc_threshold,
            pval_threshold,
            title=f"Volcano Plot: {params['group2']} vs {params['group1']}"
        )
        st.plotly_chart(fig_volcano, use_container_width=True)
    
    with tab_ma:
        st.markdown("### MA Plot")
        fig_ma = create_ma_plot_with_theoretical(
            results_df.copy(),
            fc_threshold,
            pval_threshold,
            st.session_state.get("de_theo_fc_species", {})
        )
        st.plotly_chart(fig_ma, use_container_width=True)
    
    with tab_dist:
        st.markdown("### Distribution + Boxplot")
        st.caption("Left: Histogram with KDE curves | Right: Boxplots with mean ± SD")
        fig_combined = create_combined_distplot_boxplot(
            results_df.copy(),
            fc_threshold,
            pval_threshold
        )
        st.plotly_chart(fig_combined, use_container_width=True)
    
    with tab_table:
        st.markdown("### Results Table")
        
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            show_filter = st.selectbox(
                "Show proteins",
                options=["All", "Significant only", "Up-regulated", "Down-regulated"],
                index=0,
                key="de_table_filter"
            )
        
        with col_f2:
            sort_by = st.selectbox(
                "Sort by",
                options=["P-value", "log2FC", "Mean abundance"],
                index=0,
                key="de_table_sort"
            )
        
        if show_filter == "Significant only":
            display_df = results_df[results_df["regulation"].isin(["Up-regulated", "Down-regulated"])]
        elif show_filter == "Up-regulated":
            display_df = results_df[results_df["regulation"] == "Up-regulated"]
        elif show_filter == "Down-regulated":
            display_df = results_df[results_df["regulation"] == "Down-regulated"]
        else:
            display_df = results_df
        
        if sort_by == "P-value":
            display_df = display_df.sort_values(pval_col)
        elif sort_by == "log2FC":
            display_df = display_df.sort_values("log2fc", ascending=False, key=abs)
        else:
            display_df["mean_abundance"] = (display_df["mean_group1"] + display_df["mean_group2"]) / 2
            display_df = display_df.sort_values("mean_abundance", ascending=False)
        
        display_cols = [
            "log2fc",
            "pvalue",
            "fdr",
            "mean_group1",
            "mean_group2",
            "regulation",
            "n_group1",
            "n_group2",
        ]
        
        styled_df = display_df[display_cols].style.format({
            "log2fc": "{:.3f}",
            "pvalue": "{:.2e}",
            "fdr": "{:.2e}",
            "mean_group1": "{:.2f}",
            "mean_group2": "{:.2f}",
            "n_group1": "{:.0f}",
            "n_group2": "{:.0f}",
        }).background_gradient(
            subset=["log2fc"],
            cmap="RdBu_r",
            vmin=-3,
            vmax=3
        )
        
        st.dataframe(styled_df, use_container_width=True, height=600)
        
        st.caption(f"Showing {len(display_df):,} of {len(results_df):,} proteins")
    
    st.markdown("---")
    
    # Export results
    st.markdown("### Export Results")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        csv_all = results_df.to_csv(index=True)
        st.download_button(
            label="📥 Download All Results (CSV)",
            data=csv_all,
            file_name=f"de_results_{params['group2']}_vs_{params['group1']}.csv",
            mime="text/csv",
        )
    
    with col_exp2:
        sig_df = results_df[results_df["regulation"].isin(["Up-regulated", "Down-regulated"])]
        csv_sig = sig_df.to_csv(index=True)
        st.download_button(
            label="📥 Download Significant Only (CSV)",
            data=csv_sig,
            file_name=f"de_significant_{params['group2']}_vs_{params['group1']}.csv",
            mime="text/csv",
        )

else:
    st.info("👆 Click 'Run DE Analysis' to start the analysis with current parameters")

render_navigation(back_page="pages/4_Filtering.py", next_page=None)
render_footer()
