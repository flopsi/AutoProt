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
        log2fc = mean2 - mean1
        
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


def create_ma_plot(
    results_df: pd.DataFrame,
    fc_threshold: float,
    pval_threshold: float,
) -> go.Figure:
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
    
    fig.add_hline(y=fc_threshold, line_dash="dash", line_color="red")
    fig.add_hline(y=-fc_threshold, line_dash="dash", line_color="red")
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
    
    fig.update_layout(
        title="MA Plot",
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

# [Rest of your page code continues below...]


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
# Check for filtered data from previous page
filtered_data = st.session_state.get("last_filtered_data")
filter_params = st.session_state.get("last_filtered_params")

if filtered_data is not None and not filtered_data.empty:
    use_filtered = True
    working_data = filtered_data.copy()
    st.info(f"✅ Using filtered dataset: {len(working_data):,} proteins")
else:
    # Fallback to full raw data
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
    
    # Condition selection
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
    st.caption(f"**Interpretation:** Positive log2FC = higher in {group2}")
    
    st.markdown("---")
    
    # Transformation
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
    
    # Statistical thresholds
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
    
    # Minimum valid values
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

# Get transformed data for selected transformation
transform_data_selected = get_transform_data(protein_model, transform_key)

# Apply filter if using filtered data
if use_filtered:
    transform_data_selected = transform_data_selected.loc[working_data.index]

# Get columns for each group
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
    
    # Classify results
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
    
    # ========== THEORETICAL FOLD CHANGES - SIMPLE INPUT ==========
    st.markdown("### 🎯 Theoretical Fold Changes (Optional)")
    st.caption("Enter expected log2 fold changes for spike-in proteins to calculate error rates")
    
    col_theo1, col_theo2, col_theo3 = st.columns(3)
    
    with col_theo1:
        theo_protein_id = st.text_input(
            "Protein ID",
            placeholder="e.g., HUMAN_P12345",
            key="theo_protein_id"
        )
    
    with col_theo2:
        theo_log2fc = st.number_input(
            "log2 Fold Change",
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.5,
            key="theo_log2fc"
        )
    
    with col_theo3:
        add_theo = st.button("➕ Add", key="add_theo")
    
    # Store theoretical values in session state
    if "de_theoretical" not in st.session_state:
        st.session_state.de_theoretical = {}
    
    if add_theo and theo_protein_id.strip():
        st.session_state.de_theoretical[theo_protein_id.strip()] = theo_log2fc
        st.success(f"✅ Added {theo_protein_id}: log2FC={theo_log2fc:.1f}")
    
    # Display added theoretical values
    if st.session_state.de_theoretical:
        st.markdown("#### Added Theoretical Values")
        
        theo_display = []
        for pid, fc in st.session_state.de_theoretical.items():
            theo_display.append({"Protein ID": pid, "log2FC": f"{fc:.2f}"})
        
        theo_df = pd.DataFrame(theo_display)
        
        col_theo_table, col_theo_clear = st.columns([3, 1])
        
        with col_theo_table:
            st.dataframe(theo_df, use_container_width=True, hide_index=True)
        
        with col_theo_clear:
            if st.button("🗑️ Clear All", key="clear_theo"):
                st.session_state.de_theoretical = {}
                st.rerun()
        
        # Calculate error rates
        if st.button("📊 Calculate Error Rates", key="calc_error_rates"):
            error_metrics = calculate_error_rates(
                results_df.copy(),
                st.session_state.de_theoretical,
                fc_threshold,
                pval_threshold
            )
            st.session_state.de_error_metrics = error_metrics
    
    # Display error rates if calculated
    if "de_error_metrics" in st.session_state:
        st.markdown("---")
        st.markdown("### 📈 Error Rate Analysis")
        
        metrics = st.session_state.de_error_metrics
        
        # Confusion matrix
        col_cm1, col_cm2 = st.columns(2)
        
        with col_cm1:
            st.markdown("#### Confusion Matrix")
            cm_df = pd.DataFrame({
                "Predicted Positive": [metrics["TP"], metrics["FP"]],
                "Predicted Negative": [metrics["FN"], metrics["TN"]],
            }, index=["Actual Positive", "Actual Negative"])
            st.dataframe(cm_df, use_container_width=True)
        
        with col_cm2:
            st.markdown("#### Performance Metrics")
            perf_df = pd.DataFrame({
                "Metric": ["Sensitivity", "Specificity", "Precision", "FPR", "FNR"],
                "Value": [
                    f"{metrics['sensitivity']:.1%}",
                    f"{metrics['specificity']:.1%}",
                    f"{metrics['precision']:.1%}",
                    f"{metrics['FPR']:.1%}",
                    f"{metrics['FNR']:.1%}",
                ]
            })
            st.dataframe(perf_df, hide_index=True, use_container_width=True)
    
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
        fig_ma = create_ma_plot(
            results_df.copy(),
            fc_threshold,
            pval_threshold
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
        
        # Filter options
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
        
        # Apply filter
        if show_filter == "Significant only":
            display_df = results_df[results_df["regulation"].isin(["Up-regulated", "Down-regulated"])]
        elif show_filter == "Up-regulated":
            display_df = results_df[results_df["regulation"] == "Up-regulated"]
        elif show_filter == "Down-regulated":
            display_df = results_df[results_df["regulation"] == "Down-regulated"]
        else:
            display_df = results_df
        
        # Sort
        if sort_by == "P-value":
            display_df = display_df.sort_values(pval_col)
        elif sort_by == "log2FC":
            display_df = display_df.sort_values("log2fc", ascending=False, key=abs)
        else:
            display_df["mean_abundance"] = (display_df["mean_group1"] + display_df["mean_group2"]) / 2
            display_df = display_df.sort_values("mean_abundance", ascending=False)
        
        # Display table
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
