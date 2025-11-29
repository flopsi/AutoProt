import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from components import inject_custom_css, render_header, render_navigation, render_footer, COLORS

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
        if "_" in col:
            parts = col.rsplit("_", 1)
            return (parts[0], int(parts[1]) if parts[1].isdigit() else 0)
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


# --- Transformation functions ---
def transform_log2(x):
    return np.log2(np.maximum(x, 1))


TRANSFORMATIONS = {
    "Raw": lambda x: x,
    "Log2": transform_log2,
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
        z=intensity_log2,
        x=sorted_cols,
        y=labels,
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title="log2"),
        hovertemplate="Protein: %{y}<br>Sample: %{x}<br>log2: %{z:.2f}<extra></extra>"
    ))

    fig.update_layout(
        title="Intensity distribution (log2)",
        xaxis_title="Samples",
        yaxis_title="",
        height=500,
        yaxis=dict(tickfont=dict(size=8)),
        xaxis=dict(tickangle=45),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#54585A")
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
        x=[str(i) for i in range(max_missing + 1)],
        y=counts,
        marker_color="#262262",
        hovertemplate="Missing: %{x}<br>Percent: %{y:.1f}%<extra></extra>"
    ))

    fig.update_layout(
        title=f"Missing values per {label}",
        xaxis_title="Number of missing values",
        yaxis_title="% of total",
        height=350,
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#54585A"),
        bargap=0.2
    )
    return fig


def create_condition_violin_plot(df: pd.DataFrame, numeric_cols: list[str], 
                                  cond_a: str, cond_b: str):
    """Create split violin plot with seaborn comparing two conditions."""
    groups = get_condition_groups(numeric_cols)
    cols_a = groups.get(cond_a, [])
    cols_b = groups.get(cond_b, [])

    if not cols_a or not cols_b:
        return None

    # Get mean per condition per row, log2 transform
    mean_a = np.log2(np.maximum(df[cols_a].mean(axis=1).values, 1))
    mean_b = np.log2(np.maximum(df[cols_b].mean(axis=1).values, 1))

    # Create long-form dataframe for seaborn
    plot_df = pd.DataFrame({
        'log2_intensity': np.concatenate([mean_a, mean_b]),
        'Condition': [cond_a] * len(mean_a) + [cond_b] * len(mean_b)
    })

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Split violin with inner box
    sns.violinplot(
        data=plot_df,
        x='Condition',
        y='log2_intensity',
        hue='Condition',
        split=True,
        inner='box',
        palette={cond_a: '#262262', cond_b: '#EA7600'},
        ax=ax,
        legend=False
    )

    ax.set_ylabel('log2(mean intensity)')
    ax.set_xlabel('')
    ax.set_title(f'Condition comparison: {cond_a} vs {cond_b}')

    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    return fig


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

        # --- Static heatmap and missing values ---
        col1, col2 = st.columns([2, 1])
        with col1:
            fig_heat = create_intensity_heatmap(df_json, protein_idx, numeric_cols)
            st.plotly_chart(fig_heat, use_container_width=True)
        with col2:
            fig_bar = create_missing_distribution_chart(df_json, numeric_cols, "protein groups")
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        # --- Normality analysis fragment ---
        @st.fragment
        def protein_normality():
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

        protein_normality()

        st.markdown("---")

        # --- Condition comparison fragment ---
        @st.fragment
        def protein_comparison():
            st.markdown("### Condition comparison")

            groups = get_condition_groups(numeric_cols)
            conditions = sorted(groups.keys())

            if len(conditions) < 2:
                st.warning("Need at least 2 conditions for comparison")
                return

            col1, col2 = st.columns(2)
            with col1:
                cond_a = st.selectbox("Condition A", conditions, index=0, key="prot_cond_a")
            with col2:
                cond_b = st.selectbox("Condition B", conditions, index=min(1, len(conditions)-1), key="prot_cond_b")

            if cond_a != cond_b:
                fig = create_condition_violin_plot(protein_data, numeric_cols, cond_a, cond_b)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.warning("Select different conditions to compare")

        protein_comparison()
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

        @st.fragment
        def peptide_normality():
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

        peptide_normality()

        st.markdown("---")

        @st.fragment
        def peptide_comparison():
            st.markdown("### Condition comparison")

            groups = get_condition_groups(numeric_cols)
            conditions = sorted(groups.keys())

            if len(conditions) < 2:
                st.warning("Need at least 2 conditions for comparison")
                return

            col1, col2 = st.columns(2)
            with col1:
                cond_a = st.selectbox("Condition A", conditions, index=0, key="pep_cond_a")
            with col2:
                cond_b = st.selectbox("Condition B", conditions, index=min(1, len(conditions)-1), key="pep_cond_b")

            if cond_a != cond_b:
                fig = create_condition_violin_plot(peptide_data, numeric_cols, cond_a, cond_b)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.warning("Select different conditions to compare")

        peptide_comparison()
    else:
        st.info("No peptide data uploaded yet")

render_navigation(back_page="pages/1_Data_Upload.py", next_page=None)
render_footer()
