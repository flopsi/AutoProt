import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

from components import inject_custom_css, render_header, render_navigation, render_footer, COLORS
from dataclasses import dataclass
from typing import List

st.set_page_config(
    page_title="EDA | Thermo Fisher Scientific",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
render_header()

# ----------------- data model (same as upload) -----------------
@dataclass
class MSData:
    original: pd.DataFrame
    filled: pd.DataFrame
    log2_filled: pd.DataFrame
    numeric_cols: List[str]


# Thermo Fisher chart colors
TF_CHART_COLORS = ["#262262", "#A6192E", "#EA7600", "#F1B434", "#B5BD00", "#9BD3DD"]


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
        # fall back to prefix-letter + digits (A1, B2,...)
        if len(col) >= 1 and col[0].isalpha():
            head, tail = col[0], col[1:]
            return (head, int(tail) if tail.isdigit() else 0)
        return (col, 0)
    return sorted(cols, key=sort_key)


# --- Transformation functions (unchanged) ---
def transform_raw(x):
    return x


def transform_log2(x):
    return np.log2(x + 1)


def transform_log10(x):
    return np.log10(x + 1)


def transform_sqrt(x):
    return np.sqrt(x)


def transform_cbrt(x):
    return np.cbrt(x)


def transform_yeojohnson(x):
    try:
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        return pt.fit_transform(x.reshape(-1, 1)).flatten()
    except Exception:
        return x


def transform_quantile(x):
    try:
        qt = QuantileTransformer(output_distribution="normal", random_state=42)
        return qt.fit_transform(x.reshape(-1, 1)).flatten()
    except Exception:
        return x


TRANSFORMATIONS = {
    "Raw": transform_raw,
    "Log2": transform_log2,
    "Log10": transform_log10,
    "Square root": transform_sqrt,
    "Cube root": transform_cbrt,
    "Yeo-Johnson": transform_yeojohnson,
    "Quantile": transform_quantile,
}


@st.cache_data
def compute_normality_stats(values: np.ndarray) -> dict:
    clean = values[np.isfinite(values)]
    if len(clean) < 20:
        return {"kurtosis": np.nan, "skewness": np.nan, "W": np.nan, "p": np.nan}
    if len(clean) > 5000:
        sample = np.random.choice(clean, 5000, replace=False)
    else:
        sample = clean
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
    all_values = all_values[all_values > 0]
    results = []
    for name, func in TRANSFORMATIONS.items():
        transformed = func(all_values.copy())
        stats_dict = compute_normality_stats(transformed)
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
def create_intensity_heatmap(
    df_json: str,
    index_col: str | None,
    numeric_cols: list[str],
    transform_name: str = "Log2",
) -> go.Figure:
    df = pd.read_json(df_json)

    if index_col and index_col in df.columns:
        labels = df[index_col].apply(parse_protein_group).tolist()
    else:
        labels = [f"Row {i}" for i in range(len(df))]

    sorted_cols = sort_columns_by_condition(numeric_cols)
    data = df[sorted_cols].values.astype(float)

    transform_func = TRANSFORMATIONS.get(transform_name, transform_log2)

    if transform_name in ["Yeo-Johnson", "Quantile"]:
        transformed = np.zeros_like(data, dtype=float)
        for i in range(data.shape[1]):
            col_data = data[:, i]
            transformed[:, i] = transform_func(col_data)
    else:
        transformed = transform_func(data)

    fig = go.Figure(
        data=go.Heatmap(
            z=transformed,
            x=sorted_cols,
            y=labels,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=transform_name),
            hovertemplate="Protein: %{y}<br>Sample: %{x}<br>Value: %{z:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Intensity distribution ({transform_name})",
        xaxis_title="Samples",
        yaxis_title="",
        height=600,
        yaxis=dict(tickfont=dict(size=8)),
        xaxis=dict(tickangle=45),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#54585A"),
    )
    return fig


@st.cache_data
def create_missing_distribution_chart(df_json: str, numeric_cols: list[str], label: str) -> go.Figure:
    df = pd.read_json(df_json)
    # In your original code, "missing" was encoded as 1
    missing_per_row = (df[numeric_cols] == 1).sum(axis=1)
    total_rows = len(df)
    max_missing = len(numeric_cols)

    counts = []
    labels_x = []
    for i in range(max_missing + 1):
        count = (missing_per_row == i).sum()
        pct = 100 * count / total_rows
        counts.append(pct)
        labels_x.append(str(i))

    fig = go.Figure(
        data=go.Bar(
            x=labels_x,
            y=counts,
            marker_color="#262262",  # NAVY
            hovertemplate="Missing: %{x}<br>Percent: %{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Missing values per {label}",
        xaxis_title="Number of missing values",
        yaxis_title="% of total",
        height=400,
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#54585A"),
        bargap=0.2,
    )
    return fig


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


with tab_protein:

    @st.fragment
    def protein_analysis():
        if protein_model is not None:
            df = protein_model.log2_filled  # use filled/log2-cleaned data
            numeric_cols = protein_model.numeric_cols
            st.caption(f"**{len(df):,} proteins** × **{len(numeric_cols)} samples**")

            df_json = df.to_json()

            st.markdown("### Normality analysis")
            st.caption("Testing which transformation best normalizes intensity distributions")

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
                f"Recommended: **{best_transform}** "
                f"(Shapiro W = {stats_df.loc[best_idx, 'Shapiro W']:.4f})"
            )

            selected_transform = st.selectbox(
                "Select transformation for visualization",
                options=list(TRANSFORMATIONS.keys()),
                index=list(TRANSFORMATIONS.keys()).index(best_transform),
                key="protein_transform",
            )
            st.session_state["protein_selected_transform"] = selected_transform

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                fig_heat = create_intensity_heatmap(
                    df_json, protein_idx, numeric_cols, selected_transform
                )
                st.plotly_chart(fig_heat, use_container_width=True)
            with col2:
                fig_bar = create_missing_distribution_chart(df_json, numeric_cols, "protein groups")
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No protein data uploaded yet")

    protein_analysis()


with tab_peptide:

    @st.fragment
    def peptide_analysis():
        if peptide_model is not None:
            df = peptide_model.log2_filled
            numeric_cols = peptide_model.numeric_cols
            st.caption(f"**{len(df):,} peptides** × **{len(numeric_cols)} samples**")

            df_json = df.to_json()

            st.markdown("### Normality analysis")
            st.caption("Testing which transformation best normalizes intensity distributions")

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
                f"Recommended: **{best_transform}** "
                f"(Shapiro W = {stats_df.loc[best_idx, 'Shapiro W']:.4f})"
            )

            selected_transform = st.selectbox(
                "Select transformation for visualization",
                options=list(TRANSFORMATIONS.keys()),
                index=list(TRANSFORMATIONS.keys()).index(best_transform),
                key="peptide_transform",
            )
            st.session_state["peptide_selected_transform"] = selected_transform

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                fig_heat = create_intensity_heatmap(
                    df_json, peptide_idx, numeric_cols, selected_transform
                )
                st.plotly_chart(fig_heat, use_container_width=True)
            with col2:
                fig_bar = create_missing_distribution_chart(df_json, numeric_cols, "peptides")
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No peptide data uploaded yet")

    peptide_analysis()

render_navigation(back_page="pages/1_Data_Upload.py", next_page=None)
render_footer()
