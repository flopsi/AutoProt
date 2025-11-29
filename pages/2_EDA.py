import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
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


# --- Transformation functions ---
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

def transform_boxcox(x):
    # Box-Cox requires positive values
    x_pos = x[x > 0]
    if len(x_pos) < 3:
        return x
    try:
        transformed, _ = stats.boxcox(x_pos)
        return transformed
    except:
        return x_pos


TRANSFORMATIONS = {
    "Raw": transform_raw,
    "Log2": transform_log2,
    "Log10": transform_log10,
    "Square root": transform_sqrt,
    "Cube root": transform_cbrt,
    "Box-Cox": transform_boxcox,
}


@st.cache_data
def compute_normality_stats(values: np.ndarray) -> dict:
    """Compute normality statistics for a 1D array."""
    # Remove NaN and inf
    clean = values[np.isfinite(values)]

    if len(clean) < 20:
        return {"kurtosis": np.nan, "skewness": np.nan, "W": np.nan, "p": np.nan}

    # Sample for Shapiro (max 5000)
    if len(clean) > 5000:
        sample = np.random.choice(clean, 5000, replace=False)
    else:
        sample = clean

    try:
        W, p = stats.shapiro(sample)
    except:
        W, p = np.nan, np.nan

    return {
        "kurtosis": stats.kurtosis(clean),
        "skewness": stats.skew(clean),
        "W": W,
        "p": p
    }


@st.cache_data
def analyze_transformations(df_json: str, numeric_cols: list[str]) -> pd.DataFrame:
    """Analyze all transformations and return stats DataFrame."""
    df = pd.read_json(df_json)

    # Flatten all numeric values
    all_values = df[numeric_cols].values.flatten()
    all_values = all_values[all_values > 0]  # Remove zeros for transforms

    results = []
    for name, func in TRANSFORMATIONS.items():
        transformed = func(all_values.copy())
        stats_dict = compute_normality_stats(transformed)
        results.append({
            "Transformation": name,
            "Kurtosis": stats_dict["kurtosis"],
            "Skewness": stats_dict["skewness"],
            "Shapiro W": stats_dict["W"],
            "Shapiro p": stats_dict["p"]
        })

    return pd.DataFrame(results)


@st.cache_data
def create_intensity_heatmap(df_json: str, index_col: str, numeric_cols: list[str], 
                              transform_name: str = "Log2") -> go.Figure:
    df = pd.read_json(df_json)

    if index_col and index_col in df.columns:
        labels = df[index_col].apply(parse_protein_group).tolist()
    else:
        labels = [f"Row {i}" for i in range(len(df))]

    sorted_cols = sort_columns_by_condition(numeric_cols)

    # Apply selected transformation
    transform_func = TRANSFORMATIONS.get(transform_name, transform_log2)
    intensity_transformed = df[sorted_cols].apply(lambda x: transform_func(x))

    fig = go.Figure(data=go.Heatmap(
        z=intensity_transformed.values,
        x=sorted_cols,
        y=labels,
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title=transform_name),
        hovertemplate="Protein: %{y}<br>Sample: %{x}<br>Value: %{z:.2f}<extra></extra>"
    ))

    fig.update_layout(
        title=f"Intensity distribution ({transform_name})",
        xaxis_title="Samples",
        yaxis_title="",
        height=600,
        yaxis=dict(tickfont=dict(size=8)),
        xaxis=dict(tickangle=45),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#54585A")
    )
    return fig


@st.cache_data
def create_missing_distribution_chart(df_json: str, numeric_cols: list[str], 
                                       label: str = "protein groups") -> go.Figure:
    df = pd.read_json(df_json)

    missing_per_row = (df[numeric_cols] == 1).sum(axis=1)
    total_rows = len(df)
    max_missing = len(numeric_cols)

    counts = []
    labels = []
    for i in range(max_missing + 1):
        count = (missing_per_row == i).sum()
        pct = 100 * count / total_rows
        counts.append(pct)
        labels.append(str(i))

    fig = go.Figure(data=go.Bar(
        x=labels,
        y=counts,
        marker_color="#262262",
        hovertemplate="Missing: %{x}<br>Percent: %{y:.1f}%<extra></extra>"
    ))

    fig.update_layout(
        title=f"Missing values per {label}",
        xaxis_title="Number of missing values",
        yaxis_title="% of total",
        height=400,
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#54585A"),
        bargap=0.2
    )
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
    @st.fragment
    def protein_analysis():
        if protein_data is not None:
            numeric_cols = get_numeric_cols(protein_data)
            st.caption(f"**{len(protein_data):,} proteins** × **{len(numeric_cols)} samples**")

            df_json = protein_data.to_json()

            # --- Normality Analysis ---
            st.markdown("### Normality analysis")
            st.caption("Testing which transformation best normalizes intensity distributions")

            stats_df = analyze_transformations(df_json, numeric_cols)

            # Find best transformation (highest W)
            best_idx = stats_df["Shapiro W"].idxmax()
            best_transform = stats_df.loc[best_idx, "Transformation"]

            # Display stats table
            def highlight_best(row):
                if row["Transformation"] == best_transform:
                    return ["background-color: #B5BD00; color: white"] * len(row)
                return [""] * len(row)

            styled_df = stats_df.style.apply(highlight_best, axis=1).format({
                "Kurtosis": "{:.3f}",
                "Skewness": "{:.3f}",
                "Shapiro W": "{:.4f}",
                "Shapiro p": "{:.2e}"
            })
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

            st.success(f"Recommended transformation: **{best_transform}** (highest Shapiro W = {stats_df.loc[best_idx, 'Shapiro W']:.4f})")

            # Transformation selector
            selected_transform = st.selectbox(
                "Select transformation for visualization",
                options=list(TRANSFORMATIONS.keys()),
                index=list(TRANSFORMATIONS.keys()).index(best_transform),
                key="protein_transform"
            )

            # Store selected transformation
            st.session_state["protein_selected_transform"] = selected_transform

            st.markdown("---")

            # --- Visualizations ---
            col1, col2 = st.columns(2)

            with col1:
                fig_heat = create_intensity_heatmap(df_json, protein_idx, numeric_cols, selected_transform)
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
        if peptide_data is not None:
            numeric_cols = get_numeric_cols(peptide_data)
            st.caption(f"**{len(peptide_data):,} peptides** × **{len(numeric_cols)} samples**")

            df_json = peptide_data.to_json()

            # --- Normality Analysis ---
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
                "Kurtosis": "{:.3f}",
                "Skewness": "{:.3f}",
                "Shapiro W": "{:.4f}",
                "Shapiro p": "{:.2e}"
            })
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

            st.success(f"Recommended transformation: **{best_transform}** (highest Shapiro W = {stats_df.loc[best_idx, 'Shapiro W']:.4f})")

            selected_transform = st.selectbox(
                "Select transformation for visualization",
                options=list(TRANSFORMATIONS.keys()),
                index=list(TRANSFORMATIONS.keys()).index(best_transform),
                key="peptide_transform"
            )

            st.session_state["peptide_selected_transform"] = selected_transform

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                fig_heat = create_intensity_heatmap(df_json, peptide_idx, numeric_cols, selected_transform)
                st.plotly_chart(fig_heat, use_container_width=True)

            with col2:
                fig_bar = create_missing_distribution_chart(df_json, numeric_cols, "peptides")
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No peptide data uploaded yet")

    peptide_analysis()

render_navigation(back_page="pages/1_Data_Upload.py", next_page=None)
render_footer()
