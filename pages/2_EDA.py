import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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


@st.cache_data
def create_intensity_heatmap(df_json: str, index_col: str, numeric_cols: list[str]) -> go.Figure:
    """Cached heatmap creation."""
    df = pd.read_json(df_json)

    if index_col and index_col in df.columns:
        labels = df[index_col].apply(parse_protein_group).tolist()
    else:
        labels = [f"Row {i}" for i in range(len(df))]

    sorted_cols = sort_columns_by_condition(numeric_cols)
    intensity_log2 = np.log2(df[sorted_cols])

    fig = go.Figure(data=go.Heatmap(
        z=intensity_log2.values,
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
    """Cached bar chart creation."""
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
        height=600,
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#54585A"),
        bargap=0.2
    )
    return fig


st.markdown("## Exploratory data analysis")
st.caption("Intensity distributions — missing values (imputed as 1) appear as 0 after log2 transform")

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
    def protein_charts():
        if protein_data is not None:
            numeric_cols = get_numeric_cols(protein_data)
            st.caption(f"**{len(protein_data):,} proteins** × **{len(numeric_cols)} samples**")

            # Convert to JSON for caching (DataFrames aren't hashable)
            df_json = protein_data.to_json()

            col1, col2 = st.columns(2)

            with col1:
                fig_heat = create_intensity_heatmap(df_json, protein_idx, numeric_cols)
                st.plotly_chart(fig_heat, use_container_width=True)

            with col2:
                fig_bar = create_missing_distribution_chart(df_json, numeric_cols, "protein groups")
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No protein data uploaded yet")

    protein_charts()

with tab_peptide:
    @st.fragment
    def peptide_charts():
        if peptide_data is not None:
            numeric_cols = get_numeric_cols(peptide_data)
            st.caption(f"**{len(peptide_data):,} peptides** × **{len(numeric_cols)} samples**")

            df_json = peptide_data.to_json()

            col1, col2 = st.columns(2)

            with col1:
                fig_heat = create_intensity_heatmap(df_json, peptide_idx, numeric_cols)
                st.plotly_chart(fig_heat, use_container_width=True)

            with col2:
                fig_bar = create_missing_distribution_chart(df_json, numeric_cols, "peptides")
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No peptide data uploaded yet")

    peptide_charts()

render_navigation(back_page="pages/1_Data_Upload.py", next_page=None)
render_footer()
