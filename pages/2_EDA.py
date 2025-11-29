import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from components import inject_custom_css, render_navbar, render_footer, COLORS

st.set_page_config(
    page_title="EDA | Thermo Fisher Scientific",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

inject_custom_css()
render_navbar(active_page="eda")


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


def create_intensity_heatmap(df: pd.DataFrame, index_col: str,
                              numeric_cols: list[str]) -> go.Figure:
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
        colorbar=dict(title="log2(intensity)"),
        hovertemplate="Protein: %{y}<br>Sample: %{x}<br>log2: %{z:.2f}<extra></extra>"
    ))

    fig.update_layout(
        title="Intensity distribution (log2) — missing values appear as 0",
        xaxis_title="Samples",
        yaxis_title="Protein groups",
        height=max(400, len(labels) * 2),
        yaxis=dict(tickfont=dict(size=8)),
        xaxis=dict(tickangle=45),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color=COLORS["gray"])
    )
    return fig


st.markdown("## Exploratory data analysis")
st.caption("Visualize intensity distributions — missing values (imputed as 1) appear as 0 after log2 transform")

protein_data = st.session_state.get("protein_data")
peptide_data = st.session_state.get("peptide_data")
protein_idx = st.session_state.get("protein_index_col")
peptide_idx = st.session_state.get("peptide_index_col")

if protein_data is None and peptide_data is None:
    st.warning("No data cached. Please upload data on the Data Upload page first.")
    st.stop()

tab_protein, tab_peptide = st.tabs(["Protein data", "Peptide data"])

with tab_protein:
    if protein_data is not None:
        numeric_cols = get_numeric_cols(protein_data)
        st.caption(f"**{len(protein_data):,} proteins** × **{len(numeric_cols)} samples**")

        max_rows = st.slider("Max proteins to display", 50, min(2000, len(protein_data)),
                            min(500, len(protein_data)), key="protein_slider")
        display_df = protein_data.head(max_rows)

        fig = create_intensity_heatmap(display_df, protein_idx, numeric_cols)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            median_intensity = protein_data[numeric_cols].median().median()
            st.metric("Median intensity", f"{median_intensity:,.0f}")
        with col2:
            zero_count = (protein_data[numeric_cols] == 1).sum().sum()
            total = protein_data[numeric_cols].size
            st.metric("Missing values", f"{zero_count:,} ({100*zero_count/total:.1f}%)")
    else:
        st.info("No protein data uploaded yet")

with tab_peptide:
    if peptide_data is not None:
        numeric_cols = get_numeric_cols(peptide_data)
        st.caption(f"**{len(peptide_data):,} peptides** × **{len(numeric_cols)} samples**")

        max_rows = st.slider("Max peptides to display", 50, min(2000, len(peptide_data)),
                            min(500, len(peptide_data)), key="peptide_slider")
        display_df = peptide_data.head(max_rows)

        fig = create_intensity_heatmap(display_df, peptide_idx, numeric_cols)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            median_intensity = peptide_data[numeric_cols].median().median()
            st.metric("Median intensity", f"{median_intensity:,.0f}")
        with col2:
            zero_count = (peptide_data[numeric_cols] == 1).sum().sum()
            total = peptide_data[numeric_cols].size
            st.metric("Missing values", f"{zero_count:,} ({100*zero_count/total:.1f}%)")
    else:
        st.info("No peptide data uploaded yet")

render_footer()
