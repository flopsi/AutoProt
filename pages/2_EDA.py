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
    sorted_cols = sort_columns_by
