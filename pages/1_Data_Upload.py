import streamlit as st
import pandas as pd
import numpy as np
import re
from dataclasses import dataclass
from typing import List

from components import inject_custom_css, render_header, render_navigation, render_footer, COLORS

st.set_page_config(
    page_title="Data Upload | Thermo Fisher Scientific",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
render_header()

@st.cache_data
def get_msdata(df, quant_cols):
    return build_msdata(df, quant_cols)

@dataclass
class MSData:
    original: pd.DataFrame
    filled: pd.DataFrame
    log2_filled: pd.DataFrame
    numeric_cols: List[str]
    ones_count: int


def auto_rename_columns(columns: List[str]) -> dict:
    rename_map = {}
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, col in enumerate(columns):
        cond_idx = i // 3
        rep = (i % 3) + 1
        if cond_idx < len(letters):
            new_name = f"{letters[cond_idx]}{rep}"
        else:
            new_name = f"C{cond_idx+1}_{rep}"
        rename_map[col] = new_name
    return rename_map


def filter_by_species(df: pd.DataFrame, col: str, species_tags: list[str]) -> pd.DataFrame:
    if not species_tags or not col:
        return df
    pattern = "|".join(re.escape(tag) for tag in species_tags)
    return df[df[col].astype(str).str.contains(pattern, case=False, na=False)]


def build_msdata(processed_df: pd.DataFrame, numeric_cols_renamed: List[str]) -> MSData:
    original = processed_df.copy()

    # 1) Force all selected quant columns to numeric float64 BEFORE any operations
    original[numeric_cols_renamed] = (
        original[numeric_cols_renamed]
        .apply(pd.to_numeric, errors="coerce")
        .astype("float64")
    )

    # 2) Filled: NaN, 0, 1 → 1
    filled = original.copy()
    vals = filled[numeric_cols_renamed]
    vals = vals.fillna(1.0)
    vals = vals.where(~vals.isin([0.0, 1.0]), 1.0)
    filled[numeric_cols_renamed] = vals

    # Count cells == 1 after filling
    ones_count = (filled[numeric_cols_renamed] == 1.0).to_numpy().sum()

    # 3) log2(filled)
    log2_filled = filled.copy()
    log2_filled[numeric_cols_renamed] = np.log2(log2_filled[numeric_cols_renamed])

    return MSData(
        original=original,
        filled=filled,
        log2_filled=log2_filled,
        numeric_cols=numeric_cols_renamed,
        ones_count=ones_count,
    )


# ------------- session state -------------
