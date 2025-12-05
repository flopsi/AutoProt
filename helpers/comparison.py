# helpers/comparison.py

import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple

from helpers.eda_cache import get_method_results


@st.cache_data(show_spinner=False)
def compare_transformations(
    df_raw: pd.DataFrame,
    numeric_cols: List[str],
    methods: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Use cached per-method results to build summary table.
    """
    results = []
    metrics_by_method: Dict[str, Dict] = {}

    for m in methods:
        _, _, metrics = get_method_results(df_raw, numeric_cols, m)
        metrics_by_method[m] = metrics
        results.append(
            dict(
                method=m,
                shapiro_p=metrics["shapiro_trans"],
                mean_var_corr=metrics["mean_var_corr_trans"],
            )
        )

    summary = pd.DataFrame(results)
    if not summary.empty:
        summary["combined_score"] = (
            summary["shapiro_p"].rank(ascending=False)
            + (1 - summary["mean_var_corr"].abs()).rank(ascending=False)
        )
        summary = summary.sort_values("combined_score").reset_index(drop=True)
    return summary, metrics_by_method
