# helpers/comparison.py

import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple
from helpers.transforms import cached_apply_transformation
from helpers.evaluation import cached_evaluate_transformation_metrics

@st.cache_data(show_spinner=False)
def compare_transformations(
    df_raw: pd.DataFrame,
    numeric_cols: List[str],
    methods: List[str],
    file_hash: str,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    results = []
    metrics_by_method: Dict[str, Dict] = {}

    for m in methods:
        df_t, trans_cols = cached_apply_transformation(df_raw, numeric_cols, m, file_hash)
        metrics = cached_evaluate_transformation_metrics(
            df_raw, df_t, numeric_cols, trans_cols, m, file_hash
        )
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
