# helpers/eda_cache.py

import streamlit as st
import pandas as pd
from typing import List, Dict, Tuple

from helpers.transforms import apply_transformation
from helpers.evaluation import evaluate_transformation_metrics


@st.cache_data(show_spinner=False)
def get_method_results(
    df_raw: pd.DataFrame,
    eval_cols: List[str],
    method: str,
) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    """
    Compute and cache everything needed for one method:
    - transformed df (only eval_cols)
    - transformed column names
    - metrics (Shapiro + mean–variance correlations)
    """
    df_trans, trans_cols = apply_transformation(df_raw, eval_cols, method)
    metrics = evaluate_transformation_metrics(df_raw, df_trans, eval_cols, trans_cols)
    return df_trans, trans_cols, metrics
