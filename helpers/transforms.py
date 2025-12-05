# helpers/transforms.py

"""
Simple, robust transformations from working Colab code
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from typing import Dict, List, Tuple

TRANSFORM_NAMES: Dict[str, str] = {
    "raw": "Raw (No Transform)",
    "log2": "Log2",
    "log10": "Log10",
    "ln": "Natural Log (ln)",
    "sqrt": "Square Root",
    "arcsinh": "Arcsinh",
    "boxcox": "Box-Cox",
    "yeo-johnson": "Yeo-Johnson",
    "vst": "Variance Stabilizing (VST)",
    "quantile": "Quantile (Rank-based)",
}

TRANSFORM_DESCRIPTIONS: Dict[str, str] = {
    "raw": "Original data without transformation",
    "log2": "Log base 2 - standard for proteomics fold-change",
    "log10": "Log base 10 transformation",
    "ln": "Natural logarithm (base e)",
    "sqrt": "Square root transformation",
    "arcsinh": "Inverse hyperbolic sine - handles negatives",
    "boxcox": "Box-Cox power transformation (positive values only)",
    "yeo-johnson": "Yeo-Johnson power transformation (handles zeros/negatives)",
    "vst": "Variance stabilizing transformation (asinh(x / 2*median))",
    "quantile": "Rank-based quantile transformation to an approximately normal distribution",
}


def apply_transformation(
    df: pd.DataFrame, numeric_cols: List[str], method: str = "log2"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply specified transformation to intensity data.

    Returns a new DataFrame plus the list of transformed column names.
    """
    df_out = df.copy()

    if method == "raw":
        # No new columns created, return original cols as "transformed"
        return df_out, numeric_cols

    for col in numeric_cols:
        if col not in df_out.columns:
            continue
        vals = df_out[col].dropna()

        if method == "log2":
            df_out.loc[vals.index, f"{col}_transformed"] = np.log2(
                vals.clip(lower=1e-10)
            )

        elif method == "log10":
            df_out.loc[vals.index, f"{col}_transformed"] = np.log10(
                vals.clip(lower=1e-10)
            )

        elif method == "ln":
            df_out.loc[vals.index, f"{col}_transformed"] = np.log(
                vals.clip(lower=1e-10)
            )

        elif method == "sqrt":
            df_out.loc[vals.index, f"{col}_transformed"] = np.sqrt(
                vals.clip(lower=0)
            )

        elif method == "arcsinh":
            df_out.loc[vals.index, f"{col}_transformed"] = np.arcsinh(vals)

        elif method == "boxcox":
            if (vals > 0).all():
                try:
                    transformed, _ = stats.boxcox(vals)
                    df_out.loc[vals.index, f"{col}_transformed"] = transformed
                except Exception:
                    df_out.loc[vals.index, f"{col}_transformed"] = vals
            else:
                df_out.loc[vals.index, f"{col}_transformed"] = vals

        elif method == "yeo-johnson":
            try:
                pt = PowerTransformer(method="yeo-johnson", standardize=False)
                transformed = pt.fit_transform(vals.values.reshape(-1, 1)).ravel()
                df_out.loc[vals.index, f"{col}_transformed"] = transformed
            except Exception:
                df_out.loc[vals.index, f"{col}_transformed"] = vals

        elif method == "vst":
            median_intensity = vals.median()
            if pd.isna(median_intensity) or median_intensity <= 0:
                median_intensity = 1.0
            df_out.loc[vals.index, f"{col}_transformed"] = np.arcsinh(
                vals / (2 * median_intensity)
            )

        elif method == "quantile":
            # quantile -> normal using only non-NaN values
            try:
                qt = QuantileTransformer(
                    n_quantiles=min(1000, len(vals)),
                    output_distribution="normal",
                    random_state=0,
                )
                v = vals.to_numpy().reshape(-1, 1)
                v_tr = qt.fit_transform(v).ravel()
                df_out.loc[vals.index, f"{col}_transformed"] = v_tr
            except Exception:
                df_out.loc[vals.index, f"{col}_transformed"] = vals

        else:
            # unknown method: just copy
            df_out.loc[vals.index, f"{col}_transformed"] = vals

    transformed_cols = [
        f"{col}_transformed"
        for col in numeric_cols
        if f"{col}_transformed" in df_out.columns
    ]
    return df_out, transformed_cols
