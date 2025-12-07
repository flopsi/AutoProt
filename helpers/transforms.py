"""
helpers/transforms.py - OPTIMIZED FOR PERFORMANCE
Data transformation functions for proteomics intensity normalization
Reduced to 5 essential transforms: log2, yeo-johnson, arcsin, quantile, raw
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from typing import Dict, List, Tuple

# ============================================================================
# TRANSFORMATION METADATA - REDUCED TO 5 ESSENTIAL TRANSFORMS
# ============================================================================

TRANSFORM_NAMES: Dict[str, str] = {
    "raw": "Raw (No Transform)",
    "log2": "Log2",
    "yeo-johnson": "Yeo-Johnson (Power Transform)",
    "arcsin": "Arcsin (Rare Proteins)",
    "quantile": "Quantile Normalization",
}

TRANSFORM_DESCRIPTIONS: Dict[str, str] = {
    "raw": "Original data without transformation",
    "log2": "Log base 2 - standard for proteomics fold-change (handles 10x data)",
    "yeo-johnson": "Power transformation - handles all value ranges including zeros",
    "arcsin": "Inverse sine of sqrt - stabilizes variance for rare proteins/peptides",
    "quantile": "Rank-based transformation to approximate normal distribution",
}

# ============================================================================
# TRANSFORMATION FUNCTIONS - CACHED FOR PERFORMANCE
# ============================================================================

@st.cache_data(ttl=3600, show_spinner="Applying transformation...")
def apply_transformation(
    df: pd.DataFrame,
    numeric_cols: List[str],
    method: str = "log2"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply specified mathematical transformation to intensity data.
    Results are cached for 1 hour to avoid recomputation.
    
    Args:
        df: Input DataFrame with raw intensities
        numeric_cols: List of numeric column names to transform
        method: Transformation method (see TRANSFORM_NAMES keys)
        
    Returns:
        (df_transformed, transformed_col_names) tuple
    """
    
    df_out = df.copy()
    
    if method == "raw":
        return df_out, numeric_cols
    
    for col in numeric_cols:
        if col not in df_out.columns:
            continue
        
        vals = df_out[col].dropna()
        if len(vals) == 0:
            continue
        
        if method == "log2":
            df_out.loc[vals.index, f"{col}_t"] = np.log2(vals.clip(lower=1e-10))
        
        elif method == "yeo-johnson":
            try:
                pt = PowerTransformer(method="yeo-johnson", standardize=False)
                transformed = pt.fit_transform(vals.values.reshape(-1, 1)).ravel()
                df_out.loc[vals.index, f"{col}_t"] = transformed
            except Exception:
                df_out.loc[vals.index, f"{col}_t"] = vals
        
        elif method == "arcsin":
            try:
                min_val, max_val = vals.min(), vals.max()
                normalized = (vals - min_val) / (max_val - min_val) if max_val > min_val else vals
                df_out.loc[vals.index, f"{col}_t"] = np.arcsin(np.sqrt(np.clip(normalized, 0, 1)))
            except Exception:
                df_out.loc[vals.index, f"{col}_t"] = vals
        
        elif method == "quantile":
            try:
                qt = QuantileTransformer(n_quantiles=min(1000, len(vals)),
                                        output_distribution="normal", random_state=0)
                v_tr = qt.fit_transform(vals.values.reshape(-1, 1)).ravel()
                df_out.loc[vals.index, f"{col}_t"] = v_tr
            except Exception:
                df_out.loc[vals.index, f"{col}_t"] = vals
        
        else:
            df_out.loc[vals.index, f"{col}_t"] = vals
    
    transformed_cols = [f"{col}_t" for col in numeric_cols if f"{col}_t" in df_out.columns]
    return df_out, transformed_cols


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_transform_name(method: str) -> str:
    """Get human-readable name for transformation method."""
    return TRANSFORM_NAMES.get(method, method.title())


def get_transform_description(method: str) -> str:
    """Get detailed description for transformation method."""
    return TRANSFORM_DESCRIPTIONS.get(method, "No description available.")


def list_available_transforms() -> List[str]:
    """Get list of all available transformation methods."""
    return list(TRANSFORM_NAMES.keys())


@st.cache_data(ttl=3600)
def compute_transform_comparison(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """
    Compute normality metrics for all 5 transformations to help select best one.
    """
    
    transform_stats = []
    
    for trans_name in TRANSFORM_NAMES.keys():
        test_df = df[numeric_cols] if trans_name == "raw" else apply_transformation(df[numeric_cols], numeric_cols, trans_name)[0]
        
        all_values = np.concatenate([test_df[col].dropna().values for col in test_df.columns])
        all_values = all_values[np.isfinite(all_values)]
        
        if len(all_values) > 3:
            sample_vals = np.random.choice(all_values, size=min(5000, len(all_values)), replace=False)
            stat_sw, pval_sw = stats.shapiro(sample_vals)
            kurt = stats.kurtosis(all_values)
            skew = stats.skew(all_values)
            norm_score = (1 - pval_sw) * 0.5 + (abs(kurt) / 10) * 0.3 + (abs(skew) / 5) * 0.2
            
            transform_stats.append({
                'Transform': TRANSFORM_NAMES[trans_name],
                'Shapiro_p': round(pval_sw, 4),
                'Kurtosis': round(kurt, 3),
                'Skewness': round(skew, 3),
                'Score': round(norm_score, 3),
                '_key': trans_name
            })
    
    return pd.DataFrame(transform_stats).sort_values('Score')
