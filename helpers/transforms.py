"""
helpers/transforms.py
Data transformation functions with full caching support
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from typing import Dict

# ============================================================================
# CACHED TRANSFORM COMPUTATION
# ============================================================================

@st.cache_data(show_spinner="🔄 Computing all transformations...")
def compute_all_transforms_cached(
    df: pd.DataFrame, 
    numeric_cols: list,
    _hash_key: str  # For cache invalidation when new file uploaded
) -> Dict[str, pd.DataFrame]:
    """
    Pre-compute ALL transformations once and cache them.
    This is the main performance optimization - transforms computed once,
    reused everywhere.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw data with numeric columns
    numeric_cols : list
        List of column names to transform
    _hash_key : str
        Hash key for cache invalidation (usually file path)
    
    Returns:
    --------
    dict : {
        'raw': DataFrame (original data),
        'log2': DataFrame,
        'log10': DataFrame,
        'ln': DataFrame,
        'sqrt': DataFrame,
        'arcsinh': DataFrame,
        'yeo_johnson': DataFrame,
        'vst': DataFrame (variance-stabilizing)
    }
    """
    results = {}
    
    # Store raw data
    results['raw'] = df.copy()
    
    # Extract numeric data for transformations
    df_numeric = df[numeric_cols].copy()
    
    # ========================================================================
    # LOG2 TRANSFORMATION
    # ========================================================================
    df_log2 = df.copy()
    for col in numeric_cols:
        # Handle zeros and negatives by adding small constant if needed
        vals = df_numeric[col]
        min_val = vals[vals > 0].min() if (vals > 0).any() else 1.0
        df_log2[col] = np.log2(vals.clip(lower=min_val))
    results['log2'] = df_log2
    
    # ========================================================================
    # LOG10 TRANSFORMATION
    # ========================================================================
    df_log10 = df.copy()
    for col in numeric_cols:
        vals = df_numeric[col]
        min_val = vals[vals > 0].min() if (vals > 0).any() else 1.0
        df_log10[col] = np.log10(vals.clip(lower=min_val))
    results['log10'] = df_log10
    
    # ========================================================================
    # NATURAL LOG (LN) TRANSFORMATION
    # ========================================================================
    df_ln = df.copy()
    for col in numeric_cols:
        vals = df_numeric[col]
        min_val = vals[vals > 0].min() if (vals > 0).any() else 1.0
        df_ln[col] = np.log(vals.clip(lower=min_val))
    results['ln'] = df_ln
    
    # ========================================================================
    # SQUARE ROOT TRANSFORMATION
    # ========================================================================
    df_sqrt = df.copy()
    for col in numeric_cols:
        # sqrt requires non-negative values
        vals = df_numeric[col]
        df_sqrt[col] = np.sqrt(vals.clip(lower=0))
    results['sqrt'] = df_sqrt
    
    # ========================================================================
    # ARCSINH TRANSFORMATION (inverse hyperbolic sine)
    # ========================================================================
    df_arcsinh = df.copy()
    for col in numeric_cols:
        # arcsinh can handle all real numbers, including negatives
        df_arcsinh[col] = np.arcsinh(df_numeric[col])
    results['arcsinh'] = df_arcsinh
    
    # ========================================================================
    # YEO-JOHNSON TRANSFORMATION
    # ========================================================================
    df_yj = df.copy()
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    try:
        # Transform all numeric columns at once
        transformed = pt.fit_transform(df_numeric)
        df_yj[numeric_cols] = transformed
    except Exception as e:
        # Fallback: transform column by column
        for col in numeric_cols:
            try:
                vals = df_numeric[[col]].dropna()
                if len(vals) > 0:
                    transformed = pt.fit_transform(vals)
                    df_yj.loc[vals.index, col] = transformed.flatten()
            except:
                # If transform fails, keep original
                df_yj[col] = df_numeric[col]
    results['yeo_johnson'] = df_yj
    
    # ========================================================================
    # VST (VARIANCE STABILIZING TRANSFORMATION)
    # ========================================================================
    df_vst = df.copy()
    # Calculate global median across all intensity columns
    median_intensity = df_numeric.median().median()
    if median_intensity <= 0 or np.isnan(median_intensity):
        median_intensity = 1.0
    
    for col in numeric_cols:
        # VST: arcsinh(x / (2 * median))
        df_vst[col] = np.arcsinh(df_numeric[col] / (2 * median_intensity))
    results['vst'] = df_vst
    
    return results


# ============================================================================
# INDIVIDUAL TRANSFORM FUNCTIONS (for backwards compatibility)
# ============================================================================

def apply_log2(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """Apply log2 transformation."""
    df_out = df.copy()
    for col in numeric_cols:
        vals = df[col]
        min_val = vals[vals > 0].min() if (vals > 0).any() else 1.0
        df_out[col] = np.log2(vals.clip(lower=min_val))
    return df_out


def apply_log10(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """Apply log10 transformation."""
    df_out = df.copy()
    for col in numeric_cols:
        vals = df[col]
        min_val = vals[vals > 0].min() if (vals > 0).any() else 1.0
        df_out[col] = np.log10(vals.clip(lower=min_val))
    return df_out


def apply_ln(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """Apply natural log transformation."""
    df_out = df.copy()
    for col in numeric_cols:
        vals = df[col]
        min_val = vals[vals > 0].min() if (vals > 0).any() else 1.0
        df_out[col] = np.log(vals.clip(lower=min_val))
    return df_out


def apply_sqrt(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """Apply square root transformation."""
    df_out = df.copy()
    for col in numeric_cols:
        df_out[col] = np.sqrt(df[col].clip(lower=0))
    return df_out


def apply_arcsinh(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """Apply inverse hyperbolic sine transformation."""
    df_out = df.copy()
    for col in numeric_cols:
        df_out[col] = np.arcsinh(df[col])
    return df_out


def apply_yeo_johnson(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """Apply Yeo-Johnson power transformation."""
    df_out = df.copy()
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    transformed = pt.fit_transform(df[numeric_cols])
    df_out[numeric_cols] = transformed
    return df_out


def apply_vst(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """Apply variance-stabilizing transformation."""
    df_out = df.copy()
    median_intensity = df[numeric_cols].median().median()
    if median_intensity <= 0 or np.isnan(median_intensity):
        median_intensity = 1.0
    
    for col in numeric_cols:
        df_out[col] = np.arcsinh(df[col] / (2 * median_intensity))
    return df_out


# ============================================================================
# TRANSFORM METADATA
# ============================================================================

TRANSFORM_NAMES = {
    'raw': 'Raw (No Transform)',
    'log2': 'Log2',
    'log10': 'Log10',
    'ln': 'Natural Log (ln)',
    'sqrt': 'Square Root',
    'arcsinh': 'Inverse Hyperbolic Sine',
    'yeo_johnson': 'Yeo-Johnson',
    'vst': 'Variance Stabilizing (VST)'
}

TRANSFORM_DESCRIPTIONS = {
    'raw': 'Original data without any transformation',
    'log2': 'Log base 2 transformation - standard for fold-change analysis',
    'log10': 'Log base 10 transformation - decimal log scale',
    'ln': 'Natural logarithm - base e transformation',
    'sqrt': 'Square root transformation - reduces right skew',
    'arcsinh': 'Inverse hyperbolic sine - similar to log but handles negatives',
    'yeo_johnson': 'Power transformation that handles zeros and negatives',
    'vst': 'Variance stabilizing - normalizes variance across intensity range'
}
