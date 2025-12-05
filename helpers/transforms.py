"""
helpers/transforms.py
Data transformation functions - ENHANCED VERSION
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import (
    QuantileTransformer, 
    PowerTransformer, 
    RobustScaler,
    StandardScaler,
    MinMaxScaler
)
# Add these imports at the very top if not already there
import streamlit as st
from sklearn.preprocessing import PowerTransformer
from typing import Dict

# ============================================================================
# ADD THIS NEW FUNCTION (keep all your existing functions below)
# ============================================================================

@st.cache_data(show_spinner="🔄 Computing all transformations...")
def compute_all_transforms_cached(
    df: pd.DataFrame, 
    numeric_cols: list,
    _hash_key: str
) -> Dict[str, pd.DataFrame]:
    """
    Pre-compute ALL transformations once and cache them.
    
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
    dict: {
        'raw': DataFrame (original data),
        'log2': DataFrame,
        'log10': DataFrame,
        'ln': DataFrame,
        'sqrt': DataFrame,
        'arcsinh': DataFrame,
        'yeo_johnson': DataFrame,
        'vst': DataFrame
    }
    """
    results = {}
    
    # Store raw data
    results['raw'] = df.copy()
    
    # Extract numeric data
    df_numeric = df[numeric_cols].copy()
    
    # Log2
    df_log2 = df.copy()
    for col in numeric_cols:
        vals = df_numeric[col]
        min_val = vals[vals > 0].min() if (vals > 0).any() else 1.0
        df_log2[col] = np.log2(vals.clip(lower=min_val))
    results['log2'] = df_log2
    
    # Log10
    df_log10 = df.copy()
    for col in numeric_cols:
        vals = df_numeric[col]
        min_val = vals[vals > 0].min() if (vals > 0).any() else 1.0
        df_log10[col] = np.log10(vals.clip(lower=min_val))
    results['log10'] = df_log10
    
    # Natural log
    df_ln = df.copy()
    for col in numeric_cols:
        vals = df_numeric[col]
        min_val = vals[vals > 0].min() if (vals > 0).any() else 1.0
        df_ln[col] = np.log(vals.clip(lower=min_val))
    results['ln'] = df_ln
    
    # Square root
    df_sqrt = df.copy()
    for col in numeric_cols:
        df_sqrt[col] = np.sqrt(df_numeric[col].clip(lower=0))
    results['sqrt'] = df_sqrt
    
    # Arcsinh
    df_arcsinh = df.copy()
    for col in numeric_cols:
        df_arcsinh[col] = np.arcsinh(df_numeric[col])
    results['arcsinh'] = df_arcsinh
    
    # Yeo-Johnson
    df_yj = df.copy()
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    try:
        transformed = pt.fit_transform(df_numeric)
        df_yj[numeric_cols] = transformed
    except:
        for col in numeric_cols:
            try:
                vals = df_numeric[[col]].dropna()
                if len(vals) > 0:
                    transformed = pt.fit_transform(vals)
                    df_yj.loc[vals.index, col] = transformed.flatten()
            except:
                df_yj[col] = df_numeric[col]
    results['yeo_johnson'] = df_yj
    
    # VST (Variance Stabilizing)
    df_vst = df.copy()
    median_intensity = df_numeric.median().median()
    if median_intensity <= 0 or np.isnan(median_intensity):
        median_intensity = 1.0
    
    for col in numeric_cols:
        df_vst[col] = np.arcsinh(df_numeric[col] / (2 * median_intensity))
    results['vst'] = df_vst
    
    return results


# Add these constants at the end of the file
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

def apply_transform(df: pd.DataFrame, numeric_cols: list, method: str = "log2") -> pd.DataFrame:
    """
    Apply transformation to numeric columns.
    
    Args:
        df: Input dataframe
        numeric_cols: List of numeric column names
        method: Transform method
        
    Available methods:
        - log2: Log2 transform (proteomics standard)
        - log10: Log10 transform
        - sqrt: Square root
        - cbrt: Cube root
        - yeo_johnson: Yeo-Johnson power transform
        - quantile: Quantile normalization
        - robust: Robust scaling (median/IQR)
        - zscore: Z-score standardization
        - minmax: Min-max scaling [0, 1]
        
    Returns:
        Transformed dataframe
    """
    df_transformed = df.copy()
    
    # Replace NaN and 0 with 1.0 before transformation
    for col in numeric_cols:
        if col in df_transformed.columns:
            df_transformed[col] = df_transformed[col].fillna(1.0)
            df_transformed.loc[df_transformed[col] == 0, col] = 1.0
    
    # Apply transformation
    if method == "log2":
        for col in numeric_cols:
            df_transformed[col] = np.log2(df_transformed[col])
    
    elif method == "log10":
        for col in numeric_cols:
            df_transformed[col] = np.log10(df_transformed[col])
    
    elif method == "sqrt":
        for col in numeric_cols:
            df_transformed[col] = np.sqrt(df_transformed[col])
    
    elif method == "cbrt":
        for col in numeric_cols:
            df_transformed[col] = np.cbrt(df_transformed[col])
    
    elif method == "yeo_johnson":
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        df_transformed[numeric_cols] = pt.fit_transform(df_transformed[numeric_cols])
    
    elif method == "quantile":
        qt = QuantileTransformer(output_distribution='normal')
        df_transformed[numeric_cols] = qt.fit_transform(df_transformed[numeric_cols])
    
    elif method == "robust":
        # Robust scaling using median and IQR (good for outliers)
        scaler = RobustScaler()
        df_transformed[numeric_cols] = scaler.fit_transform(df_transformed[numeric_cols])
    
    elif method == "zscore":
        # Z-score standardization (mean=0, std=1)
        scaler = StandardScaler()
        df_transformed[numeric_cols] = scaler.fit_transform(df_transformed[numeric_cols])
    
    elif method == "minmax":
        # Min-max scaling to [0, 1]
        scaler = MinMaxScaler()
        df_transformed[numeric_cols] = scaler.fit_transform(df_transformed[numeric_cols])
    
    else:
        raise ValueError(f"Unknown transform method: {method}")
    
    # Handle infinite values
    df_transformed[numeric_cols] = df_transformed[numeric_cols].replace([np.inf, -np.inf], 0)
    
    return df_transformed


def inverse_transform(df: pd.DataFrame, numeric_cols: list, method: str = "log2") -> pd.DataFrame:
    """
    Reverse transformation (useful for back-transforming results).
    
    Args:
        df: Transformed dataframe
        numeric_cols: List of numeric column names
        method: Original transform method
        
    Returns:
        Back-transformed dataframe
        
    Note: Only works for simple transforms (log, sqrt, cbrt).
          For sklearn scalers, you need the fitted scaler object.
    """
    df_original = df.copy()
    
    if method == "log2":
        for col in numeric_cols:
            df_original[col] = 2 ** df_original[col]
    
    elif method == "log10":
        for col in numeric_cols:
            df_original[col] = 10 ** df_original[col]
    
    elif method == "sqrt":
        for col in numeric_cols:
            df_original[col] = df_original[col] ** 2
    
    elif method == "cbrt":
        for col in numeric_cols:
            df_original[col] = df_original[col] ** 3
    
    else:
        raise ValueError(f"Inverse transform not supported for {method}. Store fitted scaler for sklearn methods.")
    
    return df_original


def get_transform_info(method: str) -> dict:
    """
    Get information about a transform method.
    
    Args:
        method: Transform method name
        
    Returns:
        Dict with description, use_case, and interpretation
    """
    info = {
        "log2": {
            "name": "Log2 Transform",
            "use_case": "Standard for proteomics - fold changes become linear",
            "interpretation": "Difference of 1 = 2-fold change",
            "handles_negatives": False,
            "invertible": True
        },
        "log10": {
            "name": "Log10 Transform",
            "use_case": "Wide dynamic range, easier to interpret powers of 10",
            "interpretation": "Difference of 1 = 10-fold change",
            "handles_negatives": False,
            "invertible": True
        },
        "sqrt": {
            "name": "Square Root",
            "use_case": "Gentle transform for count data, stabilizes variance",
            "interpretation": "Less aggressive than log",
            "handles_negatives": False,
            "invertible": True
        },
        "cbrt": {
            "name": "Cube Root",
            "use_case": "Very gentle transform, handles near-zero values better",
            "interpretation": "Even less aggressive than sqrt",
            "handles_negatives": True,
            "invertible": True
        },
        "yeo_johnson": {
            "name": "Yeo-Johnson Transform",
            "use_case": "Automatic optimal power transform, handles negatives",
            "interpretation": "Makes data more normal-like",
            "handles_negatives": True,
            "invertible": False
        },
        "quantile": {
            "name": "Quantile Normalization",
            "use_case": "Force normal distribution, remove batch effects",
            "interpretation": "Ranks → normal quantiles",
            "handles_negatives": True,
            "invertible": False
        },
        "robust": {
            "name": "Robust Scaling",
            "use_case": "Remove outlier influence using median/IQR",
            "interpretation": "Centered at median, scaled by IQR",
            "handles_negatives": True,
            "invertible": False
        },
        "zscore": {
            "name": "Z-Score Standardization",
            "use_case": "Compare across different scales, ML preprocessing",
            "interpretation": "Units of standard deviations from mean",
            "handles_negatives": True,
            "invertible": False
        },
        "minmax": {
            "name": "Min-Max Scaling",
            "use_case": "Scale to [0, 1] range, ML preprocessing",
            "interpretation": "0 = minimum, 1 = maximum",
            "handles_negatives": True,
            "invertible": False
        }
    }
    
    return info.get(method, {
        "name": method, 
        "use_case": "Unknown", 
        "interpretation": "N/A",
        "handles_negatives": False,
        "invertible": False
    })


def recommend_transform(df: pd.DataFrame, numeric_cols: list) -> str:
    """
    Recommend a transform based on data properties.
    
    Args:
        df: Input dataframe
        numeric_cols: List of numeric column names
        
    Returns:
        Recommended transform method
    """
    # Check for negative values
    has_negatives = (df[numeric_cols] < 0).any().any()
    
    # Check dynamic range
    valid_data = df[numeric_cols].replace(1.0, np.nan).dropna()
    if len(valid_data) > 0:
        dynamic_range = valid_data.max().max() / valid_data.min().min()
    else:
        dynamic_range = 1
    
    # Recommend based on properties
    if has_negatives:
        return "robust"  # Handles negatives well
    elif dynamic_range > 1000:
        return "log2"  # Large dynamic range
    elif dynamic_range > 100:
        return "sqrt"  # Moderate dynamic range
    else:
        return "zscore"  # Small dynamic range
