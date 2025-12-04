"""
helpers/transforms.py
Data transformation functions with Streamlit caching
Cached so expensive computations only happen once per session
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import yeojohnson
from sklearn.preprocessing import QuantileTransformer

# ============================================================================
# CACHED TRANSFORM FUNCTIONS
# ============================================================================

@st.cache_data
def transform_log2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log2 transformation: log2(x)
    Handles zeros and negatives by replacing with NaN before transform
    """
    # Replace zeros and negatives with NaN
    df_safe = df.copy()
    df_safe[df_safe <= 0] = np.nan
    return np.log2(df_safe)


@st.cache_data
def transform_log10(df: pd.DataFrame) -> pd.DataFrame:
    """Log10 transformation: log10(x)."""
    df_safe = df.copy()
    df_safe[df_safe <= 0] = np.nan
    return np.log10(df_safe)


@st.cache_data
def transform_sqrt(df: pd.DataFrame) -> pd.DataFrame:
    """Square root transformation: sqrt(x)."""
    df_safe = df.copy()
    df_safe[df_safe < 0] = np.nan
    return np.sqrt(df_safe)


@st.cache_data
def transform_cbrt(df: pd.DataFrame) -> pd.DataFrame:
    """Cube root transformation: cbrt(x) = x^(1/3)."""
    return np.cbrt(df)


@st.cache_data
def transform_yeo_johnson(df: pd.DataFrame) -> pd.DataFrame:
    """
    Yeo-Johnson transformation: variance-stabilizing transform
    Handles zeros and negatives unlike Box-Cox
    """
    result = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for col in df.columns:
        try:
            # yeojohnson returns (transformed_data, lambda_param)
            transformed, _ = yeojohnson(df[col].dropna())
            result[col] = np.nan
            result.loc[df[col].notna(), col] = transformed
        except Exception:
            result[col] = df[col]
    return result


@st.cache_data
def transform_quantile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quantile normalization: standardize across samples
    Each column gets normalized to uniform distribution
    """
    try:
        scaler = QuantileTransformer(n_quantiles=min(1000, len(df)), random_state=42)
        scaled = scaler.fit_transform(df.dropna())
        result = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
        mask = df.notna().values
        result[mask] = scaled
        return result
    except Exception:
        return df.copy()


# ============================================================================
# TRANSFORM DISPATCHER
# ============================================================================

def get_transform(df: pd.DataFrame, transform_key: str) -> pd.DataFrame:
    """
    Get transformed data by transform key name.
    
    Args:
        df: Input dataframe
        transform_key: One of ['log2', 'log10', 'sqrt', 'cbrt', 'yeo_johnson', 'quantile']
        
    Returns:
        Transformed dataframe
    """
    transforms = {
        "log2": transform_log2,
        "log10": transform_log10,
        "sqrt": transform_sqrt,
        "cbrt": transform_cbrt,
        "yeo_johnson": transform_yeo_johnson,
        "quantile": transform_quantile,
    }
    
    transform_func = transforms.get(transform_key.lower(), transform_log2)
    return transform_func(df)


# ============================================================================
# STATISTICS ON TRANSFORMED DATA
# ============================================================================

def compute_cv_per_condition(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """
    Compute coefficient of variation (CV) within each condition.
    CV is useful for assessing technical reproducibility.
    
    Args:
        df: Intensity data (proteins × samples)
        numeric_cols: List of column names
        
    Returns:
        DataFrame with CV% for each condition
    """
    # Group columns by first letter (condition)
    condition_groups = {}
    for col in numeric_cols:
        if col and col[0].isalpha():
            cond = col[0]
            if cond not in condition_groups:
                condition_groups[cond] = []
            condition_groups[cond].append(col)
    
    cv_results = {}
    for cond, cols in condition_groups.items():
        if len(cols) < 2:
            continue
        
        mean_vals = df[cols].mean(axis=1)
        std_vals = df[cols].std(axis=1, ddof=1)
        cv = (std_vals / mean_vals * 100).replace([np.inf, -np.inf], np.nan)
        cv_results[f"CV_{cond}"] = cv
    
    return pd.DataFrame(cv_results, index=df.index)


def compute_condition_means(df: pd.DataFrame, numeric_cols: list) -> Dict:
    """
    Compute mean intensity per condition.
    Used for fold change calculations and summary statistics.
    
    Args:
        df: Intensity data
        numeric_cols: List of column names
        
    Returns:
        Dict with condition → mean Series
    """
    condition_groups = {}
    for col in numeric_cols:
        if col and col[0].isalpha():
            cond = col[0]
            if cond not in condition_groups:
                condition_groups[cond] = []
            condition_groups[cond].append(col)
    
    means = {}
    for cond, cols in condition_groups.items():
        means[cond] = df[cols].mean(axis=1)
    
    return means
