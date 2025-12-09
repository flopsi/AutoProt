"""
helpers/transforms.py - Data Transformation Functions

Mathematical transformations for proteomics intensity normalization.
Essential transforms optimized for performance.
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from typing import Dict, List, Tuple


# ============================================================================
# TRANSFORMATION METADATA
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
    "log2": "Log base 2 - standard for proteomics fold-change analysis",
    "yeo-johnson": "Power transformation - handles all value ranges including zeros",
    "arcsin": "Inverse sine of sqrt - stabilizes variance for rare proteins/peptides",
    "quantile": "Rank-based transformation to approximate normal distribution",
}


# ============================================================================
# TRANSFORMATION FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600, show_spinner="Applying transformation...")
def apply_transformation(
    df: pd.DataFrame,
    numeric_cols: List[str],
    method: str = "log2"
) -> pd.DataFrame:
    """
    Apply specified mathematical transformation to intensity data.
    
    Results are cached for 1 hour to avoid recomputation.
    
    Args:
        df: Input DataFrame with raw intensities
        numeric_cols: List of numeric column names to transform
        method: Transformation method (see TRANSFORM_NAMES keys)
    
    Returns:
        DataFrame with transformed data in new columns (original columns kept)
    
    Raises:
        ValueError: If method is not recognized
        KeyError: If numeric columns don't exist
    
    Note:
        Transformed columns are named with '_transformed' suffix.
        Original columns are preserved for reference.
    """
    # Validate inputs
    if method not in TRANSFORM_NAMES:
        raise ValueError(f"Unknown transformation: {method}. Available: {list(TRANSFORM_NAMES.keys())}")
    
    missing = set(numeric_cols) - set(df.columns)
    if missing:
        raise KeyError(f"Columns not in DataFrame: {missing}")
    
    df_out = df.copy()
    
    if method == "raw":
        # Add '_transformed' copies of raw data
        for col in numeric_cols:
            df_out[f"{col}_transformed"] = df_out[col]
        return df_out
    
    for col in numeric_cols:
        if col not in df_out.columns:
            continue
        
        vals = df_out[col].dropna()
        
        if len(vals) == 0:
            df_out[f"{col}_transformed"] = np.nan
            continue
        
        try:
            if method == "log2":
                # Clip to 1e-10 to avoid log of zero
                df_out[f"{col}_transformed"] = np.log2(vals.clip(lower=1e-10))
            
            elif method == "yeo-johnson":
                # Power transformation that handles all ranges including negatives
                pt = PowerTransformer(method="yeo-johnson", standardize=False)
                transformed = pt.fit_transform(vals.values.reshape(-1, 1)).ravel()
                df_out.loc[vals.index, f"{col}_transformed"] = transformed
            
            elif method == "arcsin":
                # Arcsin(sqrt) for variance stabilization of rare proteins
                min_val, max_val = vals.min(), vals.max()
                normalized = (vals - min_val) / (max_val - min_val) if max_val > min_val else vals
                df_out.loc[vals.index, f"{col}_transformed"] = np.arcsin(np.sqrt(np.clip(normalized, 0, 1)))
            
            elif method == "quantile":
                # Rank-based normalization to approximate normal distribution
                qt = QuantileTransformer(
                    n_quantiles=min(1000, len(vals)),
                    output_distribution="normal",
                    random_state=0
                )
                v_tr = qt.fit_transform(vals.values.reshape(-1, 1)).ravel()
                df_out.loc[vals.index, f"{col}_transformed"] = v_tr
            
            else:
                # Fallback: keep original
                df_out.loc[vals.index, f"{col}_transformed"] = vals
        
        except Exception as e:
            # If transformation fails, keep original data
            print(f"Warning: Transformation {method} failed for {col}: {str(e)}")
            df_out[f"{col}_transformed"] = vals
    
    return df_out


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_transform_name(method: str) -> str:
    """
    Get human-readable name for transformation method.
    
    Args:
        method: Transformation method key
    
    Returns:
        Human-readable name
    """
    return TRANSFORM_NAMES.get(method, method.title())


def get_transform_description(method: str) -> str:
    """
    Get detailed description for transformation method.
    
    Args:
        method: Transformation method key
    
    Returns:
        Description string
    """
    return TRANSFORM_DESCRIPTIONS.get(method, "No description available.")


def list_available_transforms() -> List[str]:
    """
    Get list of all available transformation methods.
    
    Returns:
        List of transformation method keys
    """
    return list(TRANSFORM_NAMES.keys())


@st.cache_data(ttl=3600)
def compute_transform_comparison(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """
    Compute normality metrics for all transformations to help select best one.
    
    Tests each transformation and computes Shapiro-Wilk p-value, kurtosis,
    skewness, and an overall normality score.
    
    Args:
        df: Input DataFrame
        numeric_cols: Numeric columns to test
    
    Returns:
        DataFrame with transformation comparison metrics, sorted by normality score
    
    Example:
        >>> df = pd.DataFrame({'A1': [100, 200, 150], 'A2': [110, 210, 160]})
        >>> comparison = compute_transform_comparison(df, ['A1', 'A2'])
        >>> comparison[['Transform', 'Score']].head()
    """
    # Validate inputs
    if not isinstance(numeric_cols, list):
        raise TypeError("numeric_cols must be a list")
    
    missing = set(numeric_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Columns not in DataFrame: {missing}")
    
    transform_stats = []
    
    for trans_name in TRANSFORM_NAMES.keys():
        try:
            # Apply transformation
            if trans_name == "raw":
                test_df = df[numeric_cols]
            else:
                test_df = apply_transformation(df[numeric_cols], numeric_cols, trans_name)
                # Use transformed columns
                test_cols = [f"{col}_transformed" for col in numeric_cols]
                test_df = test_df[test_cols]
            
            # Collect all non-nan values
            all_values = np.concatenate([test_df[col].dropna().values for col in test_df.columns])
            all_values = all_values[np.isfinite(all_values)]
            
            if len(all_values) > 3:
                # Sample for Shapiro-Wilk test (limit to 5000 samples)
                sample_vals = np.random.choice(all_values, size=min(5000, len(all_values)), replace=False)
                stat_sw, pval_sw = stats.shapiro(sample_vals)
                
                # Compute statistics
                kurt = stats.kurtosis(all_values)
                skew = stats.skew(all_values)
                
                # Normality score: higher is better
                # p-value closer to 1 = more normal, kurtosis/skewness closer to 0 = more normal
                norm_score = (1 - pval_sw) * 0.5 + (abs(kurt) / 10) * 0.3 + (abs(skew) / 5) * 0.2
                
                transform_stats.append({
                    'Transform': TRANSFORM_NAMES[trans_name],
                    'Shapiro_p': round(pval_sw, 4),
                    'Kurtosis': round(kurt, 3),
                    'Skewness': round(skew, 3),
                    'Score': round(norm_score, 3),
                    '_key': trans_name
                })
        
        except Exception as e:
            print(f"Warning: Could not compute statistics for {trans_name}: {str(e)}")
            continue
    
    result_df = pd.DataFrame(transform_stats).sort_values('Score', ascending=False)
    return result_df
