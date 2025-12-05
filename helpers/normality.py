"""
helpers/normality.py
Normality testing and transformation analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple

# ============================================================================
# NORMALITY STATISTICS COMPUTATION
# ============================================================================

@st.cache_data(show_spinner=False)
def compute_normality_stats(values: np.ndarray) -> Dict[str, float]:
    """
    Compute normality statistics for a 1D array.
    
    Parameters:
    -----------
    values : np.ndarray
        1D array of numeric values
    
    Returns:
    --------
    dict with keys:
        - kurtosis: Excess kurtosis (0 = normal)
        - skewness: Skewness (0 = symmetric)
        - W: Shapiro-Wilk test statistic
        - p: Shapiro-Wilk p-value
    """
    # Remove non-finite values
    clean = values[np.isfinite(values)]
    
    if len(clean) < 20:
        return {
            "kurtosis": np.nan,
            "skewness": np.nan,
            "W": np.nan,
            "p": np.nan
        }
    
    # For large samples, use subsample for Shapiro-Wilk
    # (test becomes too sensitive with large N)
    if len(clean) > 5000:
        sample = np.random.choice(clean, 5000, replace=False)
    else:
        sample = clean
    
    # Shapiro-Wilk test
    try:
        W, p = stats.shapiro(sample)
    except:
        W, p = np.nan, np.nan
    
    return {
        "kurtosis": float(stats.kurtosis(clean)),
        "skewness": float(stats.skew(clean)),
        "W": float(W) if not np.isnan(W) else np.nan,
        "p": float(p) if not np.isnan(p) else np.nan
    }


# ============================================================================
# TRANSFORMATION COMPARISON
# ============================================================================

@st.cache_data(show_spinner="Analyzing transformations...")
def analyze_all_transformations(
    all_transforms: Dict[str, pd.DataFrame],
    numeric_cols: list,
    _hash_key: str
) -> pd.DataFrame:
    """
    Compare normality statistics across all transformations.
    
    Parameters:
    -----------
    all_transforms : dict
        Dictionary of {transform_name: transformed_dataframe}
    numeric_cols : list
        List of numeric column names
    _hash_key : str
        Cache invalidation key
    
    Returns:
    --------
    pd.DataFrame with columns:
        - Transformation: Name of transformation
        - Kurtosis: Average kurtosis across samples
        - Skewness: Average skewness across samples
        - Shapiro_W: Average Shapiro-Wilk W statistic
        - Shapiro_p: Average Shapiro-Wilk p-value
        - Normal_Count: Number of samples passing normality (p > 0.05)
    """
    results = []
    
    for transform_name, df_transformed in all_transforms.items():
        # Collect all numeric values across all samples
        all_values = df_transformed[numeric_cols].values.flatten()
        
        # Remove non-finite and zero/negative values
        all_values = all_values[np.isfinite(all_values)]
        all_values = all_values[all_values > 0]
        
        if len(all_values) == 0:
            continue
        
        # Compute overall statistics
        stats_dict = compute_normality_stats(all_values)
        
        # Count how many individual samples pass normality test
        normal_count = 0
        for col in numeric_cols:
            col_values = df_transformed[col].dropna().values
            col_values = col_values[col_values > 0]
            if len(col_values) >= 20:
                col_stats = compute_normality_stats(col_values)
                if col_stats["p"] > 0.05:
                    normal_count += 1
        
        results.append({
            "Transformation": transform_name,
            "Kurtosis": stats_dict["kurtosis"],
            "Skewness": stats_dict["skewness"],
            "Shapiro_W": stats_dict["W"],
            "Shapiro_p": stats_dict["p"],
            "Normal_Count": normal_count,
            "Total_Samples": len(numeric_cols)
        })
    
    return pd.DataFrame(results)


# ============================================================================
# BEST TRANSFORMATION SELECTION
# ============================================================================

def find_best_transformation(analysis_df: pd.DataFrame) -> Tuple[str, float]:
    """
    Find the best transformation based on Shapiro-Wilk W statistic.
    
    Parameters:
    -----------
    analysis_df : pd.DataFrame
        Output from analyze_all_transformations
    
    Returns:
    --------
    tuple: (best_transform_name, best_W_value)
    """
    # Remove rows with NaN Shapiro W
    valid_df = analysis_df.dropna(subset=['Shapiro_W'])
    
    if len(valid_df) == 0:
        return ("raw", 0.0)
    
    # Find transformation with highest Shapiro-Wilk W
    best_idx = valid_df['Shapiro_W'].idxmax()
    best_transform = valid_df.loc[best_idx, 'Transformation']
    best_W = valid_df.loc[best_idx, 'Shapiro_W']
    
    return (best_transform, float(best_W))
