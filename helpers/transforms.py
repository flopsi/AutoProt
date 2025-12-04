"""
helpers/transforms.py
Data transformation functions - ENHANCED VERSION
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, RobustScaler

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
        - robust: Robust scaling (median/IQR) [NEW]
        - zscore: Z-score standardization [NEW]
        - minmax: Min-max scaling [NEW]
        
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
        # NEW: Robust scaling using median and IQR (good for outliers)
        scaler = RobustScaler()
        df_transformed[numeric_cols] = scaler.fit_transform(df_transformed[numeric_cols])
    
    elif method == "zscore":
        # NEW: Z-score standardization (mean=0, std=1)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df_transformed[numeric_cols] = scaler.fit_transform(df_transformed[numeric_cols])
    
    elif method == "minmax":
        # NEW: Min-max scaling to [0, 1]
        from sklearn.preprocessing import MinMaxScaler
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
    
    # Note: yeo_johnson, quantile, robust, zscore, minmax require fitted scaler to inverse
    # Store scaler in session_state if you need inverse for these
    
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
            "handles_negatives": False
        },
        "log10": {
            "name": "Log10 Transform",
            "use_case": "Wide dynamic range, easier to interpret powers of 10",
            "interpretation": "Difference of 1 = 10-fold change",
            "handles_negatives": False
        },
        "sqrt": {
            "name": "Square Root",
            "use_case": "Gentle transform for count data, stabilizes variance",
            "interpretation": "Less aggressive than log",
            "handles_negatives": False
        },
        "robust": {
            "name": "Robust Scaling",
            "use_case": "Remove outlier influence using median/IQR",
            "interpretation": "Centered at median, scaled by IQR",
            "handles_negatives": True
        },
        "zscore": {
            "name": "Z-Score Standardization",
            "use_case": "Compare across different scales",
            "interpretation": "Units of standard deviations from mean",
            "handles_negatives": True
        }
    }
    
    return info.get(method, {"name": method, "use_case": "Unknown", "interpretation": "N/A"})
