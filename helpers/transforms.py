"""
helpers/transforms.py
Data transformation functions
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

def apply_transform(df: pd.DataFrame, numeric_cols: list, method: str = "log2") -> pd.DataFrame:
    """
    Apply transformation to numeric columns.
    
    Args:
        df: Input dataframe
        numeric_cols: List of numeric column names
        method: Transform method ("log2", "log10", "sqrt", "cbrt", "yeo_johnson", "quantile")
        
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
        pt = stats.PowerTransformer(method='yeo-johnson', standardize=False)
        df_transformed[numeric_cols] = pt.fit_transform(df_transformed[numeric_cols])
    
    elif method == "quantile":
        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer(output_distribution='normal')
        df_transformed[numeric_cols] = qt.fit_transform(df_transformed[numeric_cols])
    
    # Handle infinite values
    df_transformed[numeric_cols] = df_transformed[numeric_cols].replace([np.inf, -np.inf], 0)
    
    return df_transformed
