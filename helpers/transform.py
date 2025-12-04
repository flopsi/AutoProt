"""
helpers/data_transform.py
Data transformation utilities for proteomics analysis
"""

import pandas as pd
import numpy as np
import warnings

# ============================================================================
# LOG2 TRANSFORMATION WITH NAN/ZERO HANDLING
# ============================================================================

def replace_missing_values(
    df: pd.DataFrame,
    numeric_cols: list,
    replacement_value: float = 1.0,
    fill_method: str = 'value'
) -> pd.DataFrame:
    """
    Replace NaN and 0 values with replacement value for log2 transformation.
    
    Args:
        df: Input dataframe
        numeric_cols: List of numeric column names
        replacement_value: Value to replace NaN/0 with (default: 1.0)
        fill_method: 'value' (use replacement_value) or 'min' (use column min)
    
    Returns:
        DataFrame with NaN/0 replaced
        
    Example:
        df = replace_missing_values(df, ['Control_R1', 'Control_R2'], replacement_value=1.0)
        # Now log2(df) will work without warnings
    """
    df_clean = df.copy()
    
    for col in numeric_cols:
        if col not in df_clean.columns:
            continue
        
        # Count before replacement
        n_nan_before = df_clean[col].isna().sum()
        n_zero_before = (df_clean[col] == 0).sum()
        
        if fill_method == 'min':
            # Use minimum non-zero value in column
            col_min = df_clean[col][(df_clean[col] > 0) & (df_clean[col].notna())].min()
            if pd.isna(col_min) or col_min <= 0:
                col_min = replacement_value
        else:
            col_min = replacement_value
        
        # Replace NaN
        df_clean[col] = df_clean[col].fillna(col_min)
        
        # Replace 0
        df_clean.loc[df_clean[col] == 0, col] = col_min
        
        # Log replacement info
        n_replaced = n_nan_before + n_zero_before
        if n_replaced > 0:
            print(f"  {col}: Replaced {n_nan_before} NaN + {n_zero_before} zeros → {col_min}")
    
    return df_clean


def log2_transform(
    df: pd.DataFrame,
    numeric_cols: list,
    handle_missing: bool = True,
    replacement_value: float = 1.0
) -> pd.DataFrame:
    """
    Apply log2 transformation to numeric columns.
    
    Handles NaN and 0 values automatically:
    - NaN → replaced with replacement_value
    - 0 → replaced with replacement_value
    - Then log2 applied
    
    Args:
        df: Input dataframe
        numeric_cols: List of numeric column names to transform
        handle_missing: If True, replace NaN/0 before log2
        replacement_value: Value to use for replacements (default: 1.0)
    
    Returns:
        DataFrame with log2-transformed values
        
    Example:
        df_log2 = log2_transform(df, ['Control_R1', 'Control_R2'], replacement_value=1.0)
        # Safe to use without NaN warnings
    """
    df_transformed = df.copy()
    
    if handle_missing:
        print(f"Replacing NaN and 0 values with {replacement_value}...")
        df_transformed = replace_missing_values(
            df_transformed,
            numeric_cols,
            replacement_value=replacement_value
        )
    
    print(f"Applying log2 transformation to {len(numeric_cols)} columns...")
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        for col in numeric_cols:
            if col in df_transformed.columns:
                df_transformed[col] = np.log2(df_transformed[col])
    
    print(f"✅ Log2 transformation complete")
    
    return df_transformed


def log10_transform(
    df: pd.DataFrame,
    numeric_cols: list,
    handle_missing: bool = True,
    replacement_value: float = 1.0
) -> pd.DataFrame:
    """
    Apply log10 transformation to numeric columns.
    Same NaN/0 handling as log2_transform.
    """
    df_transformed = df.copy()
    
    if handle_missing:
        print(f"Replacing NaN and 0 values with {replacement_value}...")
        df_transformed = replace_missing_values(
            df_transformed,
            numeric_cols,
            replacement_value=replacement_value
        )
    
    print(f"Applying log10 transformation to {len(numeric_cols)} columns...")
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        for col in numeric_cols:
            if col in df_transformed.columns:
                df_transformed[col] = np.log10(df_transformed[col])
    
    print(f"✅ Log10 transformation complete")
    
    return df_transformed


def normalize_columns(
    df: pd.DataFrame,
    numeric_cols: list,
    method: str = 'zscore'
) -> pd.DataFrame:
    """
    Normalize numeric columns using various methods.
    
    Args:
        df: Input dataframe
        numeric_cols: List of numeric columns
        method: 'zscore' (default), 'minmax', 'quantile'
    
    Returns:
        Normalized dataframe
    """
    df_norm = df.copy()
    
    for col in numeric_cols:
        if col not in df_norm.columns:
            continue
        
        if method == 'zscore':
            mean = df_norm[col].mean()
            std = df_norm[col].std()
            df_norm[col] = (df_norm[col] - mean) / std
        
        elif method == 'minmax':
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        
        elif method == 'quantile':
            q25 = df_norm[col].quantile(0.25)
            q75 = df_norm[col].quantile(0.75)
            df_norm[col] = (df_norm[col] - q25) / (q75 - q25)
    
    return df_norm


# ============================================================================
# QUALITY CHECK BEFORE TRANSFORMATION
# ============================================================================

def check_data_quality_for_log(
    df: pd.DataFrame,
    numeric_cols: list
) -> dict:
    """
    Check data quality before log transformation.
    
    Returns:
        Dictionary with quality metrics
    """
    quality = {
        'total_values': 0,
        'nan_count': 0,
        'zero_count': 0,
        'negative_count': 0,
        'safe_to_log': True,
        'issues': []
    }
    
    for col in numeric_cols:
        if col not in df.columns:
            continue
        
        col_data = df[col]
        quality['total_values'] += len(col_data)
        quality['nan_count'] += col_data.isna().sum()
        quality['zero_count'] += (col_data == 0).sum()
        quality['negative_count'] += (col_data < 0).sum()
        
        if (col_data < 0).any():
            quality['issues'].append(f"Column '{col}' has negative values")
            quality['safe_to_log'] = False
    
    pct_nan = (quality['nan_count'] / quality['total_values'] * 100) if quality['total_values'] > 0 else 0
    pct_zero = (quality['zero_count'] / quality['total_values'] * 100) if quality['total_values'] > 0 else 0
    
    quality['nan_pct'] = pct_nan
    quality['zero_pct'] = pct_zero
    
    return quality


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

"""
In your analysis page:

from helpers.data_transform import log2_transform, check_data_quality_for_log

# Get protein data
protein_data = st.session_state.protein_data
df = protein_data.raw.copy()
numeric_cols = protein_data.numeric_cols

# Check quality first
quality = check_data_quality_for_log(df, numeric_cols)
st.write(f"NaN values: {quality['nan_count']} ({quality['nan_pct']:.1f}%)")
st.write(f"Zero values: {quality['zero_count']} ({quality['zero_pct']:.1f}%)")

# Apply log2 with automatic NaN/0 handling
df_log2 = log2_transform(
    df,
    numeric_cols,
    handle_missing=True,
    replacement_value=1.0
)

# Now plotting works!
fig = create_log2_plot(df_log2[numeric_cols])
st.plotly_chart(fig)
"""
