"""
helpers/transforms.py
Simple, robust transformations from working Colab code
"""
import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PowerTransformer
from typing import Dict, Optional
import streamlit as st
from typing import List, Tuple

TRANSFORM_NAMES = {
    'raw': 'Raw (No Transform)',
    'log2': 'Log2',
    'log10': 'Log10', 
    'ln': 'Natural Log (ln)',
    'sqrt': 'Square Root',
    'arcsinh': 'Arcsinh',
    'boxcox': 'Box-Cox',
    'yeo-johnson': 'Yeo-Johnson',
    'vst': 'Variance Stabilizing (VST)'
}

TRANSFORM_DESCRIPTIONS = {
    'raw': 'Original data without transformation',
    'log2': 'Log base 2 - standard for proteomics fold-change',
    'log10': 'Log base 10 transformation',
    'ln': 'Natural logarithm (base e)',
    'sqrt': 'Square root transformation',
    'arcsinh': 'Inverse hyperbolic sine - handles negatives',
    'boxcox': 'Box-Cox power transformation (positive values only)',
    'yeo-johnson': 'Yeo-Johnson power transformation (handles zeros/negatives)',
    'vst': 'Variance stabilizing transformation (asinh(x/median))'
}


def apply_transformation(df: pd.DataFrame, numeric_cols: list, method: str = 'log2') -> pd.DataFrame:
    """
    Apply specified transformation to intensity data (from working Colab code).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    numeric_cols : list
        Intensity columns to transform
    method : str
        'raw', 'log2', 'log10', 'ln', 'sqrt', 'arcsinh', 'boxcox', 'yeo-johnson', 'vst'
    
    Returns:
    --------
    pd.DataFrame with transformed columns suffixed '_transformed'
    """
    df_out = df.copy()
    
    if method == 'raw':
        # Return original data unchanged
        return df_out
    
    # Filter out non-numeric/NaN values for each column
    for col in numeric_cols:
        if col not in df_out.columns:
            continue
            
        vals = df_out[col].dropna()
        
        if method == 'log2':
            df_out.loc[vals.index, f'{col}_transformed'] = np.log2(vals.clip(lower=1e-10))
            
        elif method == 'log10':
            df_out.loc[vals.index, f'{col}_transformed'] = np.log10(vals.clip(lower=1e-10))
            
        elif method == 'ln':
            df_out.loc[vals.index, f'{col}_transformed'] = np.log(vals.clip(lower=1e-10))
            
        elif method == 'sqrt':
            df_out.loc[vals.index, f'{col}_transformed'] = np.sqrt(vals.clip(lower=0))
            
        elif method == 'arcsinh':
            df_out.loc[vals.index, f'{col}_transformed'] = np.arcsinh(vals)
            
        elif method == 'boxcox':
            # Box-Cox requires positive values
            if (vals > 0).all():
                try:
                    transformed, _ = stats.boxcox(vals)
                    df_out.loc[vals.index, f'{col}_transformed'] = transformed
                except:
                    df_out.loc[vals.index, f'{col}_transformed'] = vals
            else:
                df_out.loc[vals.index, f'{col}_transformed'] = vals
                
        elif method == 'yeo-johnson':
            try:
                pt = PowerTransformer(method='yeo-johnson', standardize=False)
                transformed = pt.fit_transform(vals.values.reshape(-1, 1)).flatten()
                df_out.loc[vals.index, f'{col}_transformed'] = transformed
            except:
                df_out.loc[vals.index, f'{col}_transformed'] = vals
                
        elif method == 'vst':
            # Simple VST: asinh(x / 2*median)
            median_intensity = vals.median()
            if pd.isna(median_intensity) or median_intensity <= 0:
                median_intensity = 1.0
            df_out.loc[vals.index, f'{col}_transformed'] = np.arcsinh(vals / (2 * median_intensity))
            
        else:
            st.error(f"Unknown transformation: {method}")
            df_out.loc[vals.index, f'{col}_transformed'] = vals
    
    # Get transformed column names
    transformed_cols = [f'{col}_transformed' for col in numeric_cols if f'{col}_transformed' in df_out.columns]
    
    return df_out, transformed_cols


def get_transformed_data(df: pd.DataFrame, numeric_cols: list, method: str) -> tuple:
    """
    Convenience wrapper: returns transformed dataframe + column names.
    """
    df_transformed, transformed_cols = apply_transformation(df, numeric_cols, method)
    return df_transformed, transformed_cols

@st.cache_data(show_spinner=False)
def cached_apply_transformation(
    df: pd.DataFrame,
    numeric_cols: List[str],
    method: str,
    file_hash: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Cached wrapper around apply_transformation.
    file_hash should change when a new file is uploaded (e.g. path or checksum).
    """
    return apply_transformation(df, numeric_cols, method)
