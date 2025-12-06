"""
helpers/transforms.py

Data transformation functions for proteomics intensity normalization
Includes log, power, and quantile transformations with caching
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from typing import Dict, List, Tuple
import streamlit as st

# ============================================================================
# TRANSFORMATION METADATA
# Human-readable names and descriptions for UI display
# ============================================================================

TRANSFORM_NAMES: Dict[str, str] = {
    "raw": "Raw (No Transform)",
    "log2": "Log2",
    "log10": "Log10",
    "ln": "Natural Log (ln)",
    "sqrt": "Square Root",
    "arcsinh": "Arcsinh",
    "boxcox": "Box-Cox",
    "yeo-johnson": "Yeo-Johnson",
    "vst": "Variance Stabilizing (VST)",
    "quantile": "Quantile (Rank-based)",
}

TRANSFORM_DESCRIPTIONS: Dict[str, str] = {
    "raw": "Original data without transformation",
    "log2": "Log base 2 - standard for proteomics fold-change",
    "log10": "Log base 10 transformation",
    "ln": "Natural logarithm (base e)",
    "sqrt": "Square root transformation",
    "arcsinh": "Inverse hyperbolic sine - handles negatives",
    "boxcox": "Box-Cox power transformation (positive values only)",
    "yeo-johnson": "Yeo-Johnson power transformation (handles zeros/negatives)",
    "vst": "Variance stabilizing: asinh(x / 2*median)",
    "quantile": "Rank-based transformation to approximate normal distribution",
}

# ============================================================================
# TRANSFORMATION FUNCTIONS
# Core transformation logic with caching for performance
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
        - For 'raw': returns original df and numeric_cols
        - For others: creates new columns with '_transformed' suffix
    
    Example:
        df_log2, log2_cols = apply_transformation(df, ["A1", "A2"], "log2")
        # Creates: df["A1_transformed"], df["A2_transformed"]
    """
    df_out = df.copy()
    
    # Special case: raw (no transformation)
    if method == "raw":
        return df_out, numeric_cols
    
    # Apply transformation to each numeric column
    for col in numeric_cols:
        if col not in df_out.columns:
            continue
        
        vals = df_out[col].dropna()
        
        # --- Simple log transformations ---
        if method == "log2":
            df_out.loc[vals.index, f"{col}_transformed"] = np.log2(
                vals.clip(lower=1e-10)  # Avoid log(0)
            )
        
        elif method == "log10":
            df_out.loc[vals.index, f"{col}_transformed"] = np.log10(
                vals.clip(lower=1e-10)
            )
        
        elif method == "ln":
            df_out.loc[vals.index, f"{col}_transformed"] = np.log(
                vals.clip(lower=1e-10)
            )
        
        # --- Root transformations ---
        elif method == "sqrt":
            df_out.loc[vals.index, f"{col}_transformed"] = np.sqrt(
                vals.clip(lower=0)
            )
        
        # --- Arcsinh (handles negatives) ---
        elif method == "arcsinh":
            df_out.loc[vals.index, f"{col}_transformed"] = np.arcsinh(vals)
        
        # --- Box-Cox (positive values only) ---
        elif method == "boxcox":
            if (vals > 0).all():
                try:
                    transformed, _ = stats.boxcox(vals)
                    df_out.loc[vals.index, f"{col}_transformed"] = transformed
                except Exception:
                    # Fallback: copy original
                    df_out.loc[vals.index, f"{col}_transformed"] = vals
            else:
                df_out.loc[vals.index, f"{col}_transformed"] = vals
        
        # --- Yeo-Johnson (handles zeros/negatives) ---
        elif method == "yeo-johnson":
            try:
                pt = PowerTransformer(method="yeo-johnson", standardize=False)
                transformed = pt.fit_transform(vals.values.reshape(-1, 1)).ravel()
                df_out.loc[vals.index, f"{col}_transformed"] = transformed
            except Exception:
                df_out.loc[vals.index, f"{col}_transformed"] = vals
        
        # --- Variance Stabilizing Transformation ---
        elif method == "vst":
            median_intensity = vals.median()
            if pd.isna(median_intensity) or median_intensity <= 0:
                median_intensity = 1.0
            df_out.loc[vals.index, f"{col}_transformed"] = np.arcsinh(
                vals / (2 * median_intensity)
            )
        
        # --- Quantile normalization to normal distribution ---
        elif method == "quantile":
            try:
                qt = QuantileTransformer(
                    n_quantiles=min(1000, len(vals)),
                    output_distribution="normal",
                    random_state=0,
                )
                v = vals.to_numpy().reshape(-1, 1)
                v_tr = qt.fit_transform(v).ravel()
                df_out.loc[vals.index, f"{col}_transformed"] = v_tr
            except Exception:
                df_out.loc[vals.index, f"{col}_transformed"] = vals
        
        # --- Unknown method: copy original ---
        else:
            df_out.loc[vals.index, f"{col}_transformed"] = vals
    
    # Collect transformed column names
    transformed_cols = [
        f"{col}_transformed"
        for col in numeric_cols
        if f"{col}_transformed" in df_out.columns
    ]
    
    return df_out, transformed_cols

# ============================================================================
# UTILITY FUNCTIONS
# Helper functions for transformation management
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
