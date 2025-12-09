"""
helpers/analysis.py - Analysis Utilities

Condition detection, filtering, and statistical functions.
Single source of truth to eliminate duplicate logic from pages.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


# ============================================================================
# CONDITION DETECTION
# ============================================================================

@st.cache_data
def detect_conditions_from_columns(numeric_cols: List[str]) -> List[str]:
    """
    Auto-detect experimental conditions from column names.
    
    Extracts the condition identifier (typically letters before digits).
    
    Args:
        numeric_cols: List of sample column names
    
    Returns:
        Sorted list of unique conditions
    
    Example:
        >>> detect_conditions_from_columns(['A1', 'A2', 'B1', 'B2'])
        ['A', 'B']
    """
    if not numeric_cols:
        return []
    
    # Extract first character(s) until digit is found
    conditions = set()
    for col in numeric_cols:
        condition = ''.join([c for c in col if not c.isdigit()]).strip('_- ')
        if condition:
            conditions.add(condition)
    
    return sorted(list(conditions))


@st.cache_data
def group_columns_by_condition(
    numeric_cols: List[str],
    condition: str
) -> List[str]:
    """
    Get all columns belonging to a specific condition.
    
    Args:
        numeric_cols: All sample columns
        condition: Target condition label (e.g., 'A')
    
    Returns:
        List of columns matching the condition
    
    Example:
        >>> group_columns_by_condition(['A1', 'A2', 'B1'], 'A')
        ['A1', 'A2']
    """
    return [col for col in numeric_cols if col.startswith(condition)]


@st.cache_data
def create_condition_mapping(numeric_cols: List[str]) -> Dict[str, str]:
    """
    Create mapping of each column to its condition.
    
    Args:
        numeric_cols: All sample columns
    
    Returns:
        Dict like {'A1': 'A', 'A2': 'A', 'B1': 'B', 'B2': 'B'}
    """
    mapping = {}
    for col in numeric_cols:
        condition = ''.join([c for c in col if not c.isdigit()]).strip('_- ')
        mapping[col] = condition if condition else col
    
    return mapping


# ============================================================================
# FILTERING FUNCTIONS
# ============================================================================

@st.cache_data
def filter_by_missing_rate(
    df: pd.DataFrame,
    numeric_cols: List[str],
    max_missing_percent: float = 50.0
) -> pd.DataFrame:
    """
    Filter rows where missing values exceed threshold.
    
    Args:
        df: Input DataFrame
        numeric_cols: Columns to check for missing values
        max_missing_percent: Max allowed missing % per row
    
    Returns:
        Filtered DataFrame
    
    Example:
        >>> df = pd.DataFrame({
        ...     'Protein': ['P1', 'P2', 'P3'],
        ...     'A1': [100, 100, np.nan],
        ...     'A2': [100, np.nan, np.nan]
        ... })
        >>> filtered = filter_by_missing_rate(df, ['A1', 'A2'], max_missing_percent=50)
        >>> len(filtered)
        2  # P3 removed (100% missing)
    """
    if not numeric_cols:
        return df
    
    missing_rates = df[numeric_cols].isna().sum(axis=1) / len(numeric_cols) * 100
    return df[missing_rates <= max_missing_percent].copy()


@st.cache_data
def filter_by_cv(
    df: pd.DataFrame,
    numeric_cols: List[str],
    condition_mapping: Dict[str, str],
    max_cv: float = 100.0
) -> pd.DataFrame:
    """
    Filter by coefficient of variation within conditions.
    
    Coefficient of Variation (CV) = (std / mean) * 100
    
    Use case: Remove proteins with high replicate variability (potential contaminants)
    
    Args:
        df: Input DataFrame with proteins/peptides as rows, samples as columns
        numeric_cols: Names of abundance columns (samples)
        condition_mapping: Dict like {'A1': 'CondA', 'A2': 'CondA', 'B1': 'CondB'}
        max_cv: Maximum CV% allowed (default: 100 = no filtering)
    
    Returns:
        DataFrame with rows where all conditions have CV ≤ max_cv
    
    Raises:
        ValueError: If numeric_cols is empty
        KeyError: If condition_mapping doesn't contain all numeric columns
    """
    if not numeric_cols:
        raise ValueError("numeric_cols cannot be empty")
    if not condition_mapping:
        raise ValueError("condition_mapping cannot be None")
    
    missing = set(numeric_cols) - set(condition_mapping.keys())
    if missing:
        raise KeyError(f"Columns not in mapping: {missing}")
    
    conditions = set(condition_mapping.values())
    keep_rows = pd.Series([True] * len(df), index=df.index)
    
    for condition in conditions:
        cols = [c for c in numeric_cols if condition_mapping.get(c) == condition]
        if not cols:
            continue
        
        # Calculate CV for each row within this condition
        subset = df[cols].dropna(axis=1)
        
        if len(subset.columns) < 2:
            continue
        
        means = subset.mean(axis=1)
        stds = subset.std(axis=1)
        
        # Use safeguard against division by zero
        cvs = (stds / (means + 1e-10)) * 100
        
        keep_rows = keep_rows & (cvs <= max_cv)
    
    return df[keep_rows].copy()


@st.cache_data
def filter_by_intensity(
    df: pd.DataFrame,
    numeric_cols: List[str],
    min_intensity: float = 1.0
) -> pd.DataFrame:
    """
    Filter rows where at least one sample exceeds minimum intensity.
    
    Args:
        df: Input DataFrame
        numeric_cols: Columns to check
        min_intensity: Minimum intensity threshold
    
    Returns:
        Filtered DataFrame
    
    Example:
        >>> df = pd.DataFrame({
        ...     'Protein': ['P1', 'P2', 'P3'],
        ...     'A1': [0.5, 100.0, 50.0],
        ...     'A2': [0.3, 110.0, 55.0]
        ... })
        >>> filtered = filter_by_intensity(df, ['A1', 'A2'], min_intensity=50.0)
        >>> len(filtered)
        2  # P1 removed (all values < 50)
    """
    if not numeric_cols:
        return df
    
    subset = df[numeric_cols]
    has_valid = (subset >= min_intensity).any(axis=1)
    
    return df[has_valid].copy()


@st.cache_data
def filter_by_valid_samples(
    df: pd.DataFrame,
    numeric_cols: List[str],
    condition_mapping: Dict[str, str],
    min_valid_per_condition: int = 2
) -> pd.DataFrame:
    """
    Filter rows with insufficient valid values per condition.
    
    Args:
        df: Input DataFrame
        numeric_cols: Sample columns
        condition_mapping: Column to condition mapping
        min_valid_per_condition: Minimum valid values per condition
    
    Returns:
        Filtered DataFrame
    
    Example:
        >>> df = pd.DataFrame({
        ...     'Protein': ['P1', 'P2'],
        ...     'A1': [100, 100],
        ...     'A2': [100, np.nan]
        ... })
        >>> cond_map = {'A1': 'A', 'A2': 'A'}
        >>> filtered = filter_by_valid_samples(df, ['A1', 'A2'], cond_map, min_valid_per_condition=2)
        >>> len(filtered)
        1  # P2 removed (only 1 valid sample in condition A)
    """
    if not numeric_cols:
        return df
    
    conditions = set(condition_mapping.values())
    keep_rows = pd.Series([True] * len(df), index=df.index)
    
    for condition in conditions:
        cols = [c for c in numeric_cols if condition_mapping.get(c) == condition]
        if not cols:
            continue
        
        valid_count = df[cols].notna().sum(axis=1)
        keep_rows = keep_rows & (valid_count >= min_valid_per_condition)
    
    return df[keep_rows].copy()


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

@st.cache_data
def compute_filtering_summary(
    df_original: pd.DataFrame,
    df_filtered: pd.DataFrame,
    id_col: str
) -> Dict[str, any]:
    """
    Compute summary statistics for filtering operations.
    
    Args:
        df_original: DataFrame before filtering
        df_filtered: DataFrame after filtering
        id_col: ID column name
    
    Returns:
        Dict with 'original', 'final', 'removed', 'removed_pct' keys
    """
    original_count = len(df_original)
    final_count = len(df_filtered)
    removed = original_count - final_count
    removed_pct = (removed / original_count * 100) if original_count > 0 else 0
    
    return {
        'original': original_count,
        'final': final_count,
        'removed': removed,
        'removed_pct': round(removed_pct, 1)
    }


@st.cache_data
def compute_sample_stats(
    df: pd.DataFrame,
    numeric_cols: List[str]
) -> pd.DataFrame:
    """
    Compute statistics for each sample (column).
    
    Args:
        df: Input DataFrame
        numeric_cols: Numeric columns to analyze
    
    Returns:
        DataFrame with count, mean, median, std for each sample
    """
    stats = []
    
    for col in numeric_cols:
        data = df[col].dropna()
        stats.append({
            'Sample': col,
            'N': len(data),
            'Mean': round(data.mean(), 2),
            'Median': round(data.median(), 2),
            'Std': round(data.std(), 2),
            'Min': round(data.min(), 2),
            'Max': round(data.max(), 2)
        })
    
    return pd.DataFrame(stats)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_conditions(conditions: List[str]) -> bool:
    """
    Check if at least 2 conditions exist for comparison.
    
    Args:
        conditions: List of condition labels
    
    Returns:
        True if 2 or more conditions exist
    """
    return len(conditions) >= 2


def validate_numeric_cols(numeric_cols: List[str]) -> bool:
    """
    Check if numeric columns exist.
    
    Args:
        numeric_cols: List of numeric column names
    
    Returns:
        True if numeric columns exist
    """
    return len(numeric_cols) > 0
