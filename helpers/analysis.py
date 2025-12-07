"""
helpers/analysis.py - OPTIMIZED
Condition detection, filtering, and grouping logic for proteins/peptides
Eliminates duplicate logic from pages - single source of truth
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
    Example: ['A1', 'A2', 'B1', 'B2'] → ['A', 'B']
    
    Args:
        numeric_cols: List of sample column names
        
    Returns:
        Sorted list of unique conditions
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
    """
    return [col for col in numeric_cols if col.startswith(condition)]


@st.cache_data
def create_condition_mapping(numeric_cols: List[str]) -> Dict[str, str]:
    """
    Create mapping of each column to its condition.
    
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
    
    Args:
        df: Input DataFrame
        numeric_cols: Columns to analyze
        condition_mapping: Dict mapping columns to conditions
        max_cv: Maximum CV% allowed
        
    Returns:
        Filtered DataFrame
    """
    if not numeric_cols:
        return df
    
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
    """Check if at least 2 conditions exist for comparison."""
    return len(conditions) >= 2


def validate_numeric_cols(numeric_cols: List[str]) -> bool:
    """Check if numeric columns exist."""
    return len(numeric_cols) > 0
