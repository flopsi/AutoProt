"""
helpers/io.py - DATA INPUT/OUTPUT UTILITIES
File loading, validation, and format detection
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import streamlit as st

# ============================================================================
# FILE LOADING
# ============================================================================

@st.cache_data(ttl=3600)
def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """Load CSV file with sensible defaults."""
    return pd.read_csv(file_path, index_col=None, **kwargs)


@st.cache_data(ttl=3600)
def load_excel(file_path: str, sheet_name: int = 0, **kwargs) -> pd.DataFrame:
    """Load Excel file."""
    return pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)

# ============================================================================
# COLUMN DETECTION
# ============================================================================

def detect_numeric_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Separate numeric and categorical columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (numeric_columns, categorical_columns)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    return numeric_cols, categorical_cols


def detect_sample_columns(columns: List[str]) -> List[str]:
    """
    Detect which columns are likely sample/condition columns.
    Heuristic: numeric-sounding names, not ID-like.
    
    Args:
        columns: List of column names
        
    Returns:
        List of likely sample columns
    """
    id_keywords = ['id', 'name', 'identifier', 'gene', 'protein', 'uniprot', 
                   'accession', 'description', 'species', 'organism', 'sequence']
    
    sample_cols = []
    for col in columns:
        col_lower = str(col).lower()
        is_id_like = any(keyword in col_lower for keyword in id_keywords)
        
        if not is_id_like:
            sample_cols.append(col)
    
    return sample_cols

# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_dataframe(
    df: pd.DataFrame,
    id_col: str,
    numeric_cols: List[str],
    min_rows: int = 1,
    min_cols: int = 1
) -> Tuple[bool, str]:
    """
    Validate DataFrame structure and contents.
    
    Args:
        df: DataFrame to validate
        id_col: ID column name
        numeric_cols: Numeric column names
        min_rows: Minimum required rows
        min_cols: Minimum required columns
        
    Returns:
        Tuple of (is_valid, message)
    """
    # Check DataFrame not empty
    if df.empty:
        return False, "DataFrame is empty"
    
    # Check minimum dimensions
    if len(df) < min_rows:
        return False, f"DataFrame has {len(df)} rows, need at least {min_rows}"
    
    if len(numeric_cols) < min_cols:
        return False, f"Only {len(numeric_cols)} numeric columns, need at least {min_cols}"
    
    # Check ID column exists
    if id_col not in df.columns:
        return False, f"ID column '{id_col}' not found"
    
    # Check numeric columns exist
    missing_cols = [col for col in numeric_cols if col not in df.columns]
    if missing_cols:
        return False, f"Missing columns: {', '.join(missing_cols)}"
    
    # Check numeric columns are actually numeric
    non_numeric = []
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            # Try to coerce
            try:
                pd.to_numeric(df[col], errors='coerce')
            except:
                non_numeric.append(col)
    
    if non_numeric:
        return False, f"Non-numeric columns: {', '.join(non_numeric)}"
    
    return True, "Validation passed"


def check_duplicates(df: pd.DataFrame, id_col: str) -> Tuple[int, List]:
    """
    Check for duplicate IDs.
    
    Args:
        df: DataFrame to check
        id_col: ID column name
        
    Returns:
        Tuple of (n_duplicates, list_of_duplicate_ids)
    """
    duplicates = df[df.duplicated(subset=[id_col], keep=False)][id_col].unique().tolist()
    return len(duplicates), duplicates


def check_missing_data(
    df: pd.DataFrame,
    numeric_cols: List[str]
) -> dict:
    """
    Detailed missing data analysis.
    
    Args:
        df: DataFrame to analyze
        numeric_cols: Numeric columns
        
    Returns:
        Dictionary with missing data statistics
    """
    missing_by_row = df[numeric_cols].isna().sum(axis=1)
    missing_by_col = df[numeric_cols].isna().sum()
    
    return {
        'total_cells': len(df) * len(numeric_cols),
        'missing_cells': df[numeric_cols].isna().sum().sum(),
        'missing_pct': round(df[numeric_cols].isna().sum().sum() / (len(df) * len(numeric_cols)) * 100, 2),
        'rows_with_missing': (missing_by_row > 0).sum(),
        'cols_with_missing': (missing_by_col > 0).sum(),
        'missing_by_row_max': missing_by_row.max(),
        'missing_by_col_max': missing_by_col.max(),
    }

# ============================================================================
# DATA STATISTICS
# ============================================================================

def get_data_summary(
    df: pd.DataFrame,
    numeric_cols: List[str],
    id_col: str
) -> dict:
    """
    Get comprehensive data summary.
    
    Args:
        df: DataFrame
        numeric_cols: Numeric columns
        id_col: ID column
        
    Returns:
        Summary dictionary
    """
    return {
        'n_rows': len(df),
        'n_samples': len(numeric_cols),
        'n_features': df.shape[1],
        'duplicate_ids': check_duplicates(df, id_col)[0],
        'missing_summary': check_missing_data(df, numeric_cols),
        'numeric_summary': {
            'min': df[numeric_cols].min().min(),
            'max': df[numeric_cols].max().max(),
            'mean': df[numeric_cols].mean().mean(),
            'median': df[numeric_cols].median().median(),
        }
    }

# ============================================================================
# DATA EXPORT
# ============================================================================

def export_to_csv(df: pd.DataFrame, filename: str) -> bytes:
    """Export DataFrame to CSV bytes."""
    return df.to_csv(index=False).encode('utf-8')


def export_to_excel(df: pd.DataFrame, filename: str) -> bytes:
    """Export DataFrame to Excel bytes."""
    import io
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine='openpyxl')
    return buffer.getvalue()
