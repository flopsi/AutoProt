"""
helpers/io.py

File I/O operations and data cleaning utilities
Handles multi-format file reading (CSV/TSV/Excel) and data validation
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
import streamlit as st

# ============================================================================
# FILE READERS
# Multi-format readers with automatic encoding detection
# ============================================================================
def download_button_excel(df: pd.DataFrame, filename: str, label: str = "Download Excel"):
    """
    Create download button for DataFrame as Excel.
    Falls back to openpyxl if xlsxwriter unavailable.
    
    Args:
        df: DataFrame to download
        filename: Output filename (should end with .xlsx)
        label: Button label
    """
    from io import BytesIO
    
    buffer = BytesIO()
    try:
        engine = 'xlsxwriter'
        with pd.ExcelWriter(buffer, engine=engine) as writer:
            df.to_excel(writer, index=True)
    except ImportError:
        # Fallback to openpyxl
        engine = 'openpyxl'
        with pd.ExcelWriter(buffer, engine=engine) as writer:
            df.to_excel(writer, index=True)
    
    st.download_button(
        label=label,
        data=buffer.getvalue(),
        file_name=filename,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )


@st.cache_data(show_spinner="Loading file...", max_entries=5)
def read_csv(filepath, sep: str = ",") -> pd.DataFrame:
    """
    Read CSV file with automatic encoding detection.
    Cached to avoid re-reading same file on page reruns.
    
    Args:
        filepath: Path or file-like object
        sep: Separator character (default comma)
    
    Returns:
        DataFrame with cleaned column names
    """
    try:
        # Try UTF-8 encoding first
        df = pd.read_csv(filepath, sep=sep, encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback to latin-1 for legacy files
        df = pd.read_csv(filepath, sep=sep, encoding="latin-1")
    
    # Clean whitespace in column names
    df.columns = df.columns.str.strip()
    return df

@st.cache_data(show_spinner="Loading TSV...", max_entries=5)
def read_tsv(filepath) -> pd.DataFrame:
    """Read tab-separated values file."""
    return read_csv(filepath, sep="\t")

@st.cache_data(show_spinner="Loading Excel...", max_entries=5)
def read_excel(filepath, sheet_name: int = 0) -> pd.DataFrame:
    """
    Read Excel file from specified sheet.
    
    Args:
        filepath: Path or file-like object
        sheet_name: Sheet index (0 = first sheet)
    
    Returns:
        DataFrame with cleaned column names
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df.columns = df.columns.str.strip()
    return df

def read_file(filepath, file_format: str = None) -> pd.DataFrame:
    """
    Unified file reader that auto-detects format.
    
    Args:
        filepath: Path or file-like object
        file_format: Optional format override ("csv", "tsv", "excel")
    
    Returns:
        DataFrame with loaded data
    """
    # Auto-detect format from filename if not specified
    if file_format is None:
        name = str(filepath.name if hasattr(filepath, 'name') else filepath)
        if name.endswith('.xlsx') or name.endswith('.xls'):
            file_format = 'excel'
        elif name.endswith('.tsv') or name.endswith('.txt'):
            file_format = 'tsv'
        else:
            file_format = 'csv'
    
    # Route to appropriate reader
    if file_format == 'excel':
        return read_excel(filepath)
    elif file_format == 'tsv':
        return read_tsv(filepath)
    else:
        return read_csv(filepath)

# ============================================================================
# COLUMN DETECTION
# Heuristic detection of special columns (IDs, species, intensities)
# ============================================================================
def detect_quantitative_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect which columns contain quantitative intensity data (float64 only).
    
    Quantitative columns must be float64 dtype for statistical analysis.
    For other numeric types (int32, int64, etc.), users must explicitly select.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of column names with float64 dtype
        
    Example:
        >>> df = pd.DataFrame({'A': [1.0, 2.0], 'B': [1, 2], 'C': ['x', 'y']})
        >>> detect_quantitative_columns(df)
        ['A']  # Only float64, not int64
    """
    quantitative_cols = df.select_dtypes(include=['float64']).columns.tolist()
    return quantitative_cols

def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect which columns contain numeric intensity data.
    
    Args:
        df: Input DataFrame
    
    Returns:
        List of column names with numeric dtype
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols

def detect_protein_id_column(df: pd.DataFrame) -> Optional[str]:
    """
    Heuristically detect protein/gene ID column.
    Looks for common patterns: "Protein", "Gene", "ID", "Accession", etc.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Column name if found, else first non-numeric column
    """
    patterns = ["protein", "gene", "id", "accession", "uniprot", "orf"]
    
    # Check for pattern matches
    for col in df.columns:
        col_lower = col.lower()
        for pattern in patterns:
            if pattern in col_lower:
                return col
    
    # Fallback: first non-numeric column
    for col in df.columns:
        if df[col].dtype == "object" or df[col].dtype == "string":
            return col
    
    # Last fallback: first column that's not purely numeric
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return col
    
    return None


def ensure_protein_id_string(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Ensure protein ID column is string type.
    
    Args:
        df: Input DataFrame
        id_col: Protein ID column name
    
    Returns:
        DataFrame with ID column converted to string
    """
    df = df.copy()
    df[id_col] = df[id_col].astype(str)
    return df
    
def clean_species_name(name: str) -> str:
    """
    Standardize species annotation format.
    Removes underscores, strips whitespace, converts to uppercase.
    
    Args:
        name: Raw species string
    
    Returns:
        Cleaned species name (e.g., "HUMAN", "YEAST")
    """
    if pd.isna(name):
        return name
    return str(name).strip().strip('_').upper()

def detect_species_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detect species/organism annotation column.
    Looks for patterns: "Species", "Organism", "Taxon", etc.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Column name if found, else None
    """
    patterns = ["species", "organism", "taxon", "tax"]
    
    for col in df.columns:
        col_lower = col.lower()
        for pattern in patterns:
            if pattern in col_lower:
                return col
    
    return None

# ============================================================================
# DATA VALIDATION
# Quality checks for uploaded data
# ============================================================================

def validate_numeric_data(df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[bool, str]:
    """
    Validate data quality before analysis.
    Checks: minimum sample count, missing data rate, empty columns.
    
    Args:
        df: Input DataFrame
        numeric_cols: List of numeric column names to validate
    
    Returns:
        (is_valid, message) tuple
        - is_valid: True if data passes all checks
        - message: Explanation of validation result
    """
    # Check 1: Minimum column count
    if len(numeric_cols) == 0:
        return False, "❌ No numeric columns detected."
    
    if len(numeric_cols) < 4:
        return False, f"❌ Only {len(numeric_cols)} numeric columns found. Need at least 4 samples."
    
    # Check 2: Missing data rate
    total_values = df[numeric_cols].size
    missing = df[numeric_cols].isna().sum().sum()
    missing_rate = (missing / total_values * 100)
    
    if missing_rate > 80:
        return False, f"❌ Too much missing data ({missing_rate:.1f}%). Maximum allowed: 80%."
    
    # Check 3: Empty columns
    for col in numeric_cols:
        if df[col].isna().all():
            return False, f"❌ Column '{col}' is entirely missing values."
    
    return True, f"✅ Data validation passed. {len(numeric_cols)} samples, {missing_rate:.1f}% missing."

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names: trim spaces, replace spaces with underscores.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with cleaned column names
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(" ", "_")
    return df

# ============================================================================
# DATA CLEANING
# Filtering and quality control operations
# ============================================================================

def drop_proteins_with_invalid_intensities(
    df: pd.DataFrame,
    intensity_cols: List[str],
    drop_value: Optional[float] = 1.0,
    drop_nan: bool = True,
) -> pd.DataFrame:
    """
    Remove proteins with invalid intensity values.
    Invalid means: NaN (if drop_nan=True) OR equal to drop_value.
    If ANY intensity column for a protein is invalid, that row is removed.
    
    Args:
        df: Input DataFrame
        intensity_cols: List of intensity column names
        drop_value: Value to consider invalid (default 1.0)
        drop_nan: Whether to drop rows with NaN values
    
    Returns:
        Filtered DataFrame with invalid rows removed
    """
    if not intensity_cols:
        return df
    
    sub = df[intensity_cols]
    invalid = False
    
    # Flag NaN values as invalid
    if drop_nan:
        invalid = sub.isna()
    
    # Flag specific value as invalid
    if drop_value is not None:
        if isinstance(invalid, pd.DataFrame):
            invalid = invalid | (sub == drop_value)
        else:
            invalid = (sub == drop_value)
    
    # Drop rows where ANY intensity is invalid
    rows_to_drop = invalid.any(axis=1)
    return df.loc[~rows_to_drop].copy()

def filter_by_missing_rate(
    df: pd.DataFrame,
    intensity_cols: List[str],
    max_missing_rate: float = 0.5
) -> pd.DataFrame:
    """
    Remove proteins exceeding maximum missing rate threshold.
    
    Args:
        df: Input DataFrame
        intensity_cols: Columns to check for missing values
        max_missing_rate: Maximum proportion of missing values (0-1)
    
    Returns:
        Filtered DataFrame
    """
    missing_per_protein = df[intensity_cols].isna().sum(axis=1) / len(intensity_cols)
    keep = missing_per_protein <= max_missing_rate
    return df[keep].copy()

def filter_by_cv(
    df: pd.DataFrame,
    intensity_cols: List[str],
    max_cv: float = 1.0
) -> pd.DataFrame:
    """
    Remove proteins with coefficient of variation exceeding threshold.
    CV = std / mean, measures relative variability.
    
    Args:
        df: Input DataFrame
        intensity_cols: Columns to compute CV from
        max_cv: Maximum allowed CV (default 1.0 = 100%)
    
    Returns:
        Filtered DataFrame
    """
    means = df[intensity_cols].mean(axis=1)
    stds = df[intensity_cols].std(axis=1)
    cv = stds / means
    keep = cv <= max_cv
    return df[keep].copy()


def read_file(file) -> pl.DataFrame:
    """Read uploaded file into Polars DataFrame with robust error handling."""
    name = file.name.lower()
    
    try:
        if name.endswith('.csv'):
            return pl.read_csv(
                file,
                null_values=["#NUM!", "#N/A", "#VALUE!", "#REF!", "#DIV/0!", "#NAME?", "#NULL!", ""],
                ignore_errors=True,
                infer_schema_length=10000
            )
        elif name.endswith(('.tsv', '.txt')):
            return pl.read_csv(
                file,
                separator='\t',
                null_values=["#NUM!", "#N/A", "#VALUE!", "#REF!", "#DIV/0!", "#NAME?", "#NULL!", ""],
                ignore_errors=True,
                infer_schema_length=10000
            )
        elif name.endswith('.xlsx'):
            # Excel files handle #NUM! automatically as NaN
            return pl.read_excel(file)
        else:
            raise ValueError(f"Unsupported format: {name}")
    except Exception as e:
        raise ValueError(f"Error reading {name}: {str(e)}")

def generate_column_names(n: int, replicates: int = 3) -> list:
    """Generate A1, A2, A3, B1, B2, B3, ..."""
    return [f"{chr(65 + i//replicates)}{i%replicates + 1}" for i in range(n)]
