"""
helpers/file_io.py
Multi-format file readers (CSV, TSV, Excel)
Handles different separators, headers, encodings gracefully
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple

# ============================================================================
# FILE READERS
# ============================================================================

def read_csv(filepath: str, sep: str = ",") -> pd.DataFrame:
    """
    Read CSV file with automatic encoding detection.
    
    Args:
        filepath: Path or file-like object
        sep: Separator (default comma)
        
    Returns:
        DataFrame
    """
    try:
        # Try UTF-8 first
        df = pd.read_csv(filepath, sep=sep, encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback to latin-1
        df = pd.read_csv(filepath, sep=sep, encoding="latin-1")
    
    # Clean up whitespace in column names
    df.columns = df.columns.str.strip()
    
    return df


def read_tsv(filepath: str) -> pd.DataFrame:
    """Read TSV (tab-separated) file."""
    return read_csv(filepath, sep="\t")


def read_excel(filepath: str, sheet_name: int = 0) -> pd.DataFrame:
    """
    Read Excel file.
    
    Args:
        filepath: Path or file-like object
        sheet_name: Sheet index (default 0 = first sheet)
        
    Returns:
        DataFrame
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    
    # Clean up whitespace
    df.columns = df.columns.str.strip()
    
    return df


def detect_numeric_columns(df: pd.DataFrame) -> list:
    """
    Detect which columns contain numeric data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of column names with numeric dtype
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols


def detect_protein_id_column(df: pd.DataFrame) -> Optional[str]:
    """
    Heuristically detect protein ID column name.
    Looks for common patterns: "Protein", "Gene", "ID", "Accession", etc.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Column name if found, else None
    """
    patterns = ["protein", "gene", "id", "accession", "uniprot", "orf"]
    
    for col in df.columns:
        col_lower = col.lower()
        for pattern in patterns:
            if pattern in col_lower:
                return col
    
    # Default to first non-numeric column
    for col in df.columns:
        if df[col].dtype == "object":
            return col
    
    return None

def clean_species_name(name: str) -> str:
    if pd.isna(name):
        return name
    return str(name).strip().strip('_').upper()


def detect_species_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detect species annotation column.
    Looks for common patterns: "Species", "Organism", "Taxon", etc.
    
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
# VALIDATION & FORMATTING
# ============================================================================

def validate_numeric_data(df: pd.DataFrame, numeric_cols: list) -> Tuple[bool, str]:
    """
    Validate numeric data quality.
    
    Args:
        df: Input DataFrame
        numeric_cols: List of numeric column names
        
    Returns:
        (is_valid, message)
    """
    if len(numeric_cols) == 0:
        return False, "No numeric columns detected."
    
    if len(numeric_cols) < 4:
        return False, f"Only {len(numeric_cols)} numeric columns. Need at least 4."
    
    total_values = df[numeric_cols].size
    missing = df[numeric_cols].isna().sum().sum()
    missing_rate = (missing / total_values * 100)
    
    if missing_rate > 80:
        return False, f"Too much missing data ({missing_rate:.1f}%)."
    
    # Check if any numeric column is all NaN
    for col in numeric_cols:
        if df[col].isna().all():
            return False, f"Column '{col}' is entirely missing."
    
    return True, "Data validation passed."


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names: remove spaces, standardize case.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with standardized column names
    """
    df = df.copy()
    # Remove leading/trailing spaces
    df.columns = df.columns.str.strip()
    # Replace spaces with underscores
    df.columns = df.columns.str.replace(" ", "_")
    return df
