"""
helpers/naming.py - COLUMN NAME UTILITIES
Functions to clean, trim, and standardize column names for display
UPDATED: Enhanced standardize_condition_names to prioritize replicate detection at the end of the name.
"""

import pandas as pd
from typing import List, Dict, Tuple
import re

# ============================================================================
# NAME CLEANING
# ============================================================================

def trim_name(name: str, max_length: int = 20) -> str:
    """
    Trim column name to reasonable length for plotting.
    
    Args:
        name: Original name
        max_length: Maximum length (default 20)
        
    Returns:
        Trimmed name
    """
    name = str(name).strip()
    
    if len(name) <= max_length:
        return name
    
    # Try to trim intelligently
    if '_' in name:
        # Abbreviate by taking first letter of each segment
        parts = name.split('_')
        if len(parts) > 1:
            abbreviated = ''.join([p[0] for p in parts if p])
            if len(abbreviated) <= max_length:
                return abbreviated
    
    # Fall back to truncation
    return name[:max_length-1] + '…'


def clean_name(name: str) -> str:
    """
    Clean column name by removing special characters and standardizing.
    
    Args:
        name: Original name
        
    Returns:
        Cleaned name
    """
    # Remove leading/trailing whitespace
    name = str(name).strip()
    
    # Replace multiple spaces with single space
    name = re.sub(r'\s+', ' ', name)
    
    # Remove special characters except underscore and dash
    name = re.sub(r'[^a-zA-Z0-9_\-\s]', '', name)
    
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    
    return name


def abbreviate_name(name: str, style: str = 'short') -> str:
    """
    Create abbreviation from column name.
    
    Args:
        name: Original name
        style: 'short' (first 3 chars) or 'smart' (first letter per segment)
        
    Returns:
        Abbreviated name
    """
    name = str(name).strip()
    
    if style == 'smart':
        # Smart abbreviation: first letter of each segment
        if '_' in name:
            parts = name.split('_')
            return ''.join([p[0].upper() for p in parts if p])
        elif ' ' in name:
            parts = name.split()
            return ''.join([p[0].upper() for p in parts if p])
        else:
            return name[:3].upper()
    
    else:  # 'short'
        return name[:3].upper()


def standardize_condition_names(columns: List[str]) -> Dict[str, str]:
    """
    Create mapping of original → standardized condition names (Condition_R#).
    PRIORITY FIX: Replicate number detection starts from the end of the name.
    
    Args:
        columns: List of column names
        
    Returns:
        Dict mapping original name → display name
    """
    mapping = {}
    
    for col in columns:
        col_str = str(col).strip()
        
        # 1. Look for a replicate/run number at the very end (e.g., _01.raw, R3, -5, _1)
        # Pattern looks for: [separator] [R/r/S optional] [one or more digits] [optional ending chars] [END]
        match_end = re.search(r'[_\-\s][rR]?(\d+)(?:\.[a-zA-Z0-9]+)?$', col_str)
        
        if match_end:
            replicate = match_end.group(1).lstrip('0')
            
            # Extract everything before the replicate number/separator as the condition name
            condition_raw = col_str[:match_end.start(0)].strip('_- ')
            
            # Clean up the condition name (remove file extensions, extraneous separators)
            # Remove any common separators at the end of the condition name
            condition = re.sub(r'[\._-]+$', '', condition_raw) 
            
            # If the condition name is still empty or too generic (e.g., just '_'), use the full raw name
            if not condition:
                 mapping[col_str] = trim_name(col_str)
                 continue
                 
            # Use only alphanumeric characters for the condition name and capitalize
            condition = re.sub(r'[^a-zA-Z0-9]', '', condition).upper()
            
            mapping[col_str] = f"{condition}_R{replicate}"
            continue

        # 2. Fallback to the old simple letter/number pattern (A1, B2)
        match_simple = re.search(r'([a-zA-Z]+)[_\-\s]*(\d+)', col_str)
        if match_simple:
            condition = match_simple.group(1).upper()
            replicate = match_simple.group(2)
            mapping[col_str] = f"{condition}_R{replicate}"
            continue

        # 3. Final Fallback to truncation
        mapping[col_str] = trim_name(col_str)
    
    return mapping


def create_short_labels(columns: List[str], length: int = 10) -> Dict[str, str]:
    """
    Create mapping of original → shortened labels for dense plots.
    
    Args:
        columns: List of column names
        length: Target length for labels
        
    Returns:
        Dict mapping original name → short label
    """
    mapping = {}
    
    for col in columns:
        # Use abbreviation if name is long
        if len(str(col)) > length:
            mapping[str(col)] = abbreviate_name(str(col), style='smart')
        else:
            mapping[str(col)] = str(col).strip()
    
    return mapping


# ============================================================================
# RENAMING DATAFRAMES
# ============================================================================

def rename_columns_for_display(
    df: pd.DataFrame,
    columns: List[str],
    style: str = 'smart'
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Rename specified columns for better display in plots.
    
    Args:
        df: Input DataFrame
        columns: Columns to rename (typically sample columns)
        style: 'trim', 'clean', 'smart' (condition-aware), or 'short'
        
    Returns:
        Tuple of (renamed_df, mapping_dict)
    """
    if style == 'trim':
        mapping = {col: trim_name(col) for col in columns}
    elif style == 'clean':
        mapping = {col: clean_name(col) for col in columns}
    elif style == 'smart':
        # Use the condition-aware standardization
        mapping = standardize_condition_names(columns)
    elif style == 'short':
        mapping = create_short_labels(columns)
    else:
        mapping = {col: col for col in columns}
    
    df_renamed = df.rename(columns=mapping)
    return df_renamed, mapping


def reverse_name_mapping(mapping: Dict[str, str]) -> Dict[str, str]:
    """Reverse a name mapping (original → display) to (display → original)."""
    return {v: k for k, v in mapping.items()}


# ============================================================================
# PLOT-SPECIFIC HELPERS
# ============================================================================

def get_display_names(columns: List[str], max_chars: int = 15) -> List[str]:
    """
    Get display-friendly names for columns in plots.
    
    Args:
        columns: List of column names
        max_chars: Maximum characters per name
        
    Returns:
        List of display names
    """
    return [trim_name(str(col), max_length=max_chars) for col in columns]


def get_abbreviated_names(columns: List[str]) -> List[str]:
    """Get abbreviated names for compact display."""
    return [abbreviate_name(str(col), style='smart') for col in columns]


def create_label_rotation_angle(
    columns: List[str],
    max_length: int = 10
) -> int:
    """
    Determine appropriate label rotation angle based on name lengths.
    
    Args:
        columns: List of column names
        max_length: Threshold for rotation
        
    Returns:
        Rotation angle in degrees (0, 45, or 90)
    """
    avg_length = sum(len(str(col)) for col in columns) / len(columns)
    
    if avg_length > max_length * 1.5:
        return 90  # Very long names
    elif avg_length > max_length:
        return 45  # Medium length
    else:
        return 0   # Short names


# ============================================================================
# VALIDATION & SAFETY
# ============================================================================

def is_name_too_long(name: str, threshold: int = 20) -> bool:
    """Check if name exceeds length threshold."""
    return len(str(name)) > threshold


def validate_names(columns: List[str]) -> Dict[str, str]:
    """
    Validate column names and suggest improvements.
    
    Returns dict with:
    - 'valid': list of acceptable names
    - 'too_long': list of names exceeding 20 chars
    - 'has_special': list of names with special characters
    """
    valid = []
    too_long = []
    has_special = []
    
    for col in columns:
        col_str = str(col)
        
        if len(col_str) > 20:
            too_long.append(col_str)
        elif re.search(r'[^a-zA-Z0-9_\-\s]', col_str):
            has_special.append(col_str)
        else:
            valid.append(col_str)
    
    return {
        'valid': valid,
        'too_long': too_long,
        'has_special': has_special
    }
