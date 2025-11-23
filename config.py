"""
Configuration module for DIA Proteomics App
UPDATED: Enhanced trimming with duplicate handling
"""
import pandas as pd
import re

# ========================================================================
# THERMO FISHER COLOR SCHEME
# ========================================================================
THERMO_COLORS = {
    'PRIMARY_RED': '#E71316',
    'PRIMARY_GRAY': '#54585A',
    'LIGHT_GRAY': '#E2E3E4',
    'NAVY': '#262262',
    'DARK_RED': '#A6192E',
    'ORANGE': '#EA7600',
    'YELLOW': '#F1B434',
    'GREEN': '#B5BD00',
    'SKY': '#9BD3DD'
}

# ========================================================================
# COLUMN DETECTION
# ========================================================================
def get_numeric_columns(df: pd.DataFrame) -> list:
    """
    Detect all numeric/quantitative columns.
    Returns columns where at least 90% of values can be converted to numbers.
    """
    numeric_cols = []
    for col in df.columns:
        numeric_vals = pd.to_numeric(df[col], errors='coerce')
        non_null_count = numeric_vals.notna().sum()
        if non_null_count / len(df) >= 0.9:
            numeric_cols.append(col)
    return numeric_cols

def get_metadata_columns(df: pd.DataFrame, numeric_cols: list) -> list:
    """
    Get all non-numeric (metadata) columns.
    These are columns NOT in the numeric_cols list.
    """
    return [col for col in df.columns if col not in numeric_cols]

# ========================================================================
# NAME TRIMMING WITH DUPLICATE HANDLING
# ========================================================================
def trim_column_names(cols: list) -> dict:
    """
    Remove common prefixes/suffixes from column names.
    Handles duplicates by adding numeric suffixes.
    Returns dict: {original_name: trimmed_name}
    """
    patterns_to_remove = [
        r'^LFQ\.intensity\.',
        r'^Intensity\.',
        r'^iBAQ\.',
        r'\.raw$',
        r'\.d$',
        r'^C:\\\\.*\\\\',  # Windows paths
        r'^/.*/',          # Unix paths
    ]
    trimmed = {}
    trimmed_counts = {}
    for col in cols:
        cleaned = col
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        if not cleaned:
            cleaned = col
        base_cleaned = cleaned
        if base_cleaned in trimmed_counts:
            trimmed_counts[base_cleaned] += 1
            cleaned = f"{base_cleaned}_{trimmed_counts[base_cleaned]}"
        else:
            trimmed_counts[base_cleaned] = 0
        trimmed[col] = cleaned
    return trimmed

# ========================================================================
# AUTO-ASSIGNMENT LOGIC
# ========================================================================
def auto_assign_conditions(quant_cols: list) -> dict:
    """
    Auto-assign Control/Treatment based on even split.
    First half = Control, Second half = Treatment
    Returns: {column_name: 'Control' or 'Treatment'}
    """
    n_cols = len(quant_cols)
    midpoint = n_cols // 2
    assignments = {}
    for idx, col in enumerate(quant_cols):
        if idx < midpoint:
            assignments[col] = 'Control'
        else:
            assignments[col] = 'Treatment'
    return assignments

# ========================================================================
# SPECIES MAPPING COLUMN DETECTION
# ========================================================================
def get_default_species_mapping_cols(df: pd.DataFrame) -> list:
    """
    Find columns likely to contain species information.
    Returns list of column names sorted by priority.
    """
    priority_keywords = [
        'fasta', 'protein', 'gene', 'accession', 'id', 'name'
    ]
    candidates = []
    metadata_cols = [col for col in df.columns if df[col].dtype == object]
    for keyword in priority_keywords:
        for col in metadata_cols:
            if keyword.lower() in col.lower() and col not in candidates:
                candidates.append(col)
    return candidates if candidates else metadata_cols

# ========================================================================
# PROTEIN GROUP COLUMN DETECTION
# ========================================================================
def get_default_group_col(df: pd.DataFrame, metadata_cols: list) -> list:
    """
    Find columns likely to contain protein group IDs.
    Returns list sorted by priority.
    """
    priority_keywords = [
        'protein.group', 'protein group', 'proteingroup', 'protein ids', 'protein.ids'
    ]
    candidates = []
    for keyword in priority_keywords:
        for col in metadata_cols:
            if keyword.lower() in col.lower() and col not in candidates:
                candidates.append(col)
    return candidates if candidates else metadata_cols

# ========================================================================
# PEPTIDE ID COLUMN DETECTION
# ========================================================================
def get_default_peptide_id_col(df: pd.DataFrame, metadata_cols: list) -> list:
    """
    Find columns likely to contain peptide/precursor IDs.
    Returns list sorted by priority.
    """
    priority_keywords = [
        'precursor', 'peptide', 'modified sequence', 'modified.sequence', 'eg.precursorid'
    ]
    candidates = []
    for keyword in priority_keywords:
        for col in metadata_cols:
            if keyword.lower() in col.lower() and col not in candidates:
                candidates.append(col)
    return candidates if candidates else metadata_cols
