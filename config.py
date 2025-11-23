"""
Configuration module for DIA Proteomics App
Handles column detection, name trimming, and condition assignment
"""

import pandas as pd
import re

# ============================================================
# THERMO FISHER BRAND COLORS
# ============================================================
PRIMARY_RED = "#E71316"
DARK_RED = "#A6192E"
PRIMARY_GRAY = "#54585A"
LIGHT_GRAY = "#E2E3E4"
NAVY = "#262262"
ORANGE = "#EA7600"
YELLOW = "#F1B434"
GREEN = "#B5BD00"
SKY = "#9BD3DD"

# ============================================================
# DATA LEVEL DETECTION PATTERNS
# ============================================================
PEPTIDE_COLUMN_PATTERNS = [
    r'peptide',
    r'modified[._\s]sequence',
    r'stripped[._\s]sequence',
    r'peptide[._\s]sequence',
    r'^pep\.',
    r'precursor'
]

PROTEIN_COLUMN_PATTERNS = [
    r'protein[._\s]group',
    r'protein[._\s]id',
    r'majority[._\s]protein',
    r'protein[._\s]name',
    r'^pg\.',
    r'uniprot',
    r'gene[._\s]name'
]

# ============================================================
# COLUMN DETECTION
# ============================================================
def detect_column_types(df):
    """
    Automatically detect metadata vs quantitative columns.
    
    Rules:
    1. Non-numerical (object/string) → metadata
    2. Numerical → quantitative
    
    Returns:
        tuple: (metadata_cols, quant_cols)
    """
    metadata_cols = []
    quant_cols = []
    
    for col in df.columns:
        # Try to convert to numeric
        numeric_test = pd.to_numeric(df[col], errors='coerce')
        
        # If all values are NaN after conversion, it's metadata
        if numeric_test.isna().all():
            metadata_cols.append(col)
        else:
            quant_cols.append(col)
    
    return metadata_cols, quant_cols

def detect_data_level(df, metadata_cols):
    """
    Detect if data contains peptide-level or protein-level information.
    
    Returns:
        str: 'peptide', 'protein', 'both', or 'unknown'
    """
    has_peptide = False
    has_protein = False
    
    # Check all columns (case-insensitive)
    all_cols_lower = [col.lower() for col in df.columns]
    
    # Check for peptide patterns
    for pattern in PEPTIDE_COLUMN_PATTERNS:
        if any(re.search(pattern, col, re.IGNORECASE) for col in all_cols_lower):
            has_peptide = True
            break
    
    # Check for protein patterns
    for pattern in PROTEIN_COLUMN_PATTERNS:
        if any(re.search(pattern, col, re.IGNORECASE) for col in all_cols_lower):
            has_protein = True
            break
    
    if has_peptide and has_protein:
        return 'both'
    elif has_peptide:
        return 'peptide'
    elif has_protein:
        return 'protein'
    else:
        return 'unknown'

# ============================================================
# NAME TRIMMING
# ============================================================
def trim_column_names(columns):
    """
    Remove common prefixes/suffixes from column names.
    
    Common patterns removed:
    - LFQ.intensity.
    - Intensity.
    - Reporter.intensity.
    - Sample_
    - _Intensity
    - Leading/trailing whitespace
    
    Returns:
        dict: {original_name: trimmed_name}
    """
    trimmed = {}
    
    patterns_to_remove = [
        r'^LFQ\.intensity\.',
        r'^Intensity\.',
        r'^Reporter\.intensity\.',
        r'^Sample[_\.]',
        r'[_\.]Intensity$',
        r'^Abundance\.',
        r'\.Raw$',
        r'^PG\.',
    ]
    
    for col in columns:
        trimmed_name = col.strip()
        
        # Remove known patterns
        for pattern in patterns_to_remove:
            trimmed_name = re.sub(pattern, '', trimmed_name, flags=re.IGNORECASE)
        
        # Remove multiple dots/underscores
        trimmed_name = re.sub(r'[._]+', '_', trimmed_name)
        
        # Remove leading/trailing underscores
        trimmed_name = trimmed_name.strip('_')
        
        trimmed[col] = trimmed_name if trimmed_name else col
    
    return trimmed

# ============================================================
# CONDITION ASSIGNMENT
# ============================================================
def auto_assign_conditions(quant_cols):
    """
    Auto-assign conditions based on column count.
    
    Rules:
    - If even number: First half = Control, Second half = Treatment
    - Generates trimmed names automatically
    
    Returns:
        dict: {
            original_col: {
                'trimmed_name': str,
                'condition': 'Control' or 'Treatment',
                'index': int
            }
        }
    """
    assignments = {}
    n_cols = len(quant_cols)
    midpoint = n_cols // 2
    
    # Trim all names first
    trimmed_names = trim_column_names(quant_cols)
    
    for idx, col in enumerate(quant_cols):
        # Determine condition
        if n_cols % 2 == 0:  # Even number
            condition = 'Control' if idx < midpoint else 'Treatment'
        else:  # Odd number - assign based on position
            condition = 'Control' if idx <= midpoint else 'Treatment'
        
        assignments[col] = {
            'trimmed_name': trimmed_names[col],
            'condition': condition,
            'index': idx
        }
    
    return assignments

# ============================================================
# CONFIGURATION CONSTANTS
# ============================================================
MISSING_VALUES = [0, 1]  # Treat 0 and 1 as missing
CV_THRESHOLD_DEFAULT = 100  # Default CV threshold percentage
PCA_VARIANCE_THRESHOLD = 0.5  # Minimum variance for PCA
MIN_VALID_VALUES = 2  # Minimum non-missing values for stats

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_condition_colors():
    """Return color mapping for conditions"""
    return {
        'Control': SKY,
        'Treatment': DARK_RED
    }

def get_css_variables():
    """Return CSS variables string for Thermo Fisher styling"""
    return f"""
    :root {{
        --primary-red: {PRIMARY_RED};
        --dark-red: {DARK_RED};
        --primary-gray: {PRIMARY_GRAY};
        --light-gray: {LIGHT_GRAY};
        --navy: {NAVY};
        --orange: {ORANGE};
        --green: {GREEN};
        --sky: {SKY};
    }}
    """
