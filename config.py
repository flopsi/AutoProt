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
# Helper to detect all unique numeric columns:
def get_numeric_columns(df):
    """Return list of columns with all- or mostly-numeric types."""
    numerics = []
    for col in df.columns:
        try:
            # If at least 90% can be converted to numbers, consider numeric.
            vals = pd.to_numeric(df[col], errors='coerce')
            pct_numeric = vals.notna().mean()
            if pct_numeric > 0.9:
                numerics.append(col)
        except Exception:
            continue
    return numerics

def get_metadata_columns(df, numeric_cols):
    """Return all non-numeric columns."""
    return [c for c in df.columns if c not in numeric_cols]

def get_default_species_mapping_cols(df):
    """Return all likely species-mapping metadata columns."""
    candidates = [
        "PG.ProteinNames", "First.Protein.Description", "Protein.Name", "Protein.Names",
        "Gene.Name", "Gene.Symbol", "Description"
    ]
    return [col for col in df.columns if any(x.lower() in col.lower() for x in candidates)]

def get_default_group_col(df):
    for col in df.columns:
        if "protein.group" in col.lower():
            return col
    return None

def get_default_peptide_id_col(df):
    for col in df.columns:
        for pat in ["precursor", "peptide"]:
            if pat in col.lower():
                return col
    return None
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
