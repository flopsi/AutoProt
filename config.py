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
#Helper to detect all unique numeric columns:
# ============================================================
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

