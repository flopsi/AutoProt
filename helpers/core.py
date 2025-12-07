"""
helpers/core.py - UPDATED
Core data classes for protein and peptide analysis
"""

from dataclasses import dataclass
import pandas as pd
from typing import Optional, Dict
import numpy as np

# ============================================================================
# PROTEIN DATA CLASS
# ============================================================================

@dataclass
class ProteinData:
    """Container for protein-level analysis data."""
    raw: pd.DataFrame
    numeric_cols: list
    id_col: str
    species_col: Optional[str]
    file_path: str
    
    @property
    def n_proteins(self) -> int:
        """Total number of proteins."""
        return len(self.raw)
    
    @property
    def missing_rate(self) -> float:
        """Overall missing value rate (%)."""
        total = len(self.raw) * len(self.numeric_cols)
        missing = self.raw[self.numeric_cols].isna().sum().sum()
        return round((missing / total * 100), 2) if total > 0 else 0
    
    @property
    def n_samples(self) -> int:
        """Number of samples/columns."""
        return len(self.numeric_cols)

# ============================================================================
# PEPTIDE DATA CLASS (NEW)
# ============================================================================

@dataclass
class PeptideData:
    """Container for peptide-level analysis data."""
    raw: pd.DataFrame
    numeric_cols: list
    id_col: str
    species_col: Optional[str]
    sequence_col: str
    file_path: str
    
    @property
    def n_peptides(self) -> int:
        """Total number of peptides."""
        return len(self.raw)
    
    @property
    def missing_rate(self) -> float:
        """Overall missing value rate (%)."""
        total = len(self.raw) * len(self.numeric_cols)
        missing = self.raw[self.numeric_cols].isna().sum().sum()
        return round((missing / total * 100), 2) if total > 0 else 0
    
    @property
    def n_samples(self) -> int:
        """Number of samples/columns."""
        return len(self.numeric_cols)
    
    @property
    def n_unique_sequences(self) -> int:
        """Number of unique peptide sequences."""
        return self.raw[self.sequence_col].nunique()
    
    def get_sequence_length_stats(self) -> dict:
        """Get statistics about peptide sequence lengths."""
        if self.sequence_col not in self.raw.columns:
            return {}
        
        lengths = self.raw[self.sequence_col].str.len()
        return {
            'min_length': int(lengths.min()),
            'max_length': int(lengths.max()),
            'mean_length': round(float(lengths.mean()), 1),
            'median_length': int(lengths.median())
        }

# ============================================================================
# THEME MANAGEMENT
# ============================================================================

THEME_COLORS = {
    'light': {
        'primary': '#2980B9',
        'success': '#27AE60',
        'danger': '#E74C3C',
        'warning': '#F39C12',
        'info': '#3498DB'
    },
    'dark': {
        'primary': '#3498DB',
        'success': '#2ECC71',
        'danger': '#E74C3C',
        'warning': '#F1C40F',
        'info': '#85C1E2'
    }
}

def get_theme_colors(theme: str = 'light') -> dict:
    """Get color palette for theme."""
    return THEME_COLORS.get(theme, THEME_COLORS['light'])
