"""
helpers/core.py - Core Data Classes for Protein and Peptide Analysis

Production-grade data containers with comprehensive properties and validation.
"""

from dataclasses import dataclass
import pandas as pd
from typing import Optional
import numpy as np


@dataclass
class ProteinData:
    """
    Container for protein-level analysis data.
    
    Attributes:
        raw: DataFrame with proteins as rows, samples as columns
        numeric_cols: List of sample/abundance column names
        id_col: Name of the protein ID column
        species_col: Name of the species/organism column (optional)
        file_path: Original file path for reference
    """
    
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
        return round((missing / total * 100), 2) if total > 0 else 0.0

    @property
    def n_samples(self) -> int:
        """Number of samples/columns."""
        return len(self.numeric_cols)


@dataclass
class PeptideData:
    """
    Container for peptide-level analysis data.
    
    Attributes:
        raw: DataFrame with peptides as rows, samples as columns
        numeric_cols: List of sample/abundance column names
        id_col: Name of the peptide ID column
        species_col: Name of the species/organism column (optional)
        sequence_col: Name of the peptide sequence column
        file_path: Original file path for reference
    """
    
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
        return round((missing / total * 100), 2) if total > 0 else 0.0

    @property
    def n_samples(self) -> int:
        """Number of samples/columns."""
        return len(self.numeric_cols)

    @property
    def n_unique_sequences(self) -> int:
        """Number of unique peptide sequences."""
        if self.sequence_col not in self.raw.columns:
            return 0
        return self.raw[self.sequence_col].nunique()

    def get_sequence_length_stats(self) -> dict:
        """
        Get statistics about peptide sequence lengths.
        
        Returns:
            Dictionary with min, max, mean, and median sequence lengths
        """
        if self.sequence_col not in self.raw.columns:
            return {}
        
        lengths = self.raw[self.sequence_col].astype(str).str.len()
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
    """
    Get color palette for specified theme.
    
    Args:
        theme: Theme name ('light' or 'dark')
    
    Returns:
        Dictionary of color values for the theme
    """
    return THEME_COLORS.get(theme, THEME_COLORS['light'])
