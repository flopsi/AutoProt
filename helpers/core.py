"""
helpers/core.py

Core configuration, themes, and data structures for AutoProt
Consolidates constants and dataclasses for centralized config management
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

# ============================================================================
# THEME DEFINITIONS
# Color schemes for UI consistency across light/dark/colorblind modes
# ============================================================================

THEME_LIGHT = {
    "name": "Light",
    "bg_primary": "#ffffff",
    "bg_secondary": "#f8f9fa",
    "text_primary": "#1a1a1a",
    "text_secondary": "#54585A",
    "color_human": "#199d76",      # Green for HUMAN proteins
    "color_yeast": "#d85f02",      # Orange for YEAST proteins
    "color_ecoli": "#7570b2",      # Purple for ECOLI proteins
    "color_up": "#d85f02",         # Orange for upregulated
    "color_down": "#199d76",       # Green for downregulated
    "color_ns": "#cccccc",         # Gray for not significant
    "color_nt": "#999999",         # Dark gray for not tested
    "grid": "#e0e0e0",
    "border": "#d0d0d0",
    "accent": "#199d76",
    "paper_bg": "rgba(0,0,0,0)",
}

THEME_DARK = {
    "name": "Dark",
    "bg_primary": "#1a1a1a",
    "bg_secondary": "#2d2d2d",
    "text_primary": "#ffffff",
    "text_secondary": "#cccccc",
    "color_human": "#4ccc9f",
    "color_yeast": "#ff9933",
    "color_ecoli": "#9e94d4",
    "color_up": "#ff9933",
    "color_down": "#4ccc9f",
    "color_ns": "#666666",
    "color_nt": "#444444",
    "grid": "#404040",
    "border": "#505050",
    "accent": "#4ccc9f",
    "paper_bg": "rgba(10,10,10,0.8)",
}

THEME_COLORBLIND = {
    "name": "Colorblind-Friendly",
    "bg_primary": "#ffffff",
    "bg_secondary": "#f8f9fa",
    "text_primary": "#1a1a1a",
    "text_secondary": "#54585A",
    "color_human": "#0173b2",      # Blue (deuteranopia-safe)
    "color_yeast": "#cc78bc",      # Magenta
    "color_ecoli": "#ca9161",      # Brown
    "color_up": "#cc78bc",
    "color_down": "#0173b2",
    "color_ns": "#cccccc",
    "color_nt": "#999999",
    "grid": "#e0e0e0",
    "border": "#d0d0d0",
    "accent": "#0173b2",
    "paper_bg": "rgba(0,0,0,0)",
}

THEME_JOURNAL = {
    "name": "Journal (B&W)",
    "bg_primary": "#ffffff",
    "bg_secondary": "#f5f5f5",
    "text_primary": "#000000",
    "text_secondary": "#333333",
    "color_human": "#404040",
    "color_yeast": "#000000",
    "color_ecoli": "#808080",
    "color_up": "#000000",
    "color_down": "#404040",
    "color_ns": "#c0c0c0",
    "color_nt": "#e0e0e0",
    "grid": "#d0d0d0",
    "border": "#a0a0a0",
    "accent": "#000000",
    "paper_bg": "rgba(0,0,0,0)",
}

THEMES: Dict[str, Dict] = {
    "light": THEME_LIGHT,
    "dark": THEME_DARK,
    "colorblind": THEME_COLORBLIND,
    "journal": THEME_JOURNAL,
}

# ============================================================================
# ANALYSIS CONSTANTS
# Default thresholds and parameters for statistical analysis
# ============================================================================

DEFAULT_FC_THRESHOLD = 1.0           # log2FC = 1.0 equals 2-fold change
DEFAULT_PVAL_THRESHOLD = 0.05        # P-value cutoff for significance
DEFAULT_MIN_VALID = 2                # Minimum values per group for t-test
SPECIES_LIST = ["HUMAN", "YEAST", "ECOLI", "MOUSE"]
TRANSFORMS = [
    "raw", "log2", "log10", "ln", "sqrt", "arcsinh",
    "boxcox", "yeo-johnson", "vst", "quantile"
]

# ============================================================================
# UI CONSTANTS
# Font configurations and plot dimensions
# ============================================================================

FONT_FAMILY = "Arial"
FONT_SIZE_BASE = 14
FONT_SIZE_TITLE = 16
FONT_SIZE_SMALL = 12
PLOT_HEIGHT_DEFAULT = 600
PLOT_HEIGHT_SMALL = 400
PLOT_WIDTH_DEFAULT = 900

# ============================================================================
# FILE I/O CONSTANTS
# Supported formats and size limits
# ============================================================================

SUPPORTED_FORMATS = ["csv", "tsv", "txt", "xlsx"]
MAX_FILE_SIZE_MB = 100

# ============================================================================
# HELPER FUNCTIONS
# Utility functions for theme and configuration access
# ============================================================================

def get_theme(theme_name: str = "light") -> Dict:
    """
    Get theme configuration by name.
    
    Args:
        theme_name: One of "light", "dark", "colorblind", "journal"
    
    Returns:
        Dictionary with color values and styling parameters
    """
    return THEMES.get(theme_name.lower(), THEME_LIGHT)

def get_theme_names() -> List[str]:
    """Get list of available theme names."""
    return list(THEMES.keys())

# ============================================================================
# DATA STRUCTURES
# Type-safe containers for protein analysis workflow
# ============================================================================

@dataclass
class ProteinData:
    """
    Main container for protein quantification data.
    Stores raw intensity matrix plus metadata for analysis pipeline.
    
    Attributes:
        raw: Original intensity data (proteins × samples)
        numeric_cols: List of quantitative column names
        species_col: Column name containing species annotations
        species_mapping: Dict mapping protein ID → species
        index_col: Protein/peptide identifier column name
        file_path: Source file path
        file_format: "csv", "tsv", or "excel"
    """
    raw: pd.DataFrame
    numeric_cols: List[str]
    species_col: Optional[str] = None
    species_mapping: Dict[str, str] = field(default_factory=dict)
    index_col: str = "Protein ID"
    file_path: str = ""
    file_format: str = "csv"
    
    @property
    def n_proteins(self) -> int:
        """Number of rows (proteins/peptides)."""
        return len(self.raw)
    
    @property
    def n_samples(self) -> int:
        """Number of samples (numeric columns)."""
        return len(self.numeric_cols)
    
    @property
    def n_conditions(self) -> int:
        """Number of unique experimental conditions."""
        if not self.numeric_cols:
            return 0
        conditions = set(col[0] for col in self.numeric_cols if col)
        return len([c for c in conditions if c.isalpha()])
    
    @property
    def missing_rate(self) -> float:
        """Percentage of missing values in numeric data."""
        total = self.raw[self.numeric_cols].size
        missing = self.raw[self.numeric_cols].isna().sum().sum()
        return (missing / total * 100) if total > 0 else 0.0

@dataclass
class TransformCache:
    """
    Cache for expensive data transformations.
    Stores computed transforms to avoid recomputation on page reruns.
    
    Attributes:
        log2: log2-transformed data
        log10: log10-transformed data
        sqrt: square root transformed data
        yeo_johnson: Yeo-Johnson normalized data
        quantile: Quantile-normalized data
        computed_at: ISO timestamp of computation
    """
    log2: Optional[pd.DataFrame] = None
    log10: Optional[pd.DataFrame] = None
    sqrt: Optional[pd.DataFrame] = None
    yeo_johnson: Optional[pd.DataFrame] = None
    quantile: Optional[pd.DataFrame] = None
    computed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def get(self, transform_key: str) -> Optional[pd.DataFrame]:
        """Get transform by key, returns None if not cached."""
        return getattr(self, transform_key, None)
    
    def has(self, transform_key: str) -> bool:
        """Check if transform exists in cache."""
        return getattr(self, transform_key, None) is not None

@dataclass
class AnalysisResults:
    """
    Differential expression analysis results container.
    
    Attributes:
        log2fc: log2 fold changes (group1 vs group2)
        pvalue: Welch's t-test p-values
        fdr: Benjamini-Hochberg FDR corrected p-values
        mean_group1: Mean intensity in reference group
        mean_group2: Mean intensity in treatment group
        n_group1: Valid values in reference group
        n_group2: Valid values in treatment group
        regulation: Classification (up/down/ns/not_tested)
        computed_at: ISO timestamp
    """
    log2fc: pd.Series
    pvalue: pd.Series
    fdr: pd.Series
    mean_group1: pd.Series
    mean_group2: pd.Series
    n_group1: pd.Series
    n_group2: pd.Series
    regulation: pd.Series = field(default_factory=pd.Series)
    computed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def n_up(self) -> int:
        """Count upregulated proteins."""
        return (self.regulation == "up").sum() if len(self.regulation) > 0 else 0
    
    @property
    def n_down(self) -> int:
        """Count downregulated proteins."""
        return (self.regulation == "down").sum() if len(self.regulation) > 0 else 0
    
    @property
    def n_significant(self) -> int:
        """Count significant proteins (up + down)."""
        return self.n_up + self.n_down

@dataclass
class TheoreticalComposition:
    """
    Theoretical species composition for validation.
    
    Attributes:
        condition_a: Dict of species → percentage for condition A
        condition_b: Dict of species → percentage for condition B
        theo_fc_species: Calculated theoretical log2FC by species
    """
    condition_a: Dict[str, float] = field(default_factory=dict)
    condition_b: Dict[str, float] = field(default_factory=dict)
    theo_fc_species: Dict[str, float] = field(default_factory=dict)
    
    @property
    def valid(self) -> bool:
        """Check if both conditions sum to ~100%."""
        sum_a = sum(self.condition_a.values())
        sum_b = sum(self.condition_b.values())
        return 99 < sum_a < 101 and 99 < sum_b < 101
    
    @property
    def sum_a(self) -> float:
        """Sum of condition A percentages."""
        return sum(self.condition_a.values())
    
    @property
    def sum_b(self) -> float:
        """Sum of condition B percentages."""
        return sum(self.condition_b.values())
