"""
helpers/dataclasses.py
Core data structures for AutoProt with caching support
Type-safe, immutable data containers for protein/peptide analysis
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

# ============================================================================
# PROTEIN/PEPTIDE DATA STRUCTURES
# ============================================================================

@dataclass
class ProteinData:
    """
    Main container for protein quantification data.
    
    Attributes:
        raw: Original intensity data (proteins × samples)
        numeric_cols: List of quantitative column names
        species_col: Column name containing species info
        species_mapping: Dict mapping protein ID to species
        index_col: Protein/peptide ID column name
        file_path: Source file path
        file_format: "csv", "tsv", "excel"
        n_proteins: Number of proteins/peptides
        conditions: Dict mapping sample → condition letter
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
        """Number of unique conditions."""
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
    Cache for expensive transformations.
    Store computed transforms to avoid recomputation on reruns.
    
    Attributes:
        log2: log2-transformed data
        log10: log10-transformed data
        sqrt: square root transformed data
        cbrt: cube root transformed data
        yeo_johnson: Yeo-Johnson normalized data
        quantile: Quantile-normalized data
        computed_at: Timestamp of computation
    """
    log2: Optional[pd.DataFrame] = None
    log10: Optional[pd.DataFrame] = None
    sqrt: Optional[pd.DataFrame] = None
    cbrt: Optional[pd.DataFrame] = None
    yeo_johnson: Optional[pd.DataFrame] = None
    quantile: Optional[pd.DataFrame] = None
    computed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def get(self, transform_key: str) -> Optional[pd.DataFrame]:
        """Get transform by key, returns None if not computed."""
        return getattr(self, transform_key, None)
    
    def has(self, transform_key: str) -> bool:
        """Check if transform is cached."""
        return getattr(self, transform_key, None) is not None


@dataclass
class AnalysisResults:
    """
    Differential expression analysis results.
    
    Attributes:
        log2fc: log2 fold changes
        pvalue: t-test p-values
        fdr: Benjamini-Hochberg FDR corrected p-values
        mean_group1: Mean intensity in reference group
        mean_group2: Mean intensity in treatment group
        n_group1: Valid values in reference group
        n_group2: Valid values in treatment group
        regulation: Classification (up/down/ns/not_tested)
        computed_at: Timestamp
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
        """Count significant proteins (up or down)."""
        return self.n_up + self.n_down


@dataclass
class TheoreticalComposition:
    """
    Theoretical species composition for fold change calculation.
    
    Attributes:
        condition_a: Dict of species percentages for condition A
        condition_b: Dict of species percentages for condition B
        theo_fc_species: Calculated log2FC by species
        valid: Whether percentages sum to 100
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


# ============================================================================
# AUDIT TRAIL & LOGGING
# ============================================================================

@dataclass
class AuditEvent:
    """
    Single data manipulation event.
    
    Attributes:
        timestamp: ISO format datetime
        page: Page name where event occurred
        action: Description of action (e.g., "Filtered 100 proteins")
        details: Additional info (protein count, threshold, etc.)
        user_id: Optional user identifier
    """
    timestamp: str
    page: str
    action: str
    details: Dict = field(default_factory=dict)
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            "timestamp": self.timestamp,
            "page": self.page,
            "action": self.action,
            "details": self.details,
            "user_id": self.user_id,
        }


@dataclass
class AuditTrail:
    """
    Complete audit log for a session.
    
    Attributes:
        events: List of audit events
        session_start: Session start time
    """
    events: List[AuditEvent] = field(default_factory=list)
    session_start: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_event(self, page: str, action: str, details: Dict = None) -> None:
        """Add an event to the trail."""
        event = AuditEvent(
            timestamp=datetime.now().isoformat(),
            page=page,
            action=action,
            details=details or {},
        )
        self.events.append(event)
    
    def __len__(self) -> int:
        """Number of events in trail."""
        return len(self.events)


# ============================================================================
# FILTERED DATA CONTAINER
# ============================================================================

@dataclass
class FilteredDataset:
    """
    Dataset after filtering and quality control.
    
    Attributes:
        data: Filtered protein data
        n_proteins_original: Original count
        n_proteins_filtered: After filtering
        filters_applied: List of filter descriptions
        transform_key: Applied transformation
    """
    data: pd.DataFrame
    n_proteins_original: int
    n_proteins_filtered: int
    filters_applied: List[str] = field(default_factory=list)
    transform_key: str = "log2"
    
    @property
    def filter_rate(self) -> float:
        """Percentage of proteins retained."""
        if self.n_proteins_original == 0:
            return 0.0
        return (self.n_proteins_filtered / self.n_proteins_original) * 100
