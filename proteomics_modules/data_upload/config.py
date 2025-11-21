"""
Configuration module for proteomics data upload.
Contains all constants, patterns, and default settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List
import re


@dataclass
class DataUploadConfig:
    """Configuration for data upload module"""
    
    # ==================== File Handling ====================
    MAX_FILE_SIZE_MB: int = 500
    ALLOWED_EXTENSIONS: List[str] = field(default_factory=lambda: ['.csv', '.tsv', '.txt'])
    TEMP_DATA_DIR: str = 'data/sessions'
    
    # ==================== Column Detection ====================
    # Standard metadata columns (order matters - first match wins)
    METADATA_COLUMNS: List[str] = field(default_factory=lambda: [
        'Protein.Group',
        'Protein.Ids', 
        'Protein.Names',
        'Genes',
        'First.Protein.Description',
        'Q.Value',
        'PG.Qvalue',
        'Global.Q.Value'
    ])
    
    # Patterns to identify quantification columns
    QUANTITY_SUFFIXES: List[str] = field(default_factory=lambda: [
        '.raw.PG.Quantity',
        '.PG.Quantity',
        '_Quantity',
        '.Intensity',
        '_Intensity',
        '.raw',
        '_LFQ'
    ])
    
    # ==================== Species Detection ====================
    SPECIES_PATTERNS: Dict[str, str] = field(default_factory=lambda: {
        'Human': r'(HUMAN|Homo sapiens|9606|HSA)',
        'Yeast': r'(YEAST|Saccharomyces|559292|SCE)',
        'E.coli': r'(ECOLI|Escherichia coli|562|ECO)',
        'C.elegans': r'(CAEEL|Caenorhabditis elegans|6239|CEL)'
    })
    
    # ==================== Sample Name Trimming ====================
    # Patterns to remove from column names (applied in order)
    TRIM_PATTERNS: Dict[str, str] = field(default_factory=lambda: {
        'date_prefix': r'^\d{8}_',              # Remove 20240419_
        'datetime_prefix': r'^\d{6,8}_\d+_',   # Remove 20240419_123456_
        'instrument_codes': r'^[A-Z]{2,4}\d+_', # Remove MP1_, QE01_
        'path_prefix': r'^.*[\\/]',            # Remove path separators
        'raw_suffix': r'\.raw.*$',              # Remove .raw.PG.Quantity
        'file_extension': r'\.(raw|mzML|d|wiff).*$',
        'pg_quantity': r'\.PG\.Quantity$',
        'quantity_suffix': r'[._](Quantity|Intensity|LFQ)$'
    })
    
    # Pattern to extract meaningful part (e.g., Y05-E45_01 from long name)
    EXTRACT_PATTERN: str = r'([A-Z]\d{2}[-_][A-Z]\d{2}[_-]\d{2}|[A-Z][a-z]+[_-]\w+[_-]\d+)'
    
    # ==================== Validation Thresholds ====================
    MIN_PROTEINS: int = 100
    MIN_SAMPLES: int = 2
    MIN_VALID_VALUES_PER_PROTEIN: int = 1
    MAX_MISSING_PERCENT: float = 95.0
    WARN_MISSING_PERCENT: float = 70.0
    
    # ==================== Metadata File ====================
    METADATA_REQUIRED_COLUMNS: List[str] = field(default_factory=lambda: ['sample_name'])
    METADATA_OPTIONAL_COLUMNS: List[str] = field(default_factory=lambda: [
        'condition', 'replicate', 'species', 'batch', 'injection_order'
    ])
    
    # ==================== Session Management ====================
    SESSION_TIMEOUT_HOURS: int = 24
    AUTO_CLEANUP: bool = True


# Singleton instance
config = DataUploadConfig()


def get_config() -> DataUploadConfig:
    """Get the global configuration instance"""
    return config


def update_config(**kwargs):
    """Update configuration parameters"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
