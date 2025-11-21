"""
Configuration settings for data upload module.
"""

from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path


@dataclass
class DataUploadConfig:
    """Configuration for data upload module"""
    
    # File validation
    ALLOWED_EXTENSIONS: List[str] = None
    MAX_FILE_SIZE_MB: int = 500
    
    # Column detection
    METADATA_COLUMNS: List[str] = None
    QUANTITY_SUFFIXES: List[str] = None
    
    # Name trimming patterns
    TRIM_PATTERNS: Dict[str, str] = None
    EXTRACT_PATTERN: str = r'([A-Z]\d{2}[-_][A-Z]\d{2}_\d{2})'
    
    # Session management
    UPLOAD_DIR: Path = Path("data/uploads")
    SESSION_TIMEOUT_HOURS: int = 24
    
    def __post_init__(self):
        """Initialize default values"""
        if self.ALLOWED_EXTENSIONS is None:
            self.ALLOWED_EXTENSIONS = ['.csv', '.tsv', '.txt']
        
        if self.METADATA_COLUMNS is None:
            self.METADATA_COLUMNS = [
                'Protein.Group',
                'Protein.Ids',
                'Protein.Names',
                'Genes',
                'First.Protein.Description',
                'Proteotypic',
                'Stripped.Sequence',
                'Modified.Sequence',
                'Precursor.Id',
                'Q.Value',
                'PEP',
                'Global.Q.Value',
                'Protein.Q.Value',
                'PG.Q.Value',
                'Global.PG.Q.Value',
                'GG.Q.Value',
                'Translated.Q.Value',
                'Lib.Q.Value',
                'Lib.PG.Q.Value',
                'Ms1.Translated',
                'Genes.Quantity',
                'Genes.Normalized',
                'Genes.MaxLFQ',
                'Genes.MaxLFQ.Unique'
            ]
        
        if self.QUANTITY_SUFFIXES is None:
            self.QUANTITY_SUFFIXES = [
                '.PG.Quantity',
                '.PG.Normalized',
                '.PG.MaxLFQ',
                'Intensity',
                'LFQ',
                '.raw'
            ]
        
        if self.TRIM_PATTERNS is None:
            self.TRIM_PATTERNS = {
                'date_prefix': r'^\d{8}_',
                'technical_suffix': r'\.(raw|PG\.Quantity|PG\.Normalized|PG\.MaxLFQ|Intensity|LFQ)$',
                'method_prefix': r'^(MP\d+_|DIA_|DDA_)',
                'instrument_codes': r'(IO\d+_|SPD_|LFQ_)',
                'concentration': r'\d+pg_'
            }


# Global config instance
_config = None


def get_config() -> DataUploadConfig:
    """Get global config instance"""
    global _config
    if _config is None:
        _config = DataUploadConfig()
    return _config


def set_config(config: DataUploadConfig):
    """Set global config instance"""
    global _config
    _config = config
