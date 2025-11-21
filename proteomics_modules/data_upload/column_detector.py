"""
Column detection and sample name processing.
Auto-detects metadata and quantification columns, trims sample names.
"""

import re
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .config import get_config


@dataclass
class ColumnClassification:
    """Result of column classification"""
    metadata_columns: List[str]
    quantity_columns: List[str]
    other_columns: List[str]
    quantity_column_mapping: Dict[str, str]  # original -> trimmed


@dataclass
class SpeciesDetectionResult:
    """Result of species detection for a protein"""
    species: str
    confidence: str  # 'high', 'medium', 'low'
    matched_pattern: str
    column_source: str  # which column was used


class ColumnDetector:
    """Detects and classifies columns in proteomics data"""
    
    def __init__(self):
        self.config = get_config()
    
    def classify_columns(self, columns: List[str]) -> ColumnClassification:
        """
        Classify columns into metadata and quantification columns
        
        Args:
            columns: List of column names from dataframe
            
        Returns:
            ColumnClassification with categorized columns
        """
        metadata_cols = []
        quantity_cols = []
        other_cols = []
        
        # First pass: identify metadata columns
        for col in columns:
            if col in self.config.METADATA_COLUMNS:
                metadata_cols.append(col)
            else:
                # Check if it looks like a quantification column
                if self._is_quantity_column(col):
                    quantity_cols.append(col)
                else:
                    other_cols.append(col)
        
        # Generate trimmed names
        quantity_mapping = {}
        for col in quantity_cols:
            trimmed = self.trim_column_name(col)
            quantity_mapping[col] = trimmed
        
        return ColumnClassification(
            metadata_columns=metadata_cols,
            quantity_columns=quantity_cols,
            other_columns=other_cols,
            quantity_column_mapping=quantity_mapping
        )
    
    def _is_quantity_column(self, col_name: str) -> bool:
        """Check if column name looks like a quantification column"""
        # Check for quantity suffixes
        for suffix in self.config.QUANTITY_SUFFIXES:
            if col_name.endswith(suffix):
                return True
        
        # Check if column name contains typical patterns
        patterns = [
            r'\d{8}_',  # Date prefix
            r'\.raw',    # Raw file
            r'_\d{2}$',  # Replicate number at end
            r'[A-Z]\d{2}[-_][A-Z]\d{2}'  # Sample codes like Y05-E45
        ]
        
        for pattern in patterns:
            if re.search(pattern, col_name):
                return True
        
        return False
    
    def trim_column_name(self, col_name: str) -> str:
        """
        Trim column name to extract meaningful part
        
        Example:
            Input: "20240419_MP1_50SPD_IO25_LFQ_250pg_Y05-E45_01.raw.PG.Quantity"
            Output: "Y05-E45_01"
        """
        trimmed = col_name
        
        # Apply trim patterns in order
        for pattern_name, pattern in self.config.TRIM_PATTERNS.items():
            trimmed = re.sub(pattern, '', trimmed)
        
        # Try to extract meaningful part using extract pattern
        extract_match = re.search(self.config.EXTRACT_PATTERN, col_name)
        if extract_match:
            extracted = extract_match.group(1)
            # Only use if shorter than current trimmed version
            if len(extracted) < len(trimmed):
                trimmed = extracted
        
        # Fallback: if still too long or empty, use last meaningful parts
        if len(trimmed) > 50 or len(trimmed) == 0:
            # Split by common separators and take last 2-3 parts
            parts = re.split(r'[_\-/\\.]', col_name)
            # Filter out empty and common noise
            parts = [p for p in parts if p and p not in 
                    ['raw', 'PG', 'Quantity', 'Intensity', 'LFQ']]
            
            if len(parts) >= 2:
                trimmed = '_'.join(parts[-2:])
            elif len(parts) == 1:
                trimmed = parts[0]
            else:
                trimmed = col_name[:30]  # Truncate if all else fails
        
        return trimmed
    
    def suggest_replicate_groups(self, trimmed_names: List[str]) -> Dict[str, List[str]]:
        """
        Suggest grouping of replicates based on trimmed names
        
        Example:
            Input: ["Y05-E45_01", "Y05-E45_02", "Y45-E05_01"]
            Output: {
                "Y05-E45": ["Y05-E45_01", "Y05-E45_02"],
                "Y45-E05": ["Y45-E05_01"]
            }
        """
        groups = {}
        
        for name in trimmed_names:
            # Try to extract base name (remove replicate number)
            base_patterns = [
                (r'(.+)[_-]\d{1,2}$', 'numeric_suffix'),  # Remove _01, -02, etc.
                (r'(.+)_rep\d+$', 'rep_suffix'),           # Remove _rep1, _rep2
                (r'(.+)[_-][ABC]$', 'letter_suffix')       # Remove _A, -B, etc.
            ]
            
            base_name = name
            for pattern, _ in base_patterns:
                match = re.match(pattern, name)
                if match:
                    base_name = match.group(1)
                    break
            
            if base_name not in groups:
                groups[base_name] = []
            groups[base_name].append(name)
        
        return groups
    
    def detect_species(self, df: pd.DataFrame, 
                      protein_name_col: str = 'Protein.Names') -> pd.Series:
        """
        Auto-detect species for each protein
        
        Args:
            df: Proteomics dataframe
            protein_name_col: Column containing protein names
            
        Returns:
            Series with species assignments
        """
        species_list = []
        
        # Priority order for checking columns
        check_columns = ['Protein.Names', 'Protein.Ids', 'First.Protein.Description', 'Protein.Group']
        check_columns = [col for col in check_columns if col in df.columns]
        
        if not check_columns:
            # No suitable column found
            return pd.Series(['Unknown'] * len(df), index=df.index)
        
        for idx, row in df.iterrows():
            detected_species = 'Unknown'
            
            # Check each column in priority order
            for col in check_columns:
                if pd.isna(row[col]):
                    continue
                
                text = str(row[col])
                
                # Try to match species patterns
                for species_name, pattern in self.config.SPECIES_PATTERNS.items():
                    if re.search(pattern, text, re.IGNORECASE):
                        detected_species = species_name
                        break
                
                if detected_species != 'Unknown':
                    break
            
            species_list.append(detected_species)
        
        return pd.Series(species_list, index=df.index)
    
    def get_species_distribution(self, species_series: pd.Series) -> Dict[str, int]:
        """Get count of proteins per species"""
        return species_series.value_counts().to_dict()


class SpeciesManager:
    """Manages custom species patterns"""
    
    def __init__(self):
        self.config = get_config()
        self.custom_patterns = {}
    
    def add_custom_species(self, species_name: str, pattern: str):
        """Add or update custom species pattern"""
        self.custom_patterns[species_name] = pattern
    
    def remove_custom_species(self, species_name: str):
        """Remove custom species pattern"""
        if species_name in self.custom_patterns:
            del self.custom_patterns[species_name]
    
    def get_all_patterns(self) -> Dict[str, str]:
        """Get all species patterns (default + custom)"""
        all_patterns = self.config.SPECIES_PATTERNS.copy()
        all_patterns.update(self.custom_patterns)
        return all_patterns
    
    def detect_with_custom_patterns(self, df: pd.DataFrame) -> pd.Series:
        """Detect species using both default and custom patterns"""
        detector = ColumnDetector()
        
        # Temporarily update config with custom patterns
        original_patterns = detector.config.SPECIES_PATTERNS.copy()
        detector.config.SPECIES_PATTERNS.update(self.custom_patterns)
        
        # Detect
        result = detector.detect_species(df)
        
        # Restore original
        detector.config.SPECIES_PATTERNS = original_patterns
        
        return result


def get_column_detector() -> ColumnDetector:
    """Get column detector instance"""
    return ColumnDetector()


def get_species_manager() -> SpeciesManager:
    """Get species manager instance"""
    return SpeciesManager()
