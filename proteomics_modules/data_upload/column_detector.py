"""
Column detection and sample name processing.
Auto-detects metadata and quantification columns, trims sample names.
Simplified species detection by keyword matching with column auto-detection.
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
    
    def find_species_column(self, df: pd.DataFrame, keyword_mapping: Dict[str, str]) -> Optional[str]:
        """
        Auto-detect which column contains species identifiers
        
        Args:
            df: Dataframe to search
            keyword_mapping: Dict of keywords to search for (e.g. {"HUMAN": "Human"})
            
        Returns:
            Column name with most keyword matches, or None
        """
        if not keyword_mapping:
            return None
        
        # Score each column by how many rows contain any keyword
        column_scores = {}
        
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype == 'string':
                # Count how many rows contain any keyword
                matches = 0
                for idx, value in df[col].items():
                    if pd.notna(value):
                        value_upper = str(value).upper()
                        for keyword in keyword_mapping.keys():
                            if keyword.upper() in value_upper:
                                matches += 1
                                break  # Count each row only once
                
                if matches > 0:
                    column_scores[col] = matches
        
        if not column_scores:
            return None
        
        # Return column with highest score
        best_column = max(column_scores.items(), key=lambda x: x[1])[0]
        return best_column
    
    def assign_species_by_keyword(self, protein_id: str, keyword_mapping: Dict[str, str]) -> str:
        """
        Assign species by simple keyword matching (case-insensitive)
        
        Example:
            protein_id = "GAL3B_HUMAN;GAL3A_HUMAN"
            keyword_mapping = {"HUMAN": "Human", "YEAST": "Yeast"}
            Returns: "Human"
        
        Args:
            protein_id: Protein ID/name string
            keyword_mapping: Dict mapping keywords (e.g. "HUMAN") to species names (e.g. "Human")
            
        Returns:
            Species name or "Unknown" if no match
        """
        if pd.isna(protein_id) or not keyword_mapping:
            return "Unknown"
        
        protein_str = str(protein_id).upper()
        
        # Check each keyword
        for keyword, species_name in keyword_mapping.items():
            if keyword.upper() in protein_str:
                return species_name
        
        return "Unknown"
    
    def get_species_distribution(self, species_series: pd.Series) -> Dict[str, int]:
        """Get count of proteins per species"""
        return species_series.value_counts().to_dict()


class SpeciesManager:
    """Manages species keyword mappings"""
    
    def __init__(self):
        self.keyword_mapping = {}  # keyword -> species name
        self.species_column = None  # Auto-detected or user-specified column
    
    def set_keyword_mapping(self, mapping: Dict[str, str]):
        """
        Set the user-provided keyword-to-species mapping
        
        Args:
            mapping: Dict like {"HUMAN": "Human", "YEAST": "Yeast", "ECOLI": "E. coli"}
        """
        self.keyword_mapping = mapping.copy()
    
    def get_keyword_mapping(self) -> Dict[str, str]:
        """Get the current keyword mapping"""
        return self.keyword_mapping.copy()
    
    def set_species_column(self, column_name: str):
        """Set the column name containing species identifiers"""
        self.species_column = column_name
    
    def get_species_column(self) -> Optional[str]:
        """Get the species column name"""
        return self.species_column
    
    def assign_species_with_keyword_mapping(self, df: pd.DataFrame,
                                          protein_col: Optional[str] = None) -> pd.Series:
        """
        Assign species to all proteins using keyword matching
        
        Args:
            df: Proteomics dataframe
            protein_col: Optional column name. If None, uses auto-detected column
            
        Returns:
            Series with species assignments
        """
        detector = ColumnDetector()
        
        # Use specified column or auto-detected column
        target_col = protein_col or self.species_column
        
        if not target_col or target_col not in df.columns:
            # Try to auto-detect
            target_col = detector.find_species_column(df, self.keyword_mapping)
            if target_col:
                self.species_column = target_col
        
        if not target_col:
            # Fallback to default column if available
            if 'Protein.Names' in df.columns:
                target_col = 'Protein.Names'
            elif 'Protein.Ids' in df.columns:
                target_col = 'Protein.Ids'
            else:
                # Use first text column
                for col in df.columns:
                    if df[col].dtype == 'object':
                        target_col = col
                        break
        
        if not target_col:
            return pd.Series(['Unknown'] * len(df), index=df.index)
        
        return df[target_col].apply(
            lambda x: detector.assign_species_by_keyword(x, self.keyword_mapping)
        )


def get_column_detector() -> ColumnDetector:
    """Get column detector instance"""
    return ColumnDetector()


def get_species_manager() -> SpeciesManager:
    """Get species manager instance"""
    return SpeciesManager()
