"""
Data parsing utilities for proteomics files.
"""

import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional

from .config import get_config


class DataParser:
    """Parse proteomics data files"""
    
    def __init__(self):
        self.config = get_config()
    
    def load_dataframe(self, file_path: Path) -> pd.DataFrame:
        """Load data file into dataframe"""
        # Detect separator
        with open(file_path, 'r') as f:
            first_line = f.readline()
        
        if '\t' in first_line:
            sep = '\t'
        else:
            sep = ','
        
        # Load dataframe
        df = pd.read_csv(file_path, sep=sep, low_memory=False)
        
        return df
    
    def prepare_analysis_data(self, df: pd.DataFrame,
                             metadata_cols: List[str],
                             quantity_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataframe into metadata and quantification matrices
        
        Returns:
            (metadata_df, quantity_df)
        """
        metadata_df = df[metadata_cols].copy()
        quantity_df = df[quantity_cols].copy()
        
        return metadata_df, quantity_df


class MetadataParser:
    """Parse metadata files"""
    
    def load_metadata(self, file_path: Path) -> pd.DataFrame:
        """Load metadata CSV"""
        return pd.read_csv(file_path)
    
    def validate_metadata(self, metadata_df: pd.DataFrame,
                         sample_names: List[str]) -> bool:
        """Check if metadata matches sample names"""
        if 'sample_name' not in metadata_df.columns:
            return False
        
        metadata_samples = set(metadata_df['sample_name'])
        data_samples = set(sample_names)
        
        return metadata_samples == data_samples


def get_data_parser() -> DataParser:
    """Get data parser instance"""
    return DataParser()


def get_metadata_parser() -> MetadataParser:
    """Get metadata parser instance"""
    return MetadataParser()
