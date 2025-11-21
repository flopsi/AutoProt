"""
File parsing functions for proteomics data.
Handles CSV/TSV reading, metadata parsing, and data loading.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
import streamlit as st

from .config import get_config
from .validators import get_validator


class ProteomicsDataParser:
    """Parse proteomics data files"""
    
    def __init__(self):
        self.config = get_config()
        self.validator = get_validator()
    
    def load_dataframe(self, file_path: Path, delimiter: Optional[str] = None) -> pd.DataFrame:
        """
        Load CSV/TSV file into dataframe with caching
        
        Args:
            file_path: Path to file
            delimiter: Optional delimiter (auto-detected if None)
            
        Returns:
            Loaded dataframe
        """
        if delimiter is None:
            delimiter = self.validator.detect_delimiter(file_path)
        
        # Use cached loading
        @st.cache_data(ttl=3600, show_spinner=False)
        def _load_cached(fpath: str, delim: str) -> pd.DataFrame:
            return pd.read_csv(
                fpath,
                sep=delim,
                low_memory=False,
                na_values=['', 'NA', 'NaN', 'N/A', 'null', 'NULL'],
                keep_default_na=True
            )
        
        return _load_cached(str(file_path), delimiter)
    
    def load_in_chunks(self, file_path: Path, chunk_size: int = 10000, 
                      delimiter: Optional[str] = None):
        """
        Generator to load large files in chunks
        
        Args:
            file_path: Path to file
            chunk_size: Number of rows per chunk
            delimiter: Optional delimiter
            
        Yields:
            Dataframe chunks
        """
        if delimiter is None:
            delimiter = self.validator.detect_delimiter(file_path)
        
        for chunk in pd.read_csv(
            file_path,
            sep=delimiter,
            chunksize=chunk_size,
            low_memory=False
        ):
            yield chunk
    
    def infer_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Infer semantic types of columns
        
        Returns:
            Dict mapping column name to type: 'metadata', 'quantification', 'quality', 'other'
        """
        column_types = {}
        
        for col in df.columns:
            # Check if it's a known metadata column
            if col in self.config.METADATA_COLUMNS:
                column_types[col] = 'metadata'
            
            # Check if it's a quality metric column
            elif any(qc in col for qc in ['Q.Value', 'Qvalue', 'PEP', 'Score']):
                column_types[col] = 'quality'
            
            # Check if numeric (likely quantification)
            elif pd.api.types.is_numeric_dtype(df[col]):
                column_types[col] = 'quantification'
            
            else:
                column_types[col] = 'other'
        
        return column_types
    
    def extract_numeric_data(self, df: pd.DataFrame, 
                           quantity_columns: list) -> pd.DataFrame:
        """
        Extract only numeric quantification data
        
        Args:
            df: Full dataframe
            quantity_columns: List of column names to extract
            
        Returns:
            Dataframe with only numeric data
        """
        # Ensure columns exist
        valid_cols = [col for col in quantity_columns if col in df.columns]
        
        # Extract and convert to numeric
        numeric_df = df[valid_cols].apply(pd.to_numeric, errors='coerce')
        
        return numeric_df
    
    def prepare_analysis_data(self, df: pd.DataFrame, 
                            metadata_cols: list,
                            quantity_cols: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into metadata and quantification matrices
        
        Args:
            df: Full dataframe
            metadata_cols: Metadata column names
            quantity_cols: Quantification column names
            
        Returns:
            Tuple of (metadata_df, quantity_df)
        """
        # Extract metadata
        valid_meta = [col for col in metadata_cols if col in df.columns]
        metadata_df = df[valid_meta].copy()
        
        # Extract quantification data
        quantity_df = self.extract_numeric_data(df, quantity_cols)
        
        # Ensure same index
        metadata_df.index = quantity_df.index
        
        return metadata_df, quantity_df


class MetadataParser:
    """Parse metadata files"""
    
    def __init__(self):
        self.config = get_config()
        self.validator = get_validator()
    
    def load_metadata(self, file_path: Path) -> pd.DataFrame:
        """
        Load metadata file
        
        Args:
            file_path: Path to metadata CSV
            
        Returns:
            Metadata dataframe
        """
        delimiter = self.validator.detect_delimiter(file_path)
        
        df_meta = pd.read_csv(file_path, sep=delimiter)
        
        return df_meta
    
    def match_samples_to_columns(self, metadata_df: pd.DataFrame,
                                column_names: list,
                                sample_col: str = 'sample_name') -> Dict[str, str]:
        """
        Match metadata sample names to data column names
        
        Args:
            metadata_df: Metadata dataframe
            column_names: List of column names from data
            sample_col: Name of sample column in metadata
            
        Returns:
            Dict mapping data column name to metadata sample name
        """
        if sample_col not in metadata_df.columns:
            raise ValueError(f"Sample column '{sample_col}' not found in metadata")
        
        metadata_samples = metadata_df[sample_col].tolist()
        matches = {}
        
        # Try exact matches first
        for col in column_names:
            if col in metadata_samples:
                matches[col] = col
        
        # Try fuzzy matching for remaining columns
        unmatched = [col for col in column_names if col not in matches]
        
        for col in unmatched:
            # Try partial matching
            col_lower = col.lower()
            for meta_sample in metadata_samples:
                meta_lower = meta_sample.lower()
                
                # Check if one contains the other
                if col_lower in meta_lower or meta_lower in col_lower:
                    matches[col] = meta_sample
                    break
        
        return matches
    
    def apply_metadata_to_columns(self, df: pd.DataFrame,
                                 metadata_df: pd.DataFrame,
                                 column_mapping: Dict[str, str],
                                 sample_col: str = 'sample_name') -> pd.DataFrame:
        """
        Apply metadata annotations to dataframe columns
        
        Args:
            df: Data dataframe
            metadata_df: Metadata dataframe
            column_mapping: Mapping of data columns to metadata samples
            sample_col: Sample column in metadata
            
        Returns:
            Dataframe with renamed columns or multiindex
        """
        # Create metadata lookup
        meta_lookup = metadata_df.set_index(sample_col).to_dict('index')
        
        # Build column multiindex
        new_columns = []
        for col in df.columns:
            if col in column_mapping:
                meta_sample = column_mapping[col]
                if meta_sample in meta_lookup:
                    meta_info = meta_lookup[meta_sample]
                    # Create tuple for multiindex
                    col_info = (col, meta_info.get('condition', 'NA'), 
                              meta_info.get('replicate', 'NA'))
                    new_columns.append(col_info)
                else:
                    new_columns.append((col, 'NA', 'NA'))
            else:
                new_columns.append((col, 'NA', 'NA'))
        
        # Apply multiindex
        df.columns = pd.MultiIndex.from_tuples(
            new_columns, 
            names=['sample', 'condition', 'replicate']
        )
        
        return df
    
    def create_design_matrix(self, metadata_df: pd.DataFrame,
                           column_mapping: Dict[str, str],
                           condition_col: str = 'condition',
                           replicate_col: str = 'replicate') -> pd.DataFrame:
        """
        Create experimental design matrix
        
        Args:
            metadata_df: Metadata dataframe
            column_mapping: Column to sample mapping
            condition_col: Condition column name
            replicate_col: Replicate column name
            
        Returns:
            Design matrix with sample assignments
        """
        design_data = []
        
        for data_col, meta_sample in column_mapping.items():
            meta_row = metadata_df[metadata_df['sample_name'] == meta_sample]
            
            if len(meta_row) > 0:
                design_data.append({
                    'sample': data_col,
                    'condition': meta_row[condition_col].values[0] if condition_col in meta_row else 'Unknown',
                    'replicate': meta_row[replicate_col].values[0] if replicate_col in meta_row else 'Unknown',
                    'species': meta_row['species'].values[0] if 'species' in meta_row else 'Unknown'
                })
        
        return pd.DataFrame(design_data)


def get_data_parser() -> ProteomicsDataParser:
    """Get data parser instance"""
    return ProteomicsDataParser()


def get_metadata_parser() -> MetadataParser:
    """Get metadata parser instance"""
    return MetadataParser()
