"""
Validation functions for proteomics data files.
Checks file format, structure, and data quality.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

from .config import get_config


@dataclass
class ValidationResult:
    """Result of validation check"""
    is_valid: bool
    message: str
    severity: str = 'info'  # 'info', 'warning', 'error'
    details: Optional[Dict] = None


class DataValidator:
    """Validates uploaded proteomics data files"""
    
    def __init__(self):
        self.config = get_config()
    
    def validate_file_extension(self, filename: str) -> ValidationResult:
        """Check if file has allowed extension"""
        file_path = Path(filename)
        ext = file_path.suffix.lower()
        
        if ext in self.config.ALLOWED_EXTENSIONS:
            return ValidationResult(
                is_valid=True,
                message=f"File extension '{ext}' is supported",
                severity='info'
            )
        else:
            return ValidationResult(
                is_valid=False,
                message=f"File extension '{ext}' not supported. Allowed: {', '.join(self.config.ALLOWED_EXTENSIONS)}",
                severity='error'
            )
    
    def validate_file_size(self, file_size: int) -> ValidationResult:
        """Check if file size is within limits"""
        size_mb = file_size / (1024 * 1024)
        max_size = self.config.MAX_FILE_SIZE_MB
        
        if size_mb <= max_size:
            return ValidationResult(
                is_valid=True,
                message=f"File size: {size_mb:.2f} MB",
                severity='info',
                details={'size_mb': size_mb}
            )
        else:
            return ValidationResult(
                is_valid=False,
                message=f"File too large: {size_mb:.2f} MB (max: {max_size} MB)",
                severity='error',
                details={'size_mb': size_mb, 'max_size_mb': max_size}
            )
    
    def detect_delimiter(self, file_path: Path, sample_lines: int = 5) -> str:
        """Auto-detect file delimiter"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            sample = ''.join([f.readline() for _ in range(sample_lines)])
        
        # Count potential delimiters
        delimiters = {
            ',': sample.count(','),
            '\t': sample.count('\t'),
            ';': sample.count(';'),
            '|': sample.count('|')
        }
        
        # Return most common
        delimiter = max(delimiters, key=delimiters.get)
        return delimiter if delimiter != '\t' else '\t'
    
    def validate_csv_structure(self, file_path: Path) -> ValidationResult:
        """Validate CSV file can be read and has proper structure"""
        try:
            # Try to detect delimiter
            delimiter = self.detect_delimiter(file_path)
            
            # Read first few rows to validate
            df = pd.read_csv(file_path, sep=delimiter, nrows=10)
            
            n_rows, n_cols = df.shape
            
            if n_cols < 2:
                return ValidationResult(
                    is_valid=False,
                    message="File must have at least 2 columns",
                    severity='error'
                )
            
            return ValidationResult(
                is_valid=True,
                message=f"CSV structure valid: {n_cols} columns detected",
                severity='info',
                details={
                    'delimiter': delimiter,
                    'n_columns': n_cols,
                    'column_names': list(df.columns)
                }
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Failed to read CSV: {str(e)}",
                severity='error',
                details={'error': str(e)}
            )
    
    def validate_proteomics_data(self, df: pd.DataFrame) -> List[ValidationResult]:
        """
        Comprehensive validation of proteomics data
        
        Args:
            df: Loaded dataframe
            
        Returns:
            List of validation results
        """
        results = []
        
        # Check minimum size
        n_rows, n_cols = df.shape
        
        if n_rows < self.config.MIN_PROTEINS:
            results.append(ValidationResult(
                is_valid=False,
                message=f"Too few proteins: {n_rows} (minimum: {self.config.MIN_PROTEINS})",
                severity='warning',
                details={'n_proteins': n_rows, 'min_required': self.config.MIN_PROTEINS}
            ))
        else:
            results.append(ValidationResult(
                is_valid=True,
                message=f"Protein count: {n_rows}",
                severity='info',
                details={'n_proteins': n_rows}
            ))
        
        # Check for metadata columns
        metadata_found = []
        for col in self.config.METADATA_COLUMNS:
            if col in df.columns:
                metadata_found.append(col)
        
        if len(metadata_found) > 0:
            results.append(ValidationResult(
                is_valid=True,
                message=f"Found metadata columns: {', '.join(metadata_found)}",
                severity='info',
                details={'metadata_columns': metadata_found}
            ))
        else:
            results.append(ValidationResult(
                is_valid=False,
                message="No standard metadata columns detected (e.g., Protein.Group, Protein.Names)",
                severity='warning'
            ))
        
        # Check for quantification columns (numeric data)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < self.config.MIN_SAMPLES:
            results.append(ValidationResult(
                is_valid=False,
                message=f"Too few numeric columns: {len(numeric_cols)} (minimum: {self.config.MIN_SAMPLES})",
                severity='error',
                details={'n_samples': len(numeric_cols), 'min_required': self.config.MIN_SAMPLES}
            ))
        else:
            results.append(ValidationResult(
                is_valid=True,
                message=f"Quantification columns: {len(numeric_cols)}",
                severity='info',
                details={'n_samples': len(numeric_cols)}
            ))
        
        # Check missing values
        if len(numeric_cols) > 0:
            numeric_data = df[numeric_cols]
            total_values = numeric_data.size
            missing_values = numeric_data.isna().sum().sum()
            missing_percent = (missing_values / total_values) * 100
            
            if missing_percent > self.config.MAX_MISSING_PERCENT:
                results.append(ValidationResult(
                    is_valid=False,
                    message=f"Too many missing values: {missing_percent:.1f}%",
                    severity='error',
                    details={'missing_percent': missing_percent}
                ))
            elif missing_percent > self.config.WARN_MISSING_PERCENT:
                results.append(ValidationResult(
                    is_valid=True,
                    message=f"High missing values: {missing_percent:.1f}%",
                    severity='warning',
                    details={'missing_percent': missing_percent}
                ))
            else:
                results.append(ValidationResult(
                    is_valid=True,
                    message=f"Missing values: {missing_percent:.1f}%",
                    severity='info',
                    details={'missing_percent': missing_percent}
                ))
        
        return results
    
    def validate_metadata_file(self, df_metadata: pd.DataFrame) -> List[ValidationResult]:
        """
        Validate metadata file structure
        
        Args:
            df_metadata: Metadata dataframe
            
        Returns:
            List of validation results
        """
        results = []
        
        # Check required columns
        for req_col in self.config.METADATA_REQUIRED_COLUMNS:
            if req_col in df_metadata.columns:
                results.append(ValidationResult(
                    is_valid=True,
                    message=f"Required column '{req_col}' found",
                    severity='info'
                ))
            else:
                results.append(ValidationResult(
                    is_valid=False,
                    message=f"Required column '{req_col}' missing",
                    severity='error'
                ))
        
        # Check optional columns
        optional_found = [col for col in self.config.METADATA_OPTIONAL_COLUMNS 
                         if col in df_metadata.columns]
        
        if optional_found:
            results.append(ValidationResult(
                is_valid=True,
                message=f"Optional columns found: {', '.join(optional_found)}",
                severity='info',
                details={'optional_columns': optional_found}
            ))
        
        # Check for duplicates in sample_name
        if 'sample_name' in df_metadata.columns:
            duplicates = df_metadata['sample_name'].duplicated().sum()
            if duplicates > 0:
                results.append(ValidationResult(
                    is_valid=False,
                    message=f"Duplicate sample names found: {duplicates}",
                    severity='error',
                    details={'n_duplicates': duplicates}
                ))
        
        return results


def get_validator() -> DataValidator:
    """Get validator instance"""
    return DataValidator()
