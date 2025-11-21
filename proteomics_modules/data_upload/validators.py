"""
Validation functions for uploaded data.
"""

import pandas as pd
from pathlib import Path
from typing import List
from dataclasses import dataclass

from .config import get_config


@dataclass
class ValidationResult:
    """Result of a validation check"""
    is_valid: bool
    message: str
    severity: str = 'error'  # error, warning, info


class DataValidator:
    """Validate uploaded proteomics data"""
    
    def __init__(self):
        self.config = get_config()
    
    def validate_file_extension(self, filename: str) -> ValidationResult:
        """Check if file extension is allowed"""
        ext = Path(filename).suffix.lower()
        
        if ext in self.config.ALLOWED_EXTENSIONS:
            return ValidationResult(
                is_valid=True,
                message=f"File extension {ext} is supported",
                severity='info'
            )
        else:
            return ValidationResult(
                is_valid=False,
                message=f"File extension {ext} not allowed. Supported: {', '.join(self.config.ALLOWED_EXTENSIONS)}",
                severity='error'
            )
    
    def validate_file_size(self, size_bytes: int) -> ValidationResult:
        """Check if file size is within limit"""
        size_mb = size_bytes / (1024 * 1024)
        
        if size_mb <= self.config.MAX_FILE_SIZE_MB:
            return ValidationResult(
                is_valid=True,
                message=f"File size: {size_mb:.1f} MB",
                severity='info'
            )
        else:
            return ValidationResult(
                is_valid=False,
                message=f"File too large: {size_mb:.1f} MB (max: {self.config.MAX_FILE_SIZE_MB} MB)",
                severity='error'
            )
    
    def validate_csv_structure(self, file_path: Path) -> ValidationResult:
        """Validate basic CSV structure"""
        try:
            # Try reading first few rows
            df = pd.read_csv(file_path, nrows=5)
            
            if len(df.columns) < 2:
                return ValidationResult(
                    is_valid=False,
                    message="File must have at least 2 columns",
                    severity='error'
                )
            
            return ValidationResult(
                is_valid=True,
                message=f"Valid CSV with {len(df.columns)} columns",
                severity='info'
            )
        
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Cannot read file: {str(e)}",
                severity='error'
            )
    
    def validate_proteomics_data(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate proteomics data content"""
        results = []
        
        # Check row count
        if len(df) < 10:
            results.append(ValidationResult(
                is_valid=False,
                message=f"Too few rows: {len(df)} (minimum: 10)",
                severity='warning'
            ))
        else:
            results.append(ValidationResult(
                is_valid=True,
                message=f"{len(df):,} proteins/peptides detected",
                severity='info'
            ))
        
        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) == 0:
            results.append(ValidationResult(
                is_valid=False,
                message="No numeric columns found",
                severity='error'
            ))
        else:
            results.append(ValidationResult(
                is_valid=True,
                message=f"{len(numeric_cols)} numeric columns detected",
                severity='info'
            ))
        
        # Check missing values
        missing_pct = (df.isna().sum().sum() / df.size) * 100
        if missing_pct > 80:
            results.append(ValidationResult(
                is_valid=False,
                message=f"Too many missing values: {missing_pct:.1f}%",
                severity='warning'
            ))
        else:
            results.append(ValidationResult(
                is_valid=True,
                message=f"Missing values: {missing_pct:.1f}%",
                severity='info'
            ))
        
        return results


def get_validator() -> DataValidator:
    """Get validator instance"""
    return DataValidator()
