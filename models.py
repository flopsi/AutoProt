"""
Data models for DIA Proteomics App
Uses dataclasses and enums for type safety and clarity
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import pandas as pd


# ============================================================================
# ENUMS
# ============================================================================

class DataLevel(Enum):
    """Type of proteomics data"""
    PROTEIN = "protein"
    PEPTIDE = "peptide"


class Condition(Enum):
    """Sample condition type"""
    CONTROL = "Control"
    TREATMENT = "Treatment"


class StatisticalTest(Enum):
    """Available statistical tests"""
    TTEST = "t-test"
    MANN_WHITNEY = "Mann-Whitney U"
    ANOVA = "ANOVA"
    KRUSKAL_WALLIS = "Kruskal-Wallis"
    
    @property
    def description(self) -> str:
        """Get test description"""
        descriptions = {
            self.TTEST: "Parametric test for two groups (assumes normal distribution)",
            self.MANN_WHITNEY: "Non-parametric test for two groups",
            self.ANOVA: "Parametric test for multiple groups (assumes normal distribution)",
            self.KRUSKAL_WALLIS: "Non-parametric test for multiple groups"
        }
        return descriptions.get(self, "")
    
    @property
    def requires_normality(self) -> bool:
        """Check if test requires normal distribution"""
        return self in [self.TTEST, self.ANOVA]
    
    @property
    def min_groups(self) -> int:
        """Minimum number of groups required"""
        return 3 if self in [self.ANOVA, self.KRUSKAL_WALLIS] else 2


class NormalizationMethod(Enum):
    """Normalization methods"""
    NONE = "None"
    LOG2 = "Log2"
    MEDIAN = "Median"
    QUANTILE = "Quantile"
    ZSCORE = "Z-Score"


class ImputationMethod(Enum):
    """Missing value imputation methods"""
    NONE = "None"
    ZERO = "Zero"
    MIN = "Minimum"
    MEAN = "Mean"
    MEDIAN = "Median"
    KNN = "K-Nearest Neighbors"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ColumnMetadata:
    """Metadata about a column in the dataset"""
    original_name: str
    trimmed_name: str
    is_quantitative: bool
    is_protein_group: bool = False
    is_species_mapping: bool = False
    is_peptide_id: bool = False
    condition: Optional[Condition] = None
    
    def __post_init__(self):
        """Validate after initialization"""
        if self.condition and not self.is_quantitative:
            raise ValueError("Only quantitative columns can have conditions")


@dataclass
class DatasetConfig:
    """Configuration for a proteomics dataset"""
    level: DataLevel
    file_name: str
    total_rows: int
    columns: List[ColumnMetadata]
    protein_group_col: Optional[str] = None
    species_col: Optional[str] = None
    peptide_id_col: Optional[str] = None
    
    @property
    def quant_columns(self) -> List[ColumnMetadata]:
        """Get all quantitative columns"""
        return [col for col in self.columns if col.is_quantitative]
    
    @property
    def metadata_columns(self) -> List[ColumnMetadata]:
        """Get all metadata columns"""
        return [col for col in self.columns if not col.is_quantitative]
    
    @property
    def control_columns(self) -> List[str]:
        """Get control sample column names"""
        return [col.trimmed_name for col in self.quant_columns 
                if col.condition == Condition.CONTROL]
    
    @property
    def treatment_columns(self) -> List[str]:
        """Get treatment sample column names"""
        return [col.trimmed_name for col in self.quant_columns 
                if col.condition == Condition.TREATMENT]
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings"""
        warnings = []
        
        if not self.protein_group_col:
            warnings.append("No protein group column selected")
        
        if not self.species_col:
            warnings.append("No species mapping column selected")
        
        if self.level == DataLevel.PEPTIDE and not self.peptide_id_col:
            warnings.append("No peptide ID column selected")
        
        if not self.control_columns and not self.treatment_columns:
            warnings.append("No control or treatment samples selected")
        
        return warnings


@dataclass
class ProteomicsDataset:
    """Complete proteomics dataset with data and configuration"""
    config: DatasetConfig
    data: pd.DataFrame
    
    def __post_init__(self):
        """Validate data matches configuration"""
        # Ensure all configured columns exist in data
        expected_cols = [col.trimmed_name for col in self.config.columns]
        missing = set(expected_cols) - set(self.data.columns)
        if missing:
            raise ValueError(f"Columns missing from data: {missing}")
    
    @property
    def n_proteins(self) -> int:
        """Number of proteins/peptides"""
        return len(self.data)
    
    @property
    def n_samples(self) -> int:
        """Number of samples (quant columns)"""
        return len(self.config.quant_columns)
    
    def get_control_data(self) -> pd.DataFrame:
        """Get only control samples"""
        return self.data[self.config.control_columns]
    
    def get_treatment_data(self) -> pd.DataFrame:
        """Get only treatment samples"""
        return self.data[self.config.treatment_columns]


@dataclass
class AnalysisParams:
    """Parameters for statistical analysis"""
    statistical_test: StatisticalTest
    normalization: NormalizationMethod = NormalizationMethod.NONE
    imputation: ImputationMethod = ImputationMethod.NONE
    alpha: float = 0.05
    fold_change_threshold: float = 2.0
    min_valid_values: int = 2
    
    def __post_init__(self):
        """Validate parameters"""
        if not 0 < self.alpha < 1:
            raise ValueError("Alpha must be between 0 and 1")
        if self.fold_change_threshold <= 0:
            raise ValueError("Fold change threshold must be positive")
        if self.min_valid_values < 1:
            raise ValueError("Min valid values must be at least 1")


# ============================================================================
# SESSION STATE KEYS (for type safety)
# ============================================================================

class SessionKeys(Enum):
    """Keys for st.session_state"""
    PROTEIN_DATASET = "protein_dataset"
    PEPTIDE_DATASET = "peptide_dataset"
    PROTEIN_CONFIG = "protein_config"
    PEPTIDE_CONFIG = "peptide_config"
    ANALYSIS_PARAMS = "analysis_params"
    RESULTS = "results"
