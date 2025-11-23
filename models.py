# models.py (simplified stub for upload pages)

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import pandas as pd

class DataLevel(Enum):
    PROTEIN = "protein"
    PEPTIDE = "peptide"

class Condition(Enum):
    CONTROL = "Control"
    TREATMENT = "Treatment"

@dataclass
class ColumnMetadata:
    original_name: str
    trimmed_name: str
    is_quantitative: bool
    is_protein_group: bool = False
    is_species_mapping: bool = False
    is_peptide_id: bool = False
    condition: Optional[Condition] = None

@dataclass
class DatasetConfig:
    level: DataLevel
    file_name: str
    total_rows: int
    columns: List[ColumnMetadata]
    protein_group_col: Optional[str] = None
    species_col: Optional[str] = None
    peptide_id_col: Optional[str] = None
    
    @property
    def quant_columns(self):
        return [col for col in self.columns if col.is_quantitative]
    @property
    def metadata_columns(self):
        return [col for col in self.columns if not col.is_quantitative]

@dataclass
class ProteomicsDataset:
    config: DatasetConfig
    data: pd.DataFrame
    @property
    def n_proteins(self): return len(self.data)
    @property
    def n_samples(self): return len(self.config.quant_columns)

class SessionKeys(Enum):
    PROTEIN_DATASET = "protein_dataset"
    PEPTIDE_DATASET = "peptide_dataset"
