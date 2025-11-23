from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd

@dataclass
class ProteomicsDataset:
    raw_df: pd.DataFrame
    metadata: pd.DataFrame
    quant_data: pd.DataFrame
    metadata_columns: List[str]
    quant_columns: List[str]
    species_column: Optional[str] = None
    species_map: Dict[int, str] = field(default_factory=dict)
    condition_mapping: Dict[str, str] = field(default_factory=dict)
    aggregation_column: Optional[str] = None
    protein_groups: Dict[int, str] = field(default_factory=dict)
    
    @property
    def n_rows(self) -> int:
        return len(self.raw_df)
    
    @property
    def n_proteins(self) -> int:
        if self.aggregation_column:
            return self.metadata[self.aggregation_column].nunique()
        return self.n_rows
    
    def get_species_counts(self) -> Dict[str, int]:
        if not self.species_map:
            return {}
        import pandas as pd
        species_series = pd.Series(self.species_map)
        return species_series.value_counts().to_dict()
    
    def get_condition_data(self, condition: str) -> pd.DataFrame:
        condition_cols = [col for col, cond in self.condition_mapping.items() 
                         if cond.startswith(condition)]
        return self.quant_data[condition_cols]
    
    def get_renamed_quant_data(self) -> pd.DataFrame:
        if not self.condition_mapping:
            return self.quant_data
        rename_map = {orig: new for orig, new in self.condition_mapping.items()}
        return self.quant_data.rename(columns=rename_map)