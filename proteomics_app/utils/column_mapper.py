from typing import Dict
import pandas as pd

def apply_column_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    return df.rename(columns=mapping)

def validate_even_replicates(mapping: Dict[str, str]) -> bool:
    return len(mapping) % 2 == 0