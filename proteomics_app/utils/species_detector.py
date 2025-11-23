from typing import Optional, Dict
import pandas as pd
from config.species import Species

def auto_detect_species_column(df: pd.DataFrame) -> Optional[str]:
    species_keywords = ['_HUMAN', '_ECOLI', '_YEAST']
    for col in df.select_dtypes(include='object').columns:
        sample_values = df[col].head(5).astype(str).str.upper()
        for keyword in species_keywords:
            if any(keyword in val for val in sample_values):
                return col
    return None

def extract_species_map(df: pd.DataFrame, species_column: str) -> Dict[int, str]:
    species_map = {}
    for idx, row in df.iterrows():
        protein_string = str(row[species_column])
        if ';' in protein_string:
            first_protein = protein_string.split(';')[0]
        else:
            first_protein = protein_string
        species = Species.from_protein_name(first_protein)
        if species:
            species_map[idx] = species.value
        else:
            species_map[idx] = 'unknown'
    return species_map

def extract_protein_groups(df: pd.DataFrame, protein_column: str) -> Dict[int, str]:
    return {idx: str(row[protein_column]) for idx, row in df.iterrows()}