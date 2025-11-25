# utils/data_generator.py
import random
from typing import List, Dict
from types import SimpleNamespace

def generate_mock_protein(id_prefix: str, i: int, replicate_names: List[str]) -> dict:
    gene = f"GENE{i:03d}"
    # create some noise-rich replicate intensities
    intensities = {rep: max(0.1, random.gauss(10000, 2000)) for rep in replicate_names}
    return {
        "id": f"{id_prefix}_{i}",
        "gene": gene,
        "description": f"{gene} description",
        "replicates": intensities,
        "missing": False
    }

def generate_mock_proteins(n: int, replicate_names: List[str]) -> List[dict]:
    return [generate_mock_protein("P", i, replicate_names) for i in range(n)]
