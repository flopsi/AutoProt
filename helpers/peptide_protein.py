"""
helpers/peptide_protein.py
Peptide-to-protein aggregation and protein inference
Converts peptide-level data to protein-level data
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple

# ============================================================================
# PEPTIDE TO PROTEIN AGGREGATION
# ============================================================================

def aggregate_peptides_by_id(
    peptide_data: pd.DataFrame,
    peptide_id_col: str,
    protein_id_col: str,
    numeric_cols: list,
    method: str = "sum",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Aggregate peptide-level intensities to protein level.
    
    Current Phase 1: Simple aggregation by protein ID.
    Expandable for: weighted average, max, median, etc.
    
    Args:
        peptide_data: DataFrame with peptide intensities
        peptide_id_col: Column name for peptide ID
        protein_id_col: Column name for protein ID (grouping key)
        numeric_cols: List of intensity columns to aggregate
        method: "sum", "mean", "median", "max" (default "sum")
        
    Returns:
        (aggregated_df, metadata_dict)
    """
    
    # ---- VALIDATION ----
    if protein_id_col not in peptide_data.columns:
        raise ValueError(f"Column '{protein_id_col}' not found in data.")
    
    if not all(col in peptide_data.columns for col in numeric_cols):
        missing = [col for col in numeric_cols if col not in peptide_data.columns]
        raise ValueError(f"Columns not found: {missing}")
    
    # ---- AGGREGATION ----
    if method == "sum":
        agg_func = "sum"
    elif method == "mean":
        agg_func = "mean"
    elif method == "median":
        agg_func = "median"
    elif method == "max":
        agg_func = "max"
    else:
        agg_func = "sum"  # Default to sum
    
    # Group by protein ID and aggregate
    protein_data = peptide_data.groupby(
        protein_id_col,
        as_index=False,
    )[numeric_cols].agg(agg_func)
    
    # Rename protein column to standard name
    protein_data = protein_data.rename(
        columns={protein_id_col: "Protein ID"}
    )
    protein_data.set_index("Protein ID", inplace=True)
    
    # ---- METADATA ----
    # Count peptides per protein
    peptides_per_protein = peptide_data.groupby(protein_id_col).size()
    
    n_peptides_original = len(peptide_data)
    n_proteins = len(protein_data)
    avg_peptides_per_protein = n_peptides_original / n_proteins if n_proteins > 0 else 0
    
    metadata = {
        "n_peptides_original": int(n_peptides_original),
        "n_proteins": int(n_proteins),
        "avg_peptides_per_protein": float(avg_peptides_per_protein),
        "aggregation_method": method,
        "min_peptides_per_protein": int(peptides_per_protein.min()),
        "max_peptides_per_protein": int(peptides_per_protein.max()),
    }
    
    return protein_data, metadata


# ============================================================================
# PROTEIN INFERENCE (PARSIMONIOUS, RAZOR, BAYESIAN)
# ============================================================================

def infer_proteins_parsimonious(
    peptide_protein_mapping: Dict[str, list],
    peptide_intensities: pd.Series,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Parsimonious protein inference: fewest proteins that explain all peptides.
    Greedy algorithm: iteratively select protein explaining most unexplained peptides.
    
    Args:
        peptide_protein_mapping: Dict mapping peptide ID → list of protein IDs
        peptide_intensities: Intensities for each peptide
        
    Returns:
        (inferred_proteins, metadata)
    """
    
    # ---- INITIALIZATION ----
    unexplained_peptides = set(peptide_protein_mapping.keys())
    inferred_proteins = []
    protein_peptide_count = {}
    
    # ---- GREEDY SELECTION ----
    while unexplained_peptides:
        # For each protein, count how many unexplained peptides it covers
        protein_coverage = {}
        
        for peptide in unexplained_peptides:
            for protein in peptide_protein_mapping[peptide]:
                protein_coverage[protein] = protein_coverage.get(protein, 0) + 1
        
        if not protein_coverage:
            break  # No more proteins can explain remaining peptides
        
        # Select protein covering most peptides
        best_protein = max(protein_coverage, key=protein_coverage.get)
        inferred_proteins.append(best_protein)
        protein_peptide_count[best_protein] = protein_coverage[best_protein]
        
        # Remove explained peptides
        for peptide, proteins in peptide_protein_mapping.items():
            if best_protein in proteins:
                unexplained_peptides.discard(peptide)
    
    # ---- RESULTS ----
    results = pd.DataFrame({
        "Protein ID": inferred_proteins,
        "Peptide Count": [protein_peptide_count[p] for p in inferred_proteins],
    })
    
    metadata = {
        "n_proteins_inferred": len(inferred_proteins),
        "n_peptides_total": len(peptide_protein_mapping),
        "n_unexplained": len(unexplained_peptides),
        "method": "parsimonious",
    }
    
    return results, metadata


def infer_proteins_razor(
    peptide_protein_mapping: Dict[str, list],
    peptide_intensities: pd.Series,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Razor protein inference: assign each peptide to single most abundant protein.
    
    Args:
        peptide_protein_mapping: Dict mapping peptide ID → list of protein IDs
        peptide_intensities: Intensities for each peptide
        
    Returns:
        (inferred_proteins, metadata)
    """
    
    protein_intensity = {}
    protein_peptide_list = {}
    
    # ---- ASSIGN PEPTIDES TO PROTEINS ----
    for peptide, proteins in peptide_protein_mapping.items():
        if not proteins:
            continue
        
        intensity = peptide_intensities.get(peptide, 0)
        
        # Assign to first protein (could be most abundant in multi-protein group)
        assigned_protein = proteins[0]
        
        protein_intensity[assigned_protein] = protein_intensity.get(assigned_protein, 0) + intensity
        
        if assigned_protein not in protein_peptide_list:
            protein_peptide_list[assigned_protein] = []
        protein_peptide_list[assigned_protein].append(peptide)
    
    # ---- RESULTS ----
    results = pd.DataFrame({
        "Protein ID": list(protein_intensity.keys()),
        "Intensity": list(protein_intensity.values()),
        "Peptide Count": [len(protein_peptide_list[p]) for p in protein_intensity.keys()],
    })
    
    results = results.sort_values("Intensity", ascending=False).reset_index(drop=True)
    
    metadata = {
        "n_proteins_inferred": len(protein_intensity),
        "n_peptides_total": len(peptide_protein_mapping),
        "method": "razor",
    }
    
    return results, metadata


# ============================================================================
# UTILITY: CHECK IF DATA IS PEPTIDE OR PROTEIN LEVEL
# ============================================================================

def detect_data_level(
    df: pd.DataFrame,
    row_id_col: str,
) -> Tuple[str, str]:
    """
    Heuristically detect if data is peptide or protein level.
    
    Peptide indicators:
    - Column names contain 'peptide', 'pep', 'seq'
    - Row IDs contain peptide-like patterns (e.g., K.PEPTIDESEQ.R)
    
    Args:
        df: Input DataFrame
        row_id_col: Column name for row identifiers
        
    Returns:
        ("peptide" or "protein", reasoning_string)
    """
    
    reasoning_parts = []
    peptide_score = 0
    protein_score = 0
    
    # Check column names
    col_names_lower = [col.lower() for col in df.columns]
    if any("peptide" in col or "pep" in col for col in col_names_lower):
        peptide_score += 2
        reasoning_parts.append("Column names contain 'peptide'")
    
    if any("protein" in col or "uniprot" in col for col in col_names_lower):
        protein_score += 2
        reasoning_parts.append("Column names contain 'protein'")
    
    # Check row ID format
    if row_id_col in df.columns:
        sample_ids = df[row_id_col].dropna().head(5).astype(str)
        
        # Peptide format: typically "R.XXXXXXX.K" or just sequence
        peptide_patterns = [
            ".",  # Flanking AA notation
            "PEPT",  # Common peptide prefixes
            "seq",
        ]
        
        for pattern in peptide_patterns:
            matching = (sample_ids.str.contains(pattern, case=False, na=False)).sum()
            if matching > 0:
                peptide_score += 1
                reasoning_parts.append(f"Row IDs contain '{pattern}'")
                break
    
    # Decide
    if peptide_score > protein_score:
        level = "peptide"
    elif protein_score > peptide_score:
        level = "protein"
    else:
        level = "protein"  # Default to protein
        reasoning_parts.append("(Default to protein)")
    
    reasoning = ", ".join(reasoning_parts) if reasoning_parts else "Unknown"
    
    return level, reasoning
