
"""
helpers/analysis.py

Analysis utilities for protein/peptide data processing
Handles group comparisons and protein-peptide relationships
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import streamlit as st

# ============================================================================
# GROUP COMPARISON
# Utilities for defining and validating experimental groups
# ============================================================================

def detect_conditions_from_columns(numeric_cols: List[str]) -> List[str]:
    """
    Auto-detect experimental conditions from column names.
    Assumes format: "A1", "A2", "B1", "B2" where letter = condition.
    
    Args:
        numeric_cols: List of sample column names
    
    Returns:
        List of unique condition letters (e.g., ["A", "B", "C"])
    """
    conditions = set()
    for col in numeric_cols:
        if len(col) > 0 and col[0].isalpha():
            conditions.add(col[0])
    return sorted(list(conditions))

def group_columns_by_condition(
    numeric_cols: List[str],
    condition_letter: str
) -> List[str]:
    """
    Get all columns belonging to a specific condition.
    
    Args:
        numeric_cols: All sample column names
        condition_letter: Condition identifier (e.g., "A")
    
    Returns:
        List of columns starting with condition_letter
    """
    return [col for col in numeric_cols if col.startswith(condition_letter)]

def create_group_dict(
    numeric_cols: List[str],
    conditions: List[str] = None
) -> Dict[str, List[str]]:
    """
    Create dictionary mapping condition names to column lists.
    
    Args:
        numeric_cols: All sample column names
        conditions: Optional list of conditions to include
    
    Returns:
        Dict like {"A": ["A1", "A2"], "B": ["B1", "B2"]}
    """
    if conditions is None:
        conditions = detect_conditions_from_columns(numeric_cols)
    
    return {
        cond: group_columns_by_condition(numeric_cols, cond)
        for cond in conditions
    }

def validate_group_comparison(
    group1_cols: List[str],
    group2_cols: List[str],
    min_samples: int = 2
) -> Tuple[bool, str]:
    """
    Validate that groups have sufficient samples for comparison.
    
    Args:
        group1_cols: Reference group columns
        group2_cols: Treatment group columns
        min_samples: Minimum required samples per group
    
    Returns:
        (is_valid, message) tuple
    """
    if len(group1_cols) < min_samples:
        return False, f"Group 1 has only {len(group1_cols)} samples. Need at least {min_samples}."
    
    if len(group2_cols) < min_samples:
        return False, f"Group 2 has only {len(group2_cols)} samples. Need at least {min_samples}."
    
    return True, f"✅ Valid comparison: {len(group1_cols)} vs {len(group2_cols)} samples"

# ============================================================================
# PROTEIN-PEPTIDE RELATIONSHIPS
# Mapping and aggregation functions
# ============================================================================

def map_peptides_to_proteins(
    df: pd.DataFrame,
    peptide_col: str = "Peptide",
    protein_col: str = "Protein"
) -> Dict[str, List[str]]:
    """
    Create mapping from proteins to their constituent peptides.
    
    Args:
        df: DataFrame with peptide and protein columns
        peptide_col: Column name containing peptide IDs
        protein_col: Column name containing protein IDs
    
    Returns:
        Dict mapping protein ID → list of peptide IDs
    """
    protein_to_peptides = {}
    
    for _, row in df.iterrows():
        protein = row[protein_col]
        peptide = row[peptide_col]
        
        if pd.notna(protein) and pd.notna(peptide):
            if protein not in protein_to_peptides:
                protein_to_peptides[protein] = []
            protein_to_peptides[protein].append(peptide)
    
    return protein_to_peptides

def aggregate_peptides_to_proteins(
    df_peptides: pd.DataFrame,
    peptide_col: str,
    protein_col: str,
    intensity_cols: List[str],
    method: str = "sum"
) -> pd.DataFrame:
    """
    Aggregate peptide-level intensities to protein-level.
    
    Args:
        df_peptides: Peptide-level data
        peptide_col: Column name for peptide IDs
        protein_col: Column name for protein IDs
        intensity_cols: Columns to aggregate
        method: Aggregation method ("sum", "mean", "median", "max")
    
    Returns:
        DataFrame with protein-level aggregated intensities
    """
    # Group by protein
    grouped = df_peptides.groupby(protein_col)
    
    # Apply aggregation
    if method == "sum":
        df_proteins = grouped[intensity_cols].sum()
    elif method == "mean":
        df_proteins = grouped[intensity_cols].mean()
    elif method == "median":
        df_proteins = grouped[intensity_cols].median()
    elif method == "max":
        df_proteins = grouped[intensity_cols].max()
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    df_proteins = df_proteins.reset_index()
    return df_proteins

def calculate_peptide_counts(
    df: pd.DataFrame,
    protein_col: str,
    peptide_col: str
) -> pd.Series:
    """
    Count number of peptides per protein.
    
    Args:
        df: DataFrame with protein and peptide columns
        protein_col: Column name for protein IDs
        peptide_col: Column name for peptide IDs
    
    Returns:
        Series with protein ID as index, peptide count as values
    """
    return df.groupby(protein_col)[peptide_col].count()

# ============================================================================
# SPECIES FILTERING
# Filter proteins by species annotation
# ============================================================================

def filter_by_species(
    df: pd.DataFrame,
    species_mapping: Dict[str, str],
    selected_species: List[str]
) -> pd.DataFrame:
    """
    Filter DataFrame to only include proteins from selected species.
    
    Args:
        df: Input DataFrame with protein IDs as index
        species_mapping: Dict mapping protein ID → species
        selected_species: List of species to keep (e.g., ["HUMAN", "YEAST"])
    
    Returns:
        Filtered DataFrame
    """
    # Get protein IDs for selected species
    keep_ids = [
        pid for pid, sp in species_mapping.items()
        if sp in selected_species
    ]
    
    # Filter
    return df.loc[df.index.intersection(keep_ids)].copy()

def count_proteins_by_species(
    df: pd.DataFrame,
    species_mapping: Dict[str, str]
) -> pd.Series:
    """
    Count proteins per species in DataFrame.
    
    Args:
        df: DataFrame with protein IDs as index
        species_mapping: Dict mapping protein ID → species
    
    Returns:
        Series with species names as index, counts as values
    """
    species_list = [species_mapping.get(pid, "UNKNOWN") for pid in df.index]
    return pd.Series(species_list).value_counts()

# ============================================================================
# THEORETICAL FOLD CHANGE CALCULATION
# For spike-in validation experiments
# ============================================================================

def calculate_theoretical_fc(
    composition_a: Dict[str, float],
    composition_b: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate theoretical log2 fold changes from species composition.
    Used for validating spike-in experiments.
    
    Args:
        composition_a: Dict of species → percentage in condition A
        composition_b: Dict of species → percentage in condition B
    
    Returns:
        Dict of species → theoretical log2FC
    
    Example:
        comp_a = {"HUMAN": 65, "YEAST": 30, "ECOLI": 5}
        comp_b = {"HUMAN": 65, "YEAST": 15, "ECOLI": 20}
        theo_fc = calculate_theoretical_fc(comp_a, comp_b)
        # {"HUMAN": 0.0, "YEAST": 1.0, "ECOLI": -2.0}
    """
    theo_fc = {}
    
    for species in set(list(composition_a.keys()) + list(composition_b.keys())):
        pct_a = composition_a.get(species, 0.0)
        pct_b = composition_b.get(species, 0.0)
        
        if pct_a > 0 and pct_b > 0:
            # log2FC = log2(A/B)
            theo_fc[species] = np.log2(pct_a / pct_b)
        elif pct_a > 0:
            theo_fc[species] = np.inf  # Only in A
        elif pct_b > 0:
            theo_fc[species] = -np.inf  # Only in B
        else:
            theo_fc[species] = 0.0
    
    return theo_fc

# ============================================================================
# DATA QUALITY METRICS (for Visual EDA)
# Per-species protein counts and missing value analysis
# ============================================================================

def count_valid_proteins_per_species_sample(
    df: pd.DataFrame,
    numeric_cols: List[str],
    species_mapping: Dict[str, str],
    missing_value: float = 1.0
) -> pd.DataFrame:
    """
    Count valid (non-missing) proteins per species per sample.
    
    A protein is "valid" in a sample if intensity ≠ missing_value.
    
    Args:
        df: DataFrame with protein IDs as index
        numeric_cols: List of sample column names
        species_mapping: Dict mapping protein ID → species name
        missing_value: Value indicating missing data (default 1.0)
        
    Returns:
        DataFrame with shape (species, samples) showing count of valid proteins
        
    Example:
        >>> df has 100 HUMAN proteins
        >>> In A1: 90 have intensity ≠ 1.0, 10 have intensity = 1.0
        >>> Result: HUMAN | A1: 90 | A2: 92 | ...
    """
    # Initialize result dict
    result_dict = {}
    
    # Get unique species from mapping
    unique_species = sorted(set(species_mapping.values()))
    
    # For each sample, count valid proteins per species
    for sample in numeric_cols:
        result_dict[sample] = {}
        
        for species in unique_species:
            # Get proteins of this species
            species_proteins = [
                pid for pid, sp in species_mapping.items() 
                if sp == species
            ]
            
            if not species_proteins:
                result_dict[sample][species] = 0
                continue
            
            # Count valid (intensity ≠ missing_value) in this species for this sample
            valid_count = 0
            for protein_id in species_proteins:
                if protein_id in df.index:
                    intensity = df.loc[protein_id, sample]
                    if pd.notna(intensity) and intensity != missing_value:
                        valid_count += 1
            
            result_dict[sample][species] = valid_count
    
    # Convert to DataFrame (species as rows, samples as columns)
    result_df = pd.DataFrame(result_dict).T
    result_df.index.name = "Species"
    
    return result_df


def count_missing_per_protein(
    df: pd.DataFrame,
    numeric_cols: List[str],
    species_mapping: Dict[str, str],
    missing_value: float = 1.0
) -> pd.DataFrame:
    """
    Count number of missing values per protein (per row).
    
    A value is missing if intensity == missing_value.
    Returns counts grouped by species.
    
    Args:
        df: DataFrame with protein IDs as index
        numeric_cols: List of sample column names
        species_mapping: Dict mapping protein ID → species name
        missing_value: Value indicating missing data (default 1.0)
        
    Returns:
        DataFrame with columns:
        - 'protein_id': Protein ID
        - 'species': Species name
        - 'missing_count': Number of missing values (0 to len(numeric_cols))
        
    Example:
        >>> Protein P1 (HUMAN) has intensities [5.0, 1.0, 3.0, 1.0] across 4 samples
        >>> missing_count = 2 (two 1.0 values)
    """
    results = []
    
    for protein_id in df.index:
        # Get species for this protein
        species = species_mapping.get(protein_id, "UNKNOWN")
        
        # Count missing values in this protein row
        row = df.loc[protein_id, numeric_cols]
        missing_count = (row == missing_value).sum()
        
        results.append({
            'protein_id': protein_id,
            'species': species,
            'missing_count': int(missing_count)
        })
    
    result_df = pd.DataFrame(results)
    return result_df
