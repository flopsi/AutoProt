import pandas as pd
from typing import Dict
from models.proteomics_data import ProteomicsDataset


def create_species_condition_summary(protein_data: ProteomicsDataset) -> pd.DataFrame:
    """
    Create a summary DataFrame with protein counts by species and condition.
    
    Returns:
        DataFrame with columns: Species, Total, Condition_A, Condition_B
    """
    species_order = ['human', 'ecoli', 'yeast']
    
    # Get total counts
    total_counts = protein_data.get_species_counts()
    
    # Get Condition A counts
    a_data = protein_data.get_condition_data('A')
    a_detected_indices = a_data.dropna(how='all').index
    a_counts = {
        sp: sum(1 for idx in a_detected_indices if protein_data.species_map.get(idx) == sp)
        for sp in species_order
    }
    
    # Get Condition B counts
    b_data = protein_data.get_condition_data('B')
    b_detected_indices = b_data.dropna(how='all').index
    b_counts = {
        sp: sum(1 for idx in b_detected_indices if protein_data.species_map.get(idx) == sp)
        for sp in species_order
    }
    
    # Create DataFrame
    df = pd.DataFrame({
        'Species': [sp.capitalize() for sp in species_order],
        'Total': [total_counts.get(sp, 0) for sp in species_order],
        'Condition_A': [a_counts[sp] for sp in species_order],
        'Condition_B': [b_counts[sp] for sp in species_order]
    })
    
    # Add percentages
    total_proteins = df['Total'].sum()
    df['Total_%'] = (df['Total'] / total_proteins * 100).round(1)
    
    total_a = df['Condition_A'].sum()
    df['Condition_A_%'] = (df['Condition_A'] / total_a * 100).round(1)
    
    total_b = df['Condition_B'].sum()
    df['Condition_B_%'] = (df['Condition_B'] / total_b * 100).round(1)
    
    return df
