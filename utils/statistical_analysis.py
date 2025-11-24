"""
Statistical analysis functions for proteomics data.
"""

import pandas as pd
import numpy as np

try:
    import limma_py
    LIMMA_AVAILABLE = True
except ImportError:
    LIMMA_AVAILABLE = False


def limma_by_species(a_data, b_data, species_map):
    """
    Perform limma differential expression analysis separately for each species.
    
    Args:
        a_data: DataFrame for Condition A (samples as columns)
        b_data: DataFrame for Condition B (samples as columns)
        species_map: Dict mapping protein IDs to species
    
    Returns:
        DataFrame with columns: protein_id, species, logFC, AveExpr, t, P.Value, adj.P.Val
    """
    if not LIMMA_AVAILABLE:
        raise ImportError("limma_py is not installed. Install with: pip install limma_py")
    
    results_list = []
    
    # Process each species separately
    for species in ['human', 'yeast', 'ecoli', 'celegans']:
        # Get proteins for this species
        species_proteins = [idx for idx, sp in species_map.items() if sp == species]
        
        if len(species_proteins) < 2:
            continue
        
        # Filter data to this species only
        species_a = a_data.loc[species_proteins]
        species_b = b_data.loc[species_proteins]
        
        # Combine data
        expr_matrix = pd.concat([species_a, species_b], axis=1)
        
        # Remove rows with too many missing values
        min_samples = 2
        valid_rows = (expr_matrix.notna().sum(axis=1) >= min_samples)
        expr_clean = expr_matrix[valid_rows]
        
        if len(expr_clean) < 2:
            continue
        
        # Fill remaining NAs with row mean
        expr_clean = expr_clean.fillna(expr_clean.mean(axis=1), axis=0)
        
        # Create design matrix
        n_a = species_a.shape[1]
        n_b = species_b.shape[1]
        group = np.array(['A'] * n_a + ['B'] * n_b)
        design_df = pd.get_dummies(group, drop_first=False)[['A', 'B']]
        design = design_df.astype(int)
        
        # Run limma
        fit = limma_py.lmFit(expr_clean, design)
        contrasts = limma_py.make_contrasts('A - B', levels=design)
        fit = limma_py.contrasts_fit(fit, contrasts)
        fit = limma_py.eBayes(fit)
        
        # Get results
        results = limma_py.toptable(fit, number=len(expr_clean))
        results['species'] = species
        results_list.append(results)
    
    # Combine all species results
    if results_list:
        all_results = pd.concat(results_list, axis=0)
        return all_results
    else:
        return pd.DataFrame()
