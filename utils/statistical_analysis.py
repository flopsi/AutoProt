def limma_by_species(a_data, b_data, species_map):
    """
    Perform differential expression analysis per species.
    Uses limma without eBayes to avoid unequal df issues.
    
    Args:
        a_data: DataFrame for Condition A (samples as columns)
        b_data: DataFrame for Condition B (samples as columns)
        species_map: Dict mapping protein IDs to species
    
    Returns:
        DataFrame with columns: logFC, AveExpr, t, P.Value, adj.P.Val, species
    """
    try:
        import limma_py
    except ImportError:
        raise ImportError("limma_py is not installed. Install with: pip install limma_py")
    
    from scipy.stats import t as t_dist
    
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
        expr_clean = expr_clean.T.fillna(expr_clean.mean(axis=1)).T
        
        # Create design matrix
        n_a = species_a.shape[1]
        n_b = species_b.shape[1]
        group = np.array(['A'] * n_a + ['B'] * n_b)
        design_df = pd.get_dummies(group, drop_first=False)[['A', 'B']]
        design = design_df.astype(int)
        
        # Run limma WITHOUT eBayes
        fit = limma_py.lmFit(expr_clean, design)
        contrasts = limma_py.make_contrasts('A - B', levels=design)
        fit = limma_py.contrasts_fit(fit, contrasts)
        
        # Manually extract results without eBayes
        logFC = fit['coefficients'].iloc[:, 0]
        
        # Calculate average expression
        aveExpr = expr_clean.mean(axis=1)
        
        # Calculate standard error and t-statistic manually
        residuals = fit['residuals']
        df_residual = n_a + n_b - 2
        
        # Calculate sigma (residual standard deviation)
        sigma = np.sqrt(np.sum(residuals**2, axis=1) / df_residual)
        
        # Standard error of the coefficient
        # For contrast A - B with equal group sizes: SE = sigma * sqrt(1/n_a + 1/n_b)
        se = sigma * np.sqrt(1/n_a + 1/n_b)
        
        # t-statistic
        t_stat = logFC / se
        
        # p-value (two-tailed)
        p_value = 2 * (1 - t_dist.cdf(np.abs(t_stat), df_residual))
        
        # Create results dataframe
        species_results = pd.DataFrame({
            'logFC': logFC,
            'AveExpr': aveExpr,
            't': t_stat,
            'P.Value': p_value,
            'species': species
        })
        
        # Multiple testing correction (Benjamini-Hochberg)
        p_vals = species_results['P.Value'].values
        n = len(p_vals)
        
        # Sort and rank
        sort_idx = np.argsort(p_vals)
        ranks = np.empty(n, dtype=int)
        ranks[sort_idx] = np.arange(1, n + 1)
        
        # BH correction
        adj_p = np.minimum(1.0, p_vals * n / ranks)
        species_results['adj.P.Val'] = adj_p
        
        results_list.append(species_results)
    
    # Combine all species results
    if results_list:
        all_results = pd.concat(results_list, axis=0)
        return all_results
    else:
        return pd.DataFrame()
