"""
helpers/comparison.py
Multi-transformation comparison with DE analysis metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from helpers.transforms import apply_transformation
from helpers.evaluation import evaluate_transformation

def find_best_transformation(summary_df: pd.DataFrame) -> Tuple[str, str]:
    """Find best transformation based on multiple criteria."""
    if summary_df.empty:
        return 'log2', 'default'
    
    # Ensure combined_score exists
    if 'combined_score' not in summary_df.columns:
        summary_df['combined_score'] = (
            summary_df['shapiro_p'].rank(ascending=False) +
            (1 - summary_df['mean_var_corr'].abs()).rank(ascending=False) +
            summary_df['n_significant'].rank(ascending=False)
        )


def compare_transformations(
    df_raw: pd.DataFrame,
    numeric_cols: list,
    transformations: List[str] = None,
    max_cols: int = 6
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Compare multiple transformations and return summary.
    
    Parameters:
    -----------
    df_raw : pd.DataFrame
        Original raw data
    numeric_cols : list
        Numeric (intensity) columns
    transformations : list
        List of transformation methods to test
    max_cols : int
        Max columns for analysis/plots
    
    Returns:
    --------
    tuple: (summary_df, all_results_dict)
        - summary_df: Comparison table
        - all_results_dict: {method: transformed_df}
    """
    if transformations is None:
        transformations = ['log2', 'log10', 'sqrt', 'arcsinh', 'vst', 'yeo-johnson']
    
    comparison_results = []
    all_transformed_data = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, method in enumerate(transformations):
        try:
            status_text.text(f"Testing {method.upper()}...")
            
            # Apply transformation
            df_transformed, trans_cols = apply_transformation(
                df_raw.copy(), 
                numeric_cols[:max_cols], 
                method
            )
            
            # Evaluate transformation quality
            eval_stats = evaluate_transformation(
                df_raw, df_transformed, numeric_cols[:max_cols], 
                trans_cols[:max_cols], method
            )
            
            # Store transformed data
            all_transformed_data[method] = df_transformed
            
            # Mock DE results (replace with your actual differential_analysis_limma_style)
            n_total = len(df_transformed)
            n_significant = int(n_total * np.random.uniform(0.05, 0.25))  # Mock
            n_up = int(n_significant * np.random.uniform(0.4, 0.6))
            n_down = n_significant - n_up
            
            comparison_results.append({
                'method': method,
                'n_significant': n_significant,
                'n_up': n_up,
                'n_down': n_down,
                'shapiro_p': eval_stats['shapiro_transformed'],
                'mean_var_corr': eval_stats['mean_var_corr_trans'],
                'improvement': '✅' if eval_stats['shapiro_transformed'] > eval_stats['shapiro_raw'] else '➖'
            })
            
        except Exception as e:
            st.error(f"❌ Error with {method}: {str(e)}")
            continue
        
        progress_bar.progress((i + 1) / len(transformations))
    
    progress_bar.empty()
    status_text.empty()
    
    summary_df = pd.DataFrame(comparison_results)
    return summary_df, all_transformed_data


def find_best_transformation(summary_df: pd.DataFrame) -> Tuple[str, str]:
    """
    Find best transformation based on multiple criteria.
    
    Returns:
    --------
    tuple: (best_method, reason)
    """
    if summary_df.empty:
        return 'log2', 'default'
    
    # Criteria 1: Highest Shapiro p-value (normality)
    best_normality = summary_df.loc[summary_df['shapiro_p'].idxmax(), 'method']
    
    # Criteria 2: Lowest |mean_var_corr| (variance stabilization)
    best_stabilization = summary_df.loc[summary_df['mean_var_corr'].abs().idxmin(), 'method']
    
    # Criteria 3: Most significant DE results
    best_de = summary_df.loc[summary_df['n_significant'].idxmax(), 'method']
    
    # Combined score
    summary_df['combined_score'] = (
        summary_df['shapiro_p'].rank(ascending=False) +
        (1 - summary_df['mean_var_corr'].abs()).rank(ascending=False) +
        summary_df['n_significant'].rank(ascending=False)
    )
    
    best_overall = summary_df.loc[summary_df['combined_score'].idxmax(), 'method']
    
    return best_overall, f"normality={best_normality}, stabilization={best_stabilization}, DE={best_de}"
