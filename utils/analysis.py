"""
Analysis utilities for proteomics data processing
"""

import pandas as pd
import numpy as np


def process_data(df: pd.DataFrame, p_value_threshold: float = 0.05, 
                 fc_threshold: float = 1.0) -> pd.DataFrame:
    """
    Process proteomics data and assign significance categories
    
    Args:
        df: DataFrame with protein data including log2FoldChange and pValue
        p_value_threshold: P-value threshold for significance (as -log10)
        fc_threshold: Fold change threshold (log2 scale)
        
    Returns:
        DataFrame with added significance column
    """
    df = df.copy()
    
    # Calculate -log10(p-value) if not present
    if 'negLog10PValue' not in df.columns and 'pValue' in df.columns:
        df['negLog10PValue'] = -np.log10(df['pValue'].clip(lower=1e-300))
    
    # Assign significance categories
    conditions = [
        (df['negLog10PValue'] >= p_value_threshold) & (df['log2FoldChange'] >= fc_threshold),
        (df['negLog10PValue'] >= p_value_threshold) & (df['log2FoldChange'] <= -fc_threshold),
    ]
    choices = ['UP', 'DOWN']
    df['significance'] = np.select(conditions, choices, default='NS')
    
    return df


def calculate_stats(df: pd.DataFrame) -> dict:
    """
    Calculate summary statistics for processed data
    
    Args:
        df: DataFrame with significance column
        
    Returns:
        Dictionary with counts of different significance categories
    """
    if 'significance' not in df.columns:
        return {
            'total': len(df),
            'up': 0,
            'down': 0,
            'ns': len(df),
            'significant': 0
        }
    
    return {
        'total': len(df),
        'up': (df['significance'] == 'UP').sum(),
        'down': (df['significance'] == 'DOWN').sum(),
        'ns': (df['significance'] == 'NS').sum(),
        'significant': ((df['significance'] == 'UP') | (df['significance'] == 'DOWN')).sum()
    }
