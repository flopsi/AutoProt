import pandas as pd
import numpy as np
from typing import Dict
def process_data(df: pd.DataFrame, p_val_cutoff: float, fc_cutoff: float) -> pd.DataFrame:
    """    Process proteomics data and assign significance categories    Args:        df: DataFrame with protein data        p_val_cutoff: -log10 p-value threshold        fc_cutoff: log2 fold change threshold    Returns:        DataFrame with significance column updated    """    df = df.copy()
    def assign_significance(row):
        if row['negLog10PValue'] >= p_val_cutoff:
            if row['log2FoldChange'] >= fc_cutoff:
                return 'UP'            elif row['log2FoldChange'] <= -fc_cutoff:
                return 'DOWN'        return 'NS'    df['significance'] = df.apply(assign_significance, axis=1)
    return df
def calculate_stats(df: pd.DataFrame) -> Dict[str, int]:
    """    Calculate summary statistics from processed data    Args:        df: Processed DataFrame with significance column    Returns:        Dictionary with counts    """    return {
        'total': len(df),
        'up': len(df[df['significance'] == 'UP']),
        'down': len(df[df['significance'] == 'DOWN']),
        'ns': len(df[df['significance'] == 'NS']),
        'significant': len(df[df['significance'] != 'NS'])
    }
