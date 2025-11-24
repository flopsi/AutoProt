"""
Minimal quality plotting utilities.
"""

import pandas as pd


def prepare_condition_data(quant_data, condition_mapping):
    """
    Split data into Condition A and B based on condition mapping.
    
    Args:
        quant_data: DataFrame with quantitative data
        condition_mapping: Dict mapping column names to conditions (A1, A2, B1, B2, etc.)
    
    Returns:
        tuple: (a_data, b_data) DataFrames for conditions A and B
    """
    a_cols = [col for col, cond in condition_mapping.items() if cond.startswith('A')]
    b_cols = [col for col, cond in condition_mapping.items() if cond.startswith('B')]
    
    a_data = quant_data[a_cols]
    b_data = quant_data[b_cols]
    
    return a_data, b_data
