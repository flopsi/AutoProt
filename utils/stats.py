"""
Statistical utilities for proteomics QC analysis
Includes PCA, normality tests, CV calculations, and data transformations
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


def calculate_cv(data: pd.DataFrame, replicate_cols: List[str]) -> pd.Series:
    """
    Calculate Coefficient of Variation for each protein across replicates
    
    Args:
        data: DataFrame with protein intensities
        replicate_cols: List of column names containing replicate data
        
    Returns:
        Series with CV values for each protein
    """
    means = data[replicate_cols].mean(axis=1)
