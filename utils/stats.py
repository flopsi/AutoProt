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
    stds = data[replicate_cols].std(axis=1)
    
    # Avoid division by zero
    cv = np.where(means != 0, (stds / means) * 100, np.nan)
    
    return pd.Series(cv, index=data.index, name='CV')


def check_normality(data: pd.Series, alpha: float = 0.05) -> Dict[str, float]:
    """
    Test normality of data distribution using Shapiro-Wilk test
    
    Args:
        data: Series of values to test
        alpha: Significance level
        
    Returns:
        Dictionary with test statistics and interpretation
    """
    # Remove NaN values
    clean_data = data.dropna()
    
    if len(clean_data) < 3:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'is_normal': False,
            'skewness': np.nan,
            'kurtosis': np.nan,
            'message': 'Insufficient data for normality test'
        }
    
    # Shapiro-Wilk test
    statistic, p_value = shapiro(clean_data)
    
    # Calculate skewness and kurtosis
    skewness = skew(clean_data)
    kurt = kurtosis(clean_data)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'is_normal': p_value > alpha,
        'skewness': skewness,
        'kurtosis': kurt,
        'message': f"{'Normal' if p_value > alpha else 'Non-normal'} distribution (p={p_value:.4f})"
    }


def log2_transform(data: pd.DataFrame, replicate_cols: List[str], 
                   pseudocount: float = 1.0) -> pd.DataFrame:
    """
    Apply log2 transformation to intensity data
    
    Args:
        data: DataFrame with protein intensities
        replicate_cols: Columns to transform
        pseudocount: Value to add before log transformation (to avoid log(0))
        
    Returns:
        DataFrame with transformed values
    """
    transformed = data.copy()
    
    for col in replicate_cols:
        transformed[col] = np.log2(data[col] + pseudocount)
    
    return transformed


def perform_pca(data: pd.DataFrame, replicate_cols: List[str], 
                n_components: int = 2) -> Tuple[pd.DataFrame, np.ndarray, PCA]:
    """
    Perform Principal Component Analysis on replicate data
    
    Args:
        data: DataFrame with protein intensities
        replicate_cols: Columns to include in PCA
        n_components: Number of principal components
        
    Returns:
        Tuple of (PC coordinates DataFrame, explained variance, PCA model)
    """
    # Transpose so samples are rows
    data_for_pca = data[replicate_cols].T
    
    # Remove columns (proteins) with any NaN
    data_clean = data_for_pca.dropna(axis=1)
    
    if data_clean.shape < n_components:
        raise ValueError(f"Not enough features for PCA. Need at least {n_components}, got {data_clean.shape}")
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_clean)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pc_coords = pca.fit_transform(data_scaled)
    
    # Create DataFrame with PC coordinates
    pc_df = pd.DataFrame(
        pc_coords,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=replicate_cols
    )
    
    return pc_df, pca.explained_variance_ratio_, pca


def calculate_missing_values(data: pd.DataFrame, replicate_cols: List[str]) -> pd.DataFrame:
    """
    Calculate missing value patterns across samples
    
    Args:
        data: DataFrame with protein intensities
        replicate_cols: Columns to analyze
        
    Returns:
        DataFrame with missing value statistics
    """
    missing_stats = pd.DataFrame({
        'Sample': replicate_cols,
        'Missing_Count': [data[col].isna().sum() for col in replicate_cols],
        'Missing_Percent': [data[col].isna().sum() / len(data) * 100 for col in replicate_cols],
        'Present_Count': [data[col].notna().sum() for col in replicate_cols]
    })
    
    return missing_stats


def calculate_quartiles(data: pd.Series) -> Dict[str, float]:
    """
    Calculate quartiles and IQR for a data series
    
    Args:
        data: Series of values
        
    Returns:
        Dictionary with quartile statistics
    """
    clean_data = data.dropna()
    
    if len(clean_data) == 0:
        return {
            'min': np.nan,
            'q1': np.nan,
            'median': np.nan,
            'q3': np.nan,
            'max': np.nan,
            'iqr': np.nan
        }
    
    q1 = np.percentile(clean_data, 25)
    median = np.percentile(clean_data, 50)
    q3 = np.percentile(clean_data, 75)
    
    return {
        'min': clean_data.min(),
        'q1': q1,
        'median': median,
        'q3': q3,
        'max': clean_data.max(),
        'iqr': q3 - q1
    }


def detect_outliers_iqr(data: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Detect outliers using IQR method
    
    Args:
        data: Series of values
        multiplier: IQR multiplier for outlier detection
        
    Returns:
        Boolean series indicating outliers
    """
    quartiles = calculate_quartiles(data)
    
    if np.isnan(quartiles['iqr']):
        return pd.Series([False] * len(data), index=data.index)
    
    lower_bound = quartiles['q1'] - multiplier * quartiles['iqr']
    upper_bound = quartiles['q3'] + multiplier * quartiles['iqr']
    
    outliers = (data < lower_bound) | (data > upper_bound)
    
    return outliers


def calculate_dynamic_range(data: pd.DataFrame, replicate_cols: List[str]) -> pd.DataFrame:
    """
    Calculate dynamic range statistics for rank plots
    
    Args:
        data: DataFrame with protein intensities
        replicate_cols: Columns to analyze
        
    Returns:
        DataFrame with mean intensity and rank for each protein
    """
    # Calculate mean across replicates
    mean_intensity = data[replicate_cols].mean(axis=1)
    
    # Remove NaN and sort
    mean_intensity_clean = mean_intensity.dropna().sort_values(ascending=False)
    
    # Create rank
    rank_df = pd.DataFrame({
        'Protein': mean_intensity_clean.index,
        'Mean_Intensity': mean_intensity_clean.values,
        'Rank': range(1, len(mean_intensity_clean) + 1)
    })
    
    return rank_df


def perform_t_test(group1: pd.Series, group2: pd.Series) -> Dict[str, float]:
    """
    Perform two-sample t-test between groups
    
    Args:
        group1: First group of values
        group2: Second group of values
        
    Returns:
        Dictionary with test results
    """
    # Remove NaN
    g1_clean = group1.dropna()
    g2_clean = group2.dropna()
    
    if len(g1_clean) < 2 or len(g2_clean) < 2:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'mean_diff': np.nan,
            'message': 'Insufficient data for t-test'
        }
    
    # Perform t-test
    statistic, p_value = stats.ttest_ind(g1_clean, g2_clean)
    
    mean_diff = g1_clean.mean() - g2_clean.mean()
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'mean_diff': mean_diff,
        'significant': p_value < 0.05
    }


def calculate_fold_change(group1_mean: float, group2_mean: float, 
                         log_scale: bool = True) -> float:
    """
    Calculate fold change between two groups
    
    Args:
        group1_mean: Mean of group 1
        group2_mean: Mean of group 2
        log_scale: Whether data is already log-transformed
        
    Returns:
        Fold change value
    """
    if log_scale:
        # If already log2, fold change is simply the difference
        return group1_mean - group2_mean
    else:
        # Calculate ratio and convert to log2
        if group2_mean == 0:
            return np.nan
        ratio = group1_mean / group2_mean
        return np.log2(ratio)


def batch_process_proteins(data: pd.DataFrame, 
                          condition_a_cols: List[str],
                          condition_b_cols: List[str],
                          log_transformed: bool = False) -> pd.DataFrame:
    """
    Process all proteins: calculate statistics and perform t-tests
    
    Args:
        data: DataFrame with protein intensities
        condition_a_cols: Replicate columns for condition A
        condition_b_cols: Replicate columns for condition B
        log_transformed: Whether data is log-transformed
        
    Returns:
        DataFrame with all statistics per protein
    """
    results = []
    
    for idx, row in data.iterrows():
        group_a = row[condition_a_cols]
        group_b = row[condition_b_cols]
        
        # Calculate means
        mean_a = group_a.mean()
        mean_b = group_b.mean()
        
        # Calculate fold change
        fc = calculate_fold_change(mean_a, mean_b, log_transformed)
        
        # Perform t-test
        ttest_result = perform_t_test(group_a, group_b)
        
        # Calculate CVs
        cv_a = calculate_cv(pd.DataFrame([row]), condition_a_cols).iloc
        cv_b = calculate_cv(pd.DataFrame([row]), condition_b_cols).iloc
        
        results.append({
            'protein_id': idx,
            'mean_condition_a': mean_a,
            'mean_condition_b': mean_b,
            'log2_fold_change': fc,
            't_statistic': ttest_result['statistic'],
            'p_value': ttest_result['p_value'],
            'cv_condition_a': cv_a,
            'cv_condition_b': cv_b,
            'significant': ttest_result.get('significant', False)
        })
    
    return pd.DataFrame(results)
