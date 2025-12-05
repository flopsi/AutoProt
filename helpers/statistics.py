# Add to existing imports at top
from scipy.stats import shapiro, anderson

# ============================================================================
# NORMALITY TESTING
# ============================================================================

def test_normality_all_samples(
    df_raw: pd.DataFrame,
    df_transformed: pd.DataFrame,
    numeric_cols: list,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Test normality (Shapiro-Wilk) for each sample before & after transformation.
    
    Parameters:
    -----------
    df_raw : pd.DataFrame
        Raw data (untransformed)
    df_transformed : pd.DataFrame
        Transformed data
    numeric_cols : list
        List of numeric column names to test
    alpha : float
        Significance level (default 0.05)
    
    Returns:
    --------
    pd.DataFrame with columns:
        - Sample: Column name
        - N: Sample size (non-missing values)
        - Raw_Statistic: Shapiro-Wilk W statistic for raw data
        - Raw_P_Value: p-value for raw data
        - Raw_Normal: True if p > alpha (normally distributed)
        - Trans_Statistic: Shapiro-Wilk W statistic for transformed data
        - Trans_P_Value: p-value for transformed data
        - Trans_Normal: True if p > alpha (normally distributed)
        - Improvement: "Yes" if transformation made data more normal
    """
    results = []
    
    for col in numeric_cols:
        # Get raw data (drop NaN)
        raw_vals = df_raw[col].dropna().values
        trans_vals = df_transformed[col].dropna().values
        
        n = len(raw_vals)
        
        # Skip if too few samples
        if n < 3:
            results.append({
                'Sample': col,
                'N': n,
                'Raw_Statistic': np.nan,
                'Raw_P_Value': np.nan,
                'Raw_Normal': False,
                'Trans_Statistic': np.nan,
                'Trans_P_Value': np.nan,
                'Trans_Normal': False,
                'Improvement': 'N/A'
            })
            continue
        
        # Shapiro-Wilk test for raw data
        try:
            if n <= 5000:  # Shapiro-Wilk works well for n <= 5000
                stat_raw, p_raw = shapiro(raw_vals)
            else:
                # For large samples, use Anderson-Darling
                result = anderson(raw_vals)
                stat_raw = result.statistic
                # Approximate p-value (conservative)
                p_raw = 0.01 if stat_raw > result.critical_values[2] else 0.10
        except Exception as e:
            stat_raw, p_raw = np.nan, np.nan
        
        # Shapiro-Wilk test for transformed data
        try:
            if n <= 5000:
                stat_trans, p_trans = shapiro(trans_vals)
            else:
                result = anderson(trans_vals)
                stat_trans = result.statistic
                p_trans = 0.01 if stat_trans > result.critical_values[2] else 0.10
        except Exception as e:
            stat_trans, p_trans = np.nan, np.nan
        
        # Determine normality
        is_normal_raw = p_raw > alpha if not np.isnan(p_raw) else False
        is_normal_trans = p_trans > alpha if not np.isnan(p_trans) else False
        
        # Check improvement
        if not is_normal_raw and is_normal_trans:
            improvement = "✅ Yes"
        elif is_normal_raw and not is_normal_trans:
            improvement = "⚠️ Worse"
        elif not is_normal_raw and not is_normal_trans:
            improvement = "➖ No Change"
        else:
            improvement = "✓ Both Normal"
        
        results.append({
            'Sample': col,
            'N': n,
            'Raw_Statistic': stat_raw,
            'Raw_P_Value': p_raw,
            'Raw_Normal': is_normal_raw,
            'Trans_Statistic': stat_trans,
            'Trans_P_Value': p_trans,
            'Trans_Normal': is_normal_trans,
            'Improvement': improvement
        })
    
    return pd.DataFrame(results)
