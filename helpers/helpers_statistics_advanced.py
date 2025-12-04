"""
helpers/statistics_advanced.py
Advanced statistical methods for multi-group comparisons
ANOVA, post-hoc tests, effect sizes, confidence intervals
"""

import pandas as pd
import numpy as np
from scipy.stats import f_oneway, tukey_hsd
from typing import Dict, Tuple, List

# ============================================================================
# ANOVA & MULTI-GROUP TESTING
# ============================================================================

def perform_anova(
    df: pd.DataFrame,
    group_cols_dict: Dict[str, List[str]],
    min_valid: int = 2,
) -> Dict:
    """
    One-way ANOVA to compare 3+ groups.
    Tests null hypothesis: all group means are equal.
    
    Args:
        df: Data matrix (proteins × samples)
        group_cols_dict: {"GroupA": ["A1", "A2"], "GroupB": ["B1", "B2"], ...}
        min_valid: Minimum valid values per group
        
    Returns:
        Dict with f_stat, p_value, group_means, variances, summary
        
    Example:
        groups = {"Control": ["C1", "C2"], "Treatment": ["T1", "T2"]}
        results = perform_anova(df, groups)
        # results["f_stat"] = 5.23
        # results["p_value"] = 0.032
    """
    
    results = []
    
    # Extract group means for each protein
    for protein_id, row in df.iterrows():
        group_means = {}
        group_sizes = {}
        group_data = []
        
        # Extract data for each group
        for group_name, cols in group_cols_dict.items():
            vals = row[cols].dropna()
            
            if len(vals) < min_valid:
                # Skip if insufficient data
                group_data = None
                break
            
            group_means[group_name] = vals.mean()
            group_sizes[group_name] = len(vals)
            group_data.append(vals.values)
        
        if group_data is None or len(group_data) < len(group_cols_dict):
            results.append({
                "protein_id": protein_id,
                "f_stat": np.nan,
                "p_value": np.nan,
                "mse": np.nan,
            })
            continue
        
        # Perform ANOVA
        try:
            f_stat, p_val = f_oneway(*group_data)
        except Exception:
            f_stat, p_val = np.nan, np.nan
        
        # Calculate MSE (mean squared error)
        grand_mean = np.concatenate(group_data).mean()
        sse = sum((vals - grand_mean).sum() for vals in group_data)
        df_residual = sum(len(g) for g in group_data) - len(group_cols_dict)
        mse = sse / df_residual if df_residual > 0 else np.nan
        
        results.append({
            "protein_id": protein_id,
            "f_stat": f_stat,
            "p_value": p_val,
            "mse": mse,
        })
    
    results_df = pd.DataFrame(results)
    results_df.set_index("protein_id", inplace=True)
    
    # Add -log10(p) for plotting
    results_df["neg_log10_pval"] = -np.log10(
        results_df["p_value"].replace(0, 1e-300)
    )
    
    return results_df


# ============================================================================
# EFFECT SIZE CALCULATIONS
# ============================================================================

def calculate_cohens_d(g1: pd.Series, g2: pd.Series) -> float:
    """
    Cohen's d effect size between two groups.
    
    d = (mean1 - mean2) / pooled_std
    
    Interpretation:
    - 0.0-0.2: negligible
    - 0.2-0.5: small
    - 0.5-0.8: medium
    - 0.8+: large
    
    Args:
        g1: First group values
        g2: Second group values
        
    Returns:
        Cohen's d (float)
    """
    g1_clean = g1.dropna()
    g2_clean = g2.dropna()
    
    if len(g1_clean) < 2 or len(g2_clean) < 2:
        return np.nan
    
    n1, n2 = len(g1_clean), len(g2_clean)
    var1, var2 = g1_clean.var(ddof=1), g2_clean.var(ddof=1)
    
    # Pooled standard deviation
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)
    
    if pooled_std == 0:
        return 0.0
    
    cohens_d = (g1_clean.mean() - g2_clean.mean()) / pooled_std
    
    return cohens_d


def calculate_eta_squared(df: pd.DataFrame, groups_dict: Dict[str, List[str]]) -> float:
    """
    Eta-squared (η²) effect size for ANOVA.
    Proportion of variance explained by group membership.
    
    η² = SS_between / SS_total
    
    Args:
        df: Data matrix
        groups_dict: {"GroupA": [...], "GroupB": [...], ...}
        
    Returns:
        eta_squared value (0 to 1)
    """
    # Combine all groups
    all_data = []
    group_assignments = []
    
    for group_name, cols in groups_dict.items():
        for col in cols:
            vals = df[col].dropna()
            all_data.extend(vals)
            group_assignments.extend([group_name] * len(vals))
    
    if len(all_data) == 0:
        return np.nan
    
    all_data = np.array(all_data)
    grand_mean = all_data.mean()
    
    # SS_total
    ss_total = np.sum((all_data - grand_mean) ** 2)
    
    # SS_between
    ss_between = 0
    for group_name, cols in groups_dict.items():
        group_data = []
        for col in cols:
            group_data.extend(df[col].dropna())
        
        group_data = np.array(group_data)
        group_mean = group_data.mean()
        ss_between += len(group_data) * (group_mean - grand_mean) ** 2
    
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    return eta_squared


# ============================================================================
# CONFIDENCE INTERVALS
# ============================================================================

def calculate_ci_95(mean: float, se: float, n: int, ci: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for mean using t-distribution.
    
    Args:
        mean: Sample mean
        se: Standard error
        n: Sample size
        ci: Confidence level (default 0.95 = 95%)
        
    Returns:
        (lower_bound, upper_bound)
    """
    from scipy.stats import t
    
    df = n - 1
    alpha = 1 - ci
    t_crit = t.ppf(1 - alpha / 2, df)
    
    margin = t_crit * se
    
    return (mean - margin, mean + margin)


def calculate_fc_confidence_intervals(
    results_df: pd.DataFrame,
    df_log2: pd.DataFrame,
    group1_cols: List[str],
    group2_cols: List[str],
    ci: float = 0.95,
) -> pd.DataFrame:
    """
    Calculate confidence intervals for log2 fold changes.
    
    Args:
        results_df: Results with log2fc
        df_log2: Log2-transformed data
        group1_cols: Columns in group 1
        group2_cols: Columns in group 2
        ci: Confidence level
        
    Returns:
        DataFrame with lower_ci, upper_ci columns
    """
    ci_results = []
    
    for protein_id, row in df_log2.iterrows():
        g1 = row[group1_cols].dropna()
        g2 = row[group2_cols].dropna()
        
        if len(g1) < 2 or len(g2) < 2:
            ci_results.append({
                "protein_id": protein_id,
                "lower_ci": np.nan,
                "upper_ci": np.nan,
            })
            continue
        
        # Combined SE
        se1 = g1.std(ddof=1) / np.sqrt(len(g1))
        se2 = g2.std(ddof=1) / np.sqrt(len(g2))
        se_combined = np.sqrt(se1**2 + se2**2)
        
        # Get log2FC
        log2fc = results_df.loc[protein_id, "log2fc"]
        
        # CI
        lower, upper = calculate_ci_95(log2fc, se_combined, len(g1) + len(g2) - 2, ci)
        
        ci_results.append({
            "protein_id": protein_id,
            "lower_ci": lower,
            "upper_ci": upper,
        })
    
    return pd.DataFrame(ci_results).set_index("protein_id")


# ============================================================================
# POST-HOC TESTING
# ============================================================================

def tukey_hsd_pairwise(
    df: pd.DataFrame,
    groups_dict: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Tukey HSD post-hoc test for pairwise comparisons.
    
    Args:
        df: Data matrix (proteins × samples)
        groups_dict: Group definitions
        
    Returns:
        DataFrame with pairwise p-values
    """
    
    group_names = list(groups_dict.keys())
    n_comparisons = len(group_names) * (len(group_names) - 1) // 2
    
    results = []
    
    for protein_id, row in df.iterrows():
        # Extract group data
        group_data = []
        for group_name in group_names:
            cols = groups_dict[group_name]
            vals = row[cols].dropna()
            group_data.append(vals.values)
        
        # Perform Tukey HSD
        try:
            res = tukey_hsd(*group_data)
            
            # Extract pairwise p-values
            idx = 0
            for i in range(len(group_names)):
                for j in range(i + 1, len(group_names)):
                    comparison = f"{group_names[i]} vs {group_names[j]}"
                    p_val = res.pvalue[i, j]
                    
                    results.append({
                        "protein_id": protein_id,
                        "comparison": comparison,
                        "p_value": p_val,
                    })
                    idx += 1
        except Exception:
            for i in range(len(group_names)):
                for j in range(i + 1, len(group_names)):
                    comparison = f"{group_names[i]} vs {group_names[j]}"
                    results.append({
                        "protein_id": protein_id,
                        "comparison": comparison,
                        "p_value": np.nan,
                    })
    
    return pd.DataFrame(results)


# ============================================================================
# POWER ANALYSIS
# ============================================================================

def estimate_power(
    effect_size: float,
    sample_size_per_group: int,
    alpha: float = 0.05,
) -> float:
    """
    Estimate statistical power for two-sample t-test.
    
    Args:
        effect_size: Cohen's d
        sample_size_per_group: n per group
        alpha: Significance level
        
    Returns:
        Estimated power (0 to 1)
    """
    from scipy.stats import nct, t
    
    # Non-centrality parameter
    lambda_nc = effect_size * np.sqrt(sample_size_per_group / 2)
    
    # Critical value
    df = 2 * (sample_size_per_group - 1)
    t_crit = t.ppf(1 - alpha / 2, df)
    
    # Power = P(T > t_crit | H1)
    power = 1 - nct.cdf(t_crit, df, lambda_nc) + nct.cdf(-t_crit, df, lambda_nc)
    
    return power


def estimate_required_sample_size(
    effect_size: float,
    target_power: float = 0.8,
    alpha: float = 0.05,
) -> int:
    """
    Estimate required sample size per group for desired power.
    
    Args:
        effect_size: Cohen's d
        target_power: Desired power (default 0.8)
        alpha: Significance level
        
    Returns:
        Required sample size per group
    """
    
    # Binary search for required n
    n_low, n_high = 2, 1000
    
    while n_high - n_low > 1:
        n_mid = (n_low + n_high) // 2
        power = estimate_power(effect_size, n_mid, alpha)
        
        if power < target_power:
            n_low = n_mid
        else:
            n_high = n_mid
    
    return n_high
