"""
helpers/stats.py

Statistical testing and analysis functions for differential expression
Consolidates t-tests, ANOVA, FDR correction, effect sizes, and normality tests
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, tukey_hsd, shapiro
from typing import Dict, Tuple, List
import streamlit as st

# ============================================================================
# NORMALITY TESTING
# Assess whether data follows normal distribution
# ============================================================================

def test_normality_shapiro(
    series: pd.Series,
    alpha: float = 0.05,
) -> Dict:
    """
    Shapiro-Wilk normality test on 1D numeric series.
    
    Args:
        series: Data to test
        alpha: Significance level (default 0.05)
    
    Returns:
        Dict with keys: statistic, p_value, is_normal, alpha, n
    """
    x = pd.to_numeric(series, errors="coerce").dropna().values
    n = len(x)
    
    if n < 3:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "is_normal": False,
            "alpha": alpha,
            "n": n,
        }
    
    stat, p = stats.shapiro(x)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "is_normal": bool(p > alpha),
        "alpha": alpha,
        "n": n,
    }

@st.cache_data(ttl=3600)
def test_normality_all_samples(
    df_raw: pd.DataFrame,
    df_transformed: pd.DataFrame,
    numeric_cols: List[str],
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Test normality (Shapiro-Wilk) for each sample before & after transformation.
    Cached to avoid recomputation on widget interactions.
    
    Args:
        df_raw: Original untransformed data
        df_transformed: Transformed data
        numeric_cols: Column names to test
        alpha: Significance threshold
    
    Returns:
        DataFrame with per-sample normality statistics
    """
    results = []
    
    for col in numeric_cols:
        # Extract data
        raw_vals = df_raw[col].dropna().values
        trans_vals = df_transformed[col].dropna().values
        n = len(raw_vals)
        
        # Skip if too few samples
        if n < 3:
            results.append({
                'Sample': col,
                'N': int(n),
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
            stat_raw, p_raw = shapiro(raw_vals)
        except:
            stat_raw, p_raw = np.nan, np.nan
        
        # Shapiro-Wilk test for transformed data
        try:
            stat_trans, p_trans = shapiro(trans_vals)
        except:
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
            'N': int(n),
            'Raw_Statistic': float(stat_raw) if not np.isnan(stat_raw) else None,
            'Raw_P_Value': float(p_raw) if not np.isnan(p_raw) else None,
            'Raw_Normal': bool(is_normal_raw),
            'Trans_Statistic': float(stat_trans) if not np.isnan(stat_trans) else None,
            'Trans_P_Value': float(p_trans) if not np.isnan(p_trans) else None,
            'Trans_Normal': bool(is_normal_trans),
            'Improvement': improvement
        })
    
    return pd.DataFrame(results)

# ============================================================================
# T-TEST & FDR CORRECTION
# Two-group differential expression testing
# ============================================================================

@st.cache_data(ttl=3600, show_spinner="Running t-tests...")
def perform_ttest(
    df: pd.DataFrame,
    group1_cols: List[str],
    group2_cols: List[str],
    min_valid: int = 2,
) -> pd.DataFrame:
    """
    Perform Welch's t-test on transformed data with FDR correction.
    Compares group1 (reference) vs group2 (treatment).
    Cached to avoid recomputation when changing visualization options.
    
    Convention: log2FC = mean(group1) - mean(group2)
    - Positive FC = higher in reference group
    - Negative FC = higher in treatment group
    
    Args:
        df: Transformed intensity data (log2-scale recommended)
        group1_cols: Column names for reference/control group
        group2_cols: Column names for treatment group
        min_valid: Minimum non-missing values required per group
    
    Returns:
        DataFrame with columns:
        - log2fc: Fold change
        - pvalue: Welch's t-test p-value
        - fdr: Benjamini-Hochberg corrected p-value
        - mean_g1, mean_g2: Group means
        - n_g1, n_g2: Valid sample counts
        - neg_log10_pval: -log10(p) for volcano plots
    """
    results = []
    
    # Loop over each protein
    for protein_id, row in df.iterrows():
        g1_vals = row[group1_cols].dropna()
        g2_vals = row[group2_cols].dropna()
        
        # Check minimum requirement
        if len(g1_vals) < min_valid or len(g2_vals) < min_valid:
            results.append({
                "protein_id": protein_id,
                "log2fc": np.nan,
                "pvalue": np.nan,
                "mean_g1": np.nan,
                "mean_g2": np.nan,
                "n_g1": len(g1_vals),
                "n_g2": len(g2_vals),
            })
            continue
        
        # Calculate means
        mean_g1 = g1_vals.mean()
        mean_g2 = g2_vals.mean()
        log2fc = mean_g1 - mean_g2
        
        # Perform Welch's t-test (unequal variances)
        try:
            t_stat, pval = ttest_ind(g1_vals, g2_vals, equal_var=False)
        except Exception:
            t_stat, pval = np.nan, np.nan
        
        results.append({
            "protein_id": protein_id,
            "log2fc": log2fc,
            "pvalue": pval,
            "mean_g1": mean_g1,
            "mean_g2": mean_g2,
            "n_g1": len(g1_vals),
            "n_g2": len(g2_vals),
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df.set_index("protein_id", inplace=True)
    
    # --- FDR Correction (Benjamini-Hochberg) ---
    pvals = results_df["pvalue"].dropna().sort_values()
    n = len(pvals)
    
    if n > 0:
        ranks = np.arange(1, n + 1)
        fdr_vals = pvals.values * n / ranks
        # Ensure FDR is monotonic
        fdr_vals = np.minimum.accumulate(fdr_vals[::-1])[::-1]
        fdr_dict = dict(zip(pvals.index, fdr_vals))
        results_df["fdr"] = results_df.index.map(fdr_dict)
    else:
        results_df["fdr"] = np.nan
    
    # Add -log10(p) for volcano plots
    results_df["neg_log10_pval"] = -np.log10(
        results_df["pvalue"].replace(0, 1e-300)
    )
    
    return results_df

# ============================================================================
# PROTEIN CLASSIFICATION
# Categorize proteins by regulation status
# ============================================================================

def classify_regulation(
    log2fc: float,
    pvalue: float,
    fc_threshold: float = 1.0,
    pval_threshold: float = 0.05,
) -> str:
    """
    Classify protein regulation status based on thresholds.
    
    Args:
        log2fc: log2 fold change
        pvalue: Statistical p-value (or FDR)
        fc_threshold: Absolute log2FC cutoff (default 1.0 = 2-fold)
        pval_threshold: P-value significance cutoff
    
    Returns:
        One of: "up", "down", "not_significant", "not_tested"
    """
    if pd.isna(log2fc) or pd.isna(pvalue):
        return "not_tested"
    
    if pvalue > pval_threshold:
        return "not_significant"
    
    if log2fc > fc_threshold:
        return "up"
    elif log2fc < -fc_threshold:
        return "down"
    else:
        return "not_significant"

# ============================================================================
# ANOVA & MULTI-GROUP TESTING
# Compare 3+ groups simultaneously
# ============================================================================

@st.cache_data(ttl=3600, show_spinner="Running ANOVA...")
def perform_anova(
    df: pd.DataFrame,
    group_cols_dict: Dict[str, List[str]],
    min_valid: int = 2,
) -> pd.DataFrame:
    """
    One-way ANOVA to compare 3+ groups.
    Tests null hypothesis: all group means are equal.
    
    Args:
        df: Data matrix (proteins × samples)
        group_cols_dict: {"GroupA": ["A1", "A2"], "GroupB": ["B1", "B2"], ...}
        min_valid: Minimum valid values per group
    
    Returns:
        DataFrame with f_stat, p_value, neg_log10_pval, mse
    
    Example:
        groups = {"Control": ["C1", "C2"], "Treatment": ["T1", "T2"]}
        results = perform_anova(df, groups)
    """
    results = []
    
    for protein_id, row in df.iterrows():
        group_data = []
        
        # Extract data for each group
        for group_name, cols in group_cols_dict.items():
            vals = row[cols].dropna()
            if len(vals) < min_valid:
                group_data = None
                break
            group_data.append(vals.values)
        
        # Skip if insufficient data
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
        sse = sum(((vals - grand_mean) ** 2).sum() for vals in group_data)
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
# EFFECT SIZES
# Quantify magnitude of differences independent of sample size
# ============================================================================

def calculate_cohens_d(g1: pd.Series, g2: pd.Series) -> float:
    """
    Cohen's d effect size between two groups.
    Measures standardized difference: d = (mean1 - mean2) / pooled_std
    
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

# ============================================================================
# ERROR METRICS & VALIDATION
# Performance evaluation when ground truth is available
# ============================================================================

def calculate_error_metrics(
    results_df: pd.DataFrame,
    true_fc_dict: Dict[str, float],
    fc_threshold: float = 1.0,
    pval_threshold: float = 0.05,
) -> Dict:
    """
    Calculate confusion matrix and performance metrics.
    
    Args:
        results_df: Results from perform_ttest() with 'regulation' column
        true_fc_dict: True log2FC by protein ID (ground truth)
        fc_threshold: FC threshold used for classification
        pval_threshold: P-value threshold used
    
    Returns:
        Dict with TP, FP, TN, FN, sensitivity, specificity, precision, FPR, FNR
    """
    results_df = results_df.copy()
    
    # Map true fold changes
    results_df["true_log2fc"] = results_df.index.map(
        lambda x: true_fc_dict.get(x, 0.0)
    )
    
    # Classify true regulation
    results_df["true_regulated"] = results_df["true_log2fc"].apply(
        lambda x: abs(x) > fc_threshold
    )
    
    # Observed regulation
    results_df["observed_regulated"] = results_df["regulation"].isin(["up", "down"])
    
    # Confusion matrix
    TP = ((results_df["true_regulated"]) & (results_df["observed_regulated"])).sum()
    FP = ((~results_df["true_regulated"]) & (results_df["observed_regulated"])).sum()
    TN = ((~results_df["true_regulated"]) & (~results_df["observed_regulated"])).sum()
    FN = ((results_df["true_regulated"]) & (~results_df["observed_regulated"])).sum()
    
    # Metrics
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    return {
        "TP": int(TP),
        "FP": int(FP),
        "TN": int(TN),
        "FN": int(FN),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "fpr": fpr,
        "fnr": fnr,
    }

def compute_species_rmse(
    results_df: pd.DataFrame,
    true_fc_dict: Dict[str, float],
    species_mapping: Dict[str, str],
    fc_threshold: float = 1.0,
    pval_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Calculate RMSE and detection rate per species.
    
    Args:
        results_df: Results DataFrame with 'regulation' column
        true_fc_dict: True log2FC by protein ID
        species_mapping: Protein ID → species mapping
        fc_threshold: For classification
        pval_threshold: For classification
    
    Returns:
        DataFrame with per-species metrics (N, Theo FC, RMSE, MAE, Detection)
    """
    species_metrics = []
    
    for species in ["HUMAN", "YEAST", "ECOLI"]:
        # Get proteins for this species
        species_proteins = [
            pid for pid, sp in species_mapping.items() if sp == species
        ]
        
        if not species_proteins:
            continue
        
        # Get results for this species
        species_results = results_df.loc[
            results_df.index.intersection(species_proteins)
        ].copy()
        
        if len(species_results) == 0:
            continue
        
        # Get theoretical FC
        theo_fc = true_fc_dict.get(species_proteins[0], 0.0)
        
        # Calculate error
        species_results["error"] = species_results["log2fc"] - theo_fc
        n_proteins = len(species_results)
        rmse = np.sqrt((species_results["error"] ** 2).mean())
        mae = species_results["error"].abs().mean()
        
        # Detection rate
        if "regulation" not in species_results.columns:
            species_results["regulation"] = species_results.apply(
                lambda row: classify_regulation(
                    row["log2fc"], row["pvalue"], fc_threshold, pval_threshold
                ),
                axis=1,
            )
        
        n_detected = (
            species_results["regulation"].isin(["up", "down"]).sum()
        )
        detection_rate = n_detected / n_proteins if n_proteins > 0 else 0
        
        species_metrics.append({
            "Species": species,
            "N": n_proteins,
            "Theo FC": f"{theo_fc:.2f}",
            "RMSE": f"{rmse:.3f}",
            "MAE": f"{mae:.3f}",
            "Detection": f"{detection_rate:.1%}",
        })
    
    return pd.DataFrame(species_metrics)
