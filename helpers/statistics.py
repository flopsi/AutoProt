"""
helpers/statistics.py
Statistical tests and metrics for differential expression analysis
Includes t-test, FDR, RMSE, ROC, precision-recall calculations
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from typing import Dict, Tuple
from scipy import stats

def test_normality_shapiro(
    series: pd.Series,
    alpha: float = 0.05,
) -> dict:
    """
    Shapiro–Wilk normality test on a 1D numeric series.

    Returns:
        {
            "statistic": float,
            "p_value": float,
            "is_normal": bool,
            "alpha": float,
            "n": int,
        }
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

# ============================================================================
# T-TEST & FDR CALCULATION
# ============================================================================

def perform_ttest(
    df: pd.DataFrame,
    group1_cols: list,
    group2_cols: list,
    min_valid: int = 2,
) -> pd.DataFrame:
    """
    Perform Welch's t-test on log2-transformed data.
    Compares group1 (reference) vs group2 (treatment).
    
    Uses Limma convention: log2FC = mean(group1) - mean(group2)
    Positive FC = higher in reference group
    
    Args:
        df: Transformed intensity data
        group1_cols: Column names for reference/control
        group2_cols: Column names for treatment
        min_valid: Minimum non-missing values required per group
        
    Returns:
        DataFrame with log2fc, pvalue, fdr, means, n_valid
    """
    results = []
    
    # ---- Loop over each protein ----
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
        
        # Limma convention: log2FC = log2(g1 / g2)
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
    
    # ---- FDR Correction (Benjamini-Hochberg) ----
    pvals = results_df["pvalue"].dropna().sort_values()
    n = len(pvals)
    
    if n > 0:
        ranks = np.arange(1, n + 1)
        fdr_vals = pvals.values * n / ranks
        # Ensure FDR is monotonic (increasing from right to left)
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
# CLASSIFICATION & ERROR RATES
# ============================================================================

def classify_regulation(
    log2fc: float,
    pvalue: float,
    fc_threshold: float = 1.0,
    pval_threshold: float = 0.05,
) -> str:
    """
    Classify protein regulation status.
    
    Args:
        log2fc: log2 fold change
        pvalue: Statistical p-value (or FDR)
        fc_threshold: Absolute log2FC cutoff
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
        true_fc_dict: True log2FC by protein ID
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
    
    # Classify true regulation (using same thresholds)
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


# ============================================================================
# ROC & PRECISION-RECALL CURVES
# ============================================================================

def compute_roc_curve(
    results_df: pd.DataFrame,
    true_fc_dict: Dict[str, float],
    fc_threshold: float = 1.0,
) -> Tuple[list, list, list]:
    """
    Compute ROC curve: false positive rate vs true positive rate.
    Varies p-value threshold from 1.0 to 0.0.
    
    Args:
        results_df: Results from perform_ttest() with pvalue
        true_fc_dict: True log2FC by protein ID
        fc_threshold: FC threshold for true regulation
        
    Returns:
        (fpr_list, tpr_list, threshold_list)
    """
    results_df = results_df.copy()
    results_df["true_regulated"] = results_df.index.map(
        lambda x: abs(true_fc_dict.get(x, 0.0)) > fc_threshold
    )
    
    # Sort by p-value
    results_df = results_df.sort_values("pvalue")
    
    fpr_list = []
    tpr_list = []
    thresholds = []
    
    n_neg = (~results_df["true_regulated"]).sum()
    n_pos = results_df["true_regulated"].sum()
    
    if n_neg == 0 or n_pos == 0:
        return [0, 1], [0, 1], [1, 0]
    
    # Vary threshold across p-values
    for pval_threshold in np.linspace(1, 0, 50):
        results_df["predicted"] = results_df["pvalue"] < pval_threshold
        
        tp = (results_df["true_regulated"] & results_df["predicted"]).sum()
        fp = (~results_df["true_regulated"] & results_df["predicted"]).sum()
        
        tpr = tp / n_pos
        fpr = fp / n_neg
        
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        thresholds.append(pval_threshold)
    
    return fpr_list, tpr_list, thresholds


def compute_precision_recall_curve(
    results_df: pd.DataFrame,
    true_fc_dict: Dict[str, float],
    fc_threshold: float = 1.0,
) -> Tuple[list, list, list]:
    """
    Compute precision-recall curve.
    Varies p-value threshold from 1.0 to 0.0.
    
    Args:
        results_df: Results from perform_ttest()
        true_fc_dict: True log2FC by protein ID
        fc_threshold: FC threshold
        
    Returns:
        (recall_list, precision_list, threshold_list)
    """
    results_df = results_df.copy()
    results_df["true_regulated"] = results_df.index.map(
        lambda x: abs(true_fc_dict.get(x, 0.0)) > fc_threshold
    )
    
    results_df = results_df.sort_values("pvalue")
    
    precision_list = []
    recall_list = []
    thresholds = []
    
    n_pos = results_df["true_regulated"].sum()
    
    if n_pos == 0:
        return [0, 1], [1, 0], [1, 0]
    
    for pval_threshold in np.linspace(1, 0, 50):
        results_df["predicted"] = results_df["pvalue"] < pval_threshold
        
        tp = (results_df["true_regulated"] & results_df["predicted"]).sum()
        fp = (~results_df["true_regulated"] & results_df["predicted"]).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / n_pos
        
        precision_list.append(precision)
        recall_list.append(recall)
        thresholds.append(pval_threshold)
    
    return recall_list, precision_list, thresholds


# ============================================================================
# PER-SPECIES METRICS
# ============================================================================

def compute_species_rmse(
    results_df: pd.DataFrame,
    true_fc_dict: Dict[str, float],
    species_mapping: Dict[str, str],
    fc_threshold: float = 1.0,
    pval_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Calculate RMSE and other metrics per species.
    
    Args:
        results_df: Results DataFrame with 'regulation' column
        true_fc_dict: True log2FC by protein ID
        species_mapping: Protein ID → species mapping
        fc_threshold: For classification
        pval_threshold: For classification
        
    Returns:
        DataFrame with per-species metrics
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
        bias = species_results["error"].mean()
        
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
