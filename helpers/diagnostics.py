"""
helpers/diagnostics.py - STATISTICAL DIAGNOSTICS & AUTO-GENERATED COMMENTS

Generate intelligent, actionable comments based on normality and variance diagnostics.
"""

import numpy as np
import pandas as pd


def classify_skewness(skewness: float) -> tuple:
    """
    Classify skewness level and direction.
    
    Returns: (level, direction)
    - level: 'symmetric', 'moderately_skewed', 'highly_skewed'
    - direction: 'symmetric', 'right', 'left'
    """
    abs_skew = abs(skewness)
    
    if abs_skew < 0.5:
        direction = 'symmetric'
        level = 'symmetric'
    elif abs_skew < 1.0:
        direction = 'right' if skewness > 0 else 'left'
        level = 'moderately_skewed'
    else:
        direction = 'right' if skewness > 0 else 'left'
        level = 'highly_skewed'
    
    return level, direction


def classify_kurtosis(kurtosis: float) -> tuple:
    """
    Classify kurtosis level.
    
    Returns: (level, type)
    - level: 'light', 'normal', 'heavy'
    - type: descriptive string
    """
    if kurtosis < -1:
        return 'light', 'light-tailed (uniform-like)'
    elif -1 <= kurtosis <= 1:
        return 'normal', 'normal-like'
    else:
        return 'heavy', 'heavy-tailed (outlier-prone)'


def generate_normality_comment(diag_dict: dict) -> str:
    """
    Generate diagnostic comment about normality status.
    
    Input keys: shapiro_p, dagostino_p, shapiro_w
    """
    shapiro_p = diag_dict.get('shapiro_p', np.nan)
    dagostino_p = diag_dict.get('dagostino_p', np.nan)
    shapiro_w = diag_dict.get('shapiro_w', np.nan)
    
    if np.isnan(shapiro_p):
        return "Insufficient data for normality testing"
    
    shapiro_pass = shapiro_p > 0.05
    dagostino_pass = np.isnan(dagostino_p) or dagostino_p > 0.05
    
    # Determine overall normality
    if not np.isnan(dagostino_p):
        is_normal = shapiro_pass and dagostino_pass
        if is_normal:
            return f"✓ Normal distribution (Shapiro W={shapiro_w:.3f}, p={shapiro_p:.4f}; D'Agostino p={dagostino_p:.4f})"
        else:
            failed_tests = []
            if not shapiro_pass:
                failed_tests.append(f"Shapiro-Wilk (p={shapiro_p:.4f})")
            if not dagostino_pass:
                failed_tests.append(f"D'Agostino (p={dagostino_p:.4f})")
            return f"✗ Non-normal distribution - Failed: {', '.join(failed_tests)}"
    else:
        if shapiro_pass:
            return f"✓ Normal distribution (Shapiro W={shapiro_w:.3f}, p={shapiro_p:.4f})"
        else:
            return f"✗ Non-normal distribution (Shapiro-Wilk p={shapiro_p:.4f})"


def generate_distribution_comment(diag_dict: dict) -> str:
    """
    Generate comment about distribution shape (skewness & kurtosis).
    """
    skewness = diag_dict.get('skewness', np.nan)
    kurtosis = diag_dict.get('kurtosis', np.nan)
    
    if np.isnan(skewness) or np.isnan(kurtosis):
        return "Insufficient data for distribution analysis"
    
    skew_level, skew_direction = classify_skewness(skewness)
    kurt_level, kurt_type = classify_kurtosis(kurtosis)
    
    comments = []
    
    # Skewness comment
    if skew_level == 'symmetric':
        comments.append(f"Symmetric distribution (skewness={skewness:.2f})")
    elif skew_direction == 'right':
        comments.append(f"Right-skewed (skewness={skewness:.2f}, peak left, tail right)")
    else:
        comments.append(f"Left-skewed (skewness={skewness:.2f}, peak right, tail left)")
    
    # Kurtosis comment
    if kurt_level == 'normal':
        comments.append(f"Normal-like tails (kurtosis={kurtosis:.2f})")
    elif kurt_level == 'heavy':
        comments.append(f"Heavy-tailed - prone to outliers (kurtosis={kurtosis:.2f})")
    else:
        comments.append(f"Light-tailed - uniform-like (kurtosis={kurtosis:.2f})")
    
    return " | ".join(comments)


def generate_variance_comment(levene_p: float) -> str:
    """
    Generate comment about variance homogeneity (Levene's test).
    """
    if np.isnan(levene_p):
        return "Levene's test not performed (single sample per condition)"
    
    if levene_p > 0.05:
        return f"✓ Homogeneous variance (Levene p={levene_p:.4f})"
    else:
        return f"✗ Heterogeneous variance (Levene p={levene_p:.4f}) - Consider Welch's tests"


def generate_recommendations(diag_dict: dict, sample_size: int = None) -> list:
    """
    Generate actionable recommendations based on all diagnostics.
    
    Returns list of recommendation strings.
    """
    shapiro_p = diag_dict.get('shapiro_p', np.nan)
    dagostino_p = diag_dict.get('dagostino_p', np.nan)
    skewness = diag_dict.get('skewness', np.nan)
    kurtosis = diag_dict.get('kurtosis', np.nan)
    levene_p = diag_dict.get('levene_p', np.nan)
    
    recommendations = []
    
    # Determine normality
    shapiro_pass = not np.isnan(shapiro_p) and shapiro_p > 0.05
    dagostino_pass = np.isnan(dagostino_p) or dagostino_p > 0.05
    is_normal = shapiro_pass and dagostino_pass
    
    # Determine variance homogeneity
    is_homogeneous = np.isnan(levene_p) or levene_p > 0.05
    
    # Sample size considerations
    if sample_size and sample_size < 8:
        recommendations.append("⚠️ Small sample size (n<8): Non-parametric tests preferable")
    
    # Main recommendation path
    if is_normal and is_homogeneous:
        recommendations.append("✓ READY FOR PARAMETRIC TESTS")
        recommendations.append("  → Use t-test, ANOVA, linear regression")
        recommendations.append("  → Assumptions met for standard statistical methods")
    
    elif is_normal and not is_homogeneous:
        recommendations.append("✓ Data is normal but variance is unequal")
        recommendations.append("  → Use Welch's t-test or Welch's ANOVA (handles unequal variance)")
        recommendations.append("  → Avoid standard t-test/ANOVA")
    
    elif not is_normal:
        # Check skewness for transformation recommendation
        if not np.isnan(skewness) and abs(skewness) > 0.5:
            skew_direction = "right" if skewness > 0 else "left"
            
            if skew_direction == 'right':
                recommendations.append("📊 Data is right-skewed")
                recommendations.append("  → PRIMARY: Apply log2 transformation (standard for proteomics)")
                recommendations.append("  → SECONDARY: Try sqrt or Box-Cox transformation")
                recommendations.append("  → Then retest normality")
            else:
                recommendations.append("📊 Data is left-skewed")
                recommendations.append("  → Apply reflection + log transformation (max-x, then log)")
                recommendations.append("  → Alternative: Box-Cox or Yeo-Johnson transform")
        
        if sample_size and sample_size < 30:
            recommendations.append("  → If transformation doesn't help: Use Mann-Whitney U or Kruskal-Wallis")
        else:
            recommendations.append("  → Consider robust non-parametric tests")
    
    # Outlier/heavy-tail handling
    if not np.isnan(kurtosis) and kurtosis > 1.5:
        recommendations.append("🔴 Heavy tails detected - possible outliers")
        recommendations.append("  → Inspect outliers manually first")
        recommendations.append("  → Consider: Winsorization, robust regression, or trimmed means")
    
    return recommendations


def generate_full_comment(diag_dict: dict, sample_size: int = None) -> dict:
    """
    Generate complete 3-layer comment system.
    
    Returns dict with:
    - diagnostic: What we observe
    - distribution: Shape characteristics
    - variance: Homogeneity status
    - recommendations: List of actionable steps
    - summary: One-line summary
    """
    recommendations = generate_recommendations(diag_dict, sample_size)
    
    return {
        'diagnostic': generate_normality_comment(diag_dict),
        'distribution': generate_distribution_comment(diag_dict),
        'variance': generate_variance_comment(diag_dict.get('levene_p', np.nan)),
        'recommendations': recommendations,
        'summary': recommendations[0] if recommendations else "Insufficient data"
    }


def generate_condition_summary(results_df: pd.DataFrame) -> dict:
    """
    Generate summary-level comments across all samples in a condition.
    
    Input: DataFrame with diagnostic columns for all samples
    Output: Aggregated summary and overall recommendations
    """
    normal_count = (results_df['Shapiro p'] > 0.05).sum()
    homogeneous_count = (results_df['Levene p'] > 0.05).sum()
    total = len(results_df)
    
    normal_pct = (normal_count / total * 100) if total > 0 else 0
    homogeneous_pct = (homogeneous_count / total * 100) if total > 0 else 0
    
    # Aggregate statistics
    mean_skewness = results_df['Skewness'].mean()
    mean_kurtosis = results_df['Kurtosis'].mean()
    
    summary_text = (
        f"Condition Summary: {normal_count}/{total} samples normal ({normal_pct:.0f}%) | "
        f"{homogeneous_count}/{total} homogeneous ({homogeneous_pct:.0f}%) | "
        f"Mean skewness={mean_skewness:.2f}, kurtosis={mean_kurtosis:.2f}"
    )
    
    # Overall recommendation
    if normal_pct > 80 and homogeneous_pct > 80:
        overall_rec = "✓ CONDITION READY for parametric analysis"
    elif normal_pct > 50 and homogeneous_pct > 50:
        overall_rec = "⚠️ MIXED - Consider transformation or paired non-parametric tests"
    else:
        overall_rec = "❌ POOR quality - Recommend log2 transformation or robust methods"
    
    return {
        'summary': summary_text,
        'recommendation': overall_rec,
        'normal_pct': normal_pct,
        'homogeneous_pct': homogeneous_pct,
        'mean_skewness': mean_skewness,
        'mean_kurtosis': mean_kurtosis
    }
