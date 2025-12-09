# helpers/diagnostics.py

def generate_sample_comments(diag_dict):
    """
    Generate 3-layer comments from diagnostic results.
    
    Input: {
        'shapiro_w': 0.95,
        'shapiro_p': 0.001,
        'dagostino_p': 0.0001,
        'skewness': 0.8,
        'kurtosis': 1.2,
        'levene_p': 0.12,
        'n': 50
    }
    
    Output: {
        'diagnostic': "Data is right-skewed with moderate heavy tails",
        'interpretation': "Non-normal distribution violates parametric assumptions",
        'recommendation': "Apply log2 transformation or use non-parametric tests"
    }
    """
    
    # Layer 1: Diagnose normality
    is_normal = (diag_dict['shapiro_p'] > 0.05) and (diag_dict['dagostino_p'] > 0.05)
    
    # Layer 2: Diagnose distribution shape
    skew_level = classify_skewness(diag_dict['skewness'])  # 'symmetric', 'right', 'left'
    kurtosis_level = classify_kurtosis(diag_dict['kurtosis'])  # 'normal', 'heavy', 'light'
    
    # Layer 3: Diagnose variance homogeneity
    is_homogeneous = diag_dict['levene_p'] > 0.05
    
    # Generate comments based on combinations
    return {
        'diagnostic': f"...",  # Describe what we see
        'interpretation': f"...",  # Why it matters
        'recommendation': f"..."  # What to do
    }
