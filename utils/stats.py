# utils/stats.py
import math
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import shapiro, pearsonr
from sklearn.decomposition import PCA

def log2_transform(values: List[float]) -> List[float]:
    return [math.log2(v) if v > 0 else math.log2(1e-6) for v in values]

def compute_cv(values: List[float]) -> float:
    arr = np.array(values)
    mean = arr.mean()
    std = arr.std(ddof=1)
    return (std / mean) if mean != 0 else float('inf')

def missing_fraction(replicates: Dict[str, float]) -> float:
    # treat NaN as missing
    total = len(replicates)
    missing = sum(1 for v in replicates.values() if v is None or (isinstance(v, float) and np.isnan(v)))
    return missing / total if total > 0 else 0.0

def normality_test(values: List[float]) -> Tuple[float, float]:
    # Shapiro-Wilk test; small sample size caveat
    w, p = shapiro(np.array(values))
    return w, p

def quartiles(values: List[float]) -> Tuple[float, float, float]:
    q1 = np.percentile(values, 25)
    q2 = np.percentile(values, 50)
    q3 = np.percentile(values, 75)
    return q1, q2, q3

def perform_pca(df: pd.DataFrame, n_components: int = 2) -> Tuple[np.ndarray, PCA]:
    # df: rows=proteins, cols=replicates
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df.values)
    return components, pca

def t_test_between_groups(group1: List[float], group2: List[float]) -> float:
    # Placeholder for two-sample t-test statistic (p-value calculation)
    # For robust use, replace with scipy.stats.ttest_ind as needed.
    from scipy.stats import ttest_ind
    t_stat, pval = ttest_ind(group1, group2, equal_var=False, nan_policy='omit')
    return pval

def prepare_dataframe_from_proteins(proteins: List[dict], replicate_names: List[str]) -> pd.DataFrame:
    # proteins: list of dict containing 'gene', 'replicates' (dict)
    rows = []
    for p in proteins:
        row = {rep: p['replicates'].get(rep, None) for rep in replicate_names}
        row.update({'gene': p['gene'], 'id': p['id']})
        rows.append(row)
    return pd.DataFrame(rows).set_index('id')
