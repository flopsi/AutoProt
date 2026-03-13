import polars as pl
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skbio.stats.distance import permanova, DistanceMatrix
from scipy.spatial.distance import pdist, squareform

def apply_transform(df: pl.DataFrame, cols: list[str], method: str) -> pl.DataFrame:
    if method == "log2":
        return df.with_columns([
            (pl.col(c) + 1.0).log(base=2).alias(c) for c in cols
        ])
    elif method == "glog":
        mat = df.select(cols).to_numpy().astype(np.float64)
        row_vars = np.nanvar(mat, axis=1)
        lam = float(np.nanmedian(row_vars))
        transformed = np.log2(mat + np.sqrt(mat**2 + lam))
        transformed_df = pl.DataFrame(
            {col: transformed[:, i] for i, col in enumerate(cols)}
        )
        non_intensity = [c for c in df.columns if c not in cols]
        return pl.concat([df.select(non_intensity), transformed_df], how="horizontal")
    return df

def compute_pca(df: pl.DataFrame, cols: list[str]) -> tuple:
    mat = df.select(cols).to_numpy().T
    valid_mask = ~np.isnan(mat).any(axis=0)
    mat_clean = mat[:, valid_mask]

    scaler = StandardScaler()
    mat_clean = scaler.fit_transform(mat_clean)

    n_comp = min(3, min(mat_clean.shape))
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(mat_clean)

    return scores, pca.explained_variance_ratio_.tolist()

def compute_permanova(df: pl.DataFrame, cols: list[str], conditions: list[str]) -> dict:
    mat = df.select(cols).to_numpy().T
    valid_mask = ~np.isnan(mat).any(axis=0)
    mat_clean = mat[:, valid_mask]

    dist_condensed = pdist(mat_clean, metric="euclidean")
    dist_square = squareform(dist_condensed)
    dm = DistanceMatrix(dist_square, ids=[str(i) for i in range(len(conditions))])

    # Skbio permanova
    res = permanova(dm, conditions, permutations=999, seed=42)

    f_stat = res["test statistic"]
    n = len(conditions)
    k = len(set(conditions))
    r_squared = (f_stat * (k - 1)) / (f_stat * (k - 1) + (n - k))

    return {
        "pseudo_f": f_stat,
        "p_value": res["p-value"],
        "r_squared": r_squared
    }

def compute_cvs(df: pl.DataFrame, cols: list[str], conditions: list[str]) -> list:
    results = []
    unique_conds = list(set(conditions))

    for cond in unique_conds:
        # Get columns matching this condition
        cond_cols = [c for i, c in enumerate(cols) if conditions[i] == cond]
        if not cond_cols:
            continue

        cv_expr = (
            pl.concat_list(cond_cols).list.eval(
                pl.element().std() / pl.element().mean() * 100
            ).list.first()
        )
        cond_df = df.with_columns(cv_expr.alias("cv")).drop_nulls("cv")
        cv_array = cond_df["cv"].to_list()

        results.append({
            "condition": cond,
            "cv_values": cv_array
        })
    return results
