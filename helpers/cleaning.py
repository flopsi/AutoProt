# helpers/cleaning.py

import pandas as pd
from typing import List


def drop_invalid_intensity_rows(
    df: pd.DataFrame,
    intensity_cols: List[str],
    drop_value: float = 1.0,
) -> pd.DataFrame:
    """
    Drop rows where ALL selected intensity columns are:
    - NaN, or
    - exactly drop_value (default 1.0).

    Rows with at least one valid intensity are kept.
    """
    if not intensity_cols:
        return df

    sub = df[intensity_cols]

    # True where value is NaN or == drop_value
    invalid_mask = sub.isna() | (sub == drop_value)

    # Rows where all intensities are invalid
    rows_to_drop = invalid_mask.all(axis=1)

    return df.loc[~rows_to_drop].copy()
