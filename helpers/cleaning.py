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
    - NaN, OR
    - equal to drop_value (default 1.0).

    Rows that have at least one valid value are kept.
    """
    if not intensity_cols:
        return df

    sub = df[intensity_cols]

    # Condition: for each row, all values are NaN or == drop_value
    mask_all_invalid = sub.isna() | (sub == drop_value)
    rows_to_drop = mask_all_invalid.all(axis=1)

    return df.loc[~rows_to_drop].copy()
