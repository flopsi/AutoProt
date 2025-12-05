# helpers/cleaning.py

import pandas as pd
from typing import List


def drop_proteins_with_invalid_intensities(
    df: pd.DataFrame,
    intensity_cols: List[str],
    drop_value: float | None = 1.0,
    drop_nan: bool = True,
) -> pd.DataFrame:
    """
    Drop rows (proteins) that contain at least one invalid intensity
    in the given columns.

    Invalid means:
      - NaN (if drop_nan=True)
      - equal to drop_value (if drop_value is not None, default 1.0)

    So if ANY intensity column for a protein is NaN or 1.0, that row is removed.
    """
    if not intensity_cols:
        return df

    sub = df[intensity_cols]

    invalid = False
    if drop_nan:
        invalid = sub.isna()
    if drop_value is not None:
        invalid = invalid | (sub == drop_value) if isinstance(invalid, pd.DataFrame) else (sub == drop_value)

    # rows where any intensity is invalid
    rows_to_drop = invalid.any(axis=1)

    return df.loc[~rows_to_drop].copy()
