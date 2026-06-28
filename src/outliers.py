import pandas as pd
import numpy as np


NUMERIC_COLS = ["UnitPrice", "TotalPrice", "Quantity", "ItemsInCart"]


def get_iqr_bounds(series):
    """Returns the lower and upper IQR fence for a given series."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper


def audit_outliers(df, cols=NUMERIC_COLS):
    """
    Prints a quick count of how many outliers exist per column
    before we do anything about them. Good to run first so you
    know what you're dealing with.
    """
    report = {}
    for col in cols:
        lower, upper = get_iqr_bounds(df[col])
        n_out = df[(df[col] < lower) | (df[col] > upper)].shape[0]
        report[col] = {
            "outlier_count": n_out,
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2),
        }
    return pd.DataFrame(report).T


def winsorize(df, cols=NUMERIC_COLS):
    """
    Cap outliers at the IQR boundary instead of dropping the rows.
    With only 1200 records, losing entire rows over one extreme value
    in one column isn't worth it — especially when the other features
    in that row are perfectly fine.

    np.clip does this cleanly without any loops.
    """
    df = df.copy()
    for col in cols:
        lower, upper = get_iqr_bounds(df[col])
        df[col] = np.clip(df[col], lower, upper)
    return df