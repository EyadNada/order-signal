import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder


def check_missing(df):
    """Quick summary of what's missing and by how much."""
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    return report[report["missing_count"] > 0]


def impute_coupon_code(df, n_neighbors=5):
    """
    CouponCode is 25.75% missing — past the 20% threshold where KNN
    makes more sense than a simple global fill. The coupon someone uses
    tends to correlate with how they found the site and what they bought,
    so KNN can actually pick up on those patterns.

    Steps:
      1. Ordinal-encode CouponCode so KNN has numbers to work with
      2. Run KNNImputer
      3. Round and inverse-transform back to the original labels
    """
    df = df.copy()

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    df["_coupon_enc"] = encoder.fit_transform(df[["CouponCode"]])
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df["_coupon_enc"] = imputer.fit_transform(df[["_coupon_enc"]])
    df["CouponCode"] = encoder.inverse_transform(
        df[["_coupon_enc"]].round().astype(int)
    )
    df.drop(columns=["_coupon_enc"], inplace=True)

    return df