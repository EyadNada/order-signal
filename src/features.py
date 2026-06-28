import pandas as pd
import numpy as np


CATEGORICAL_COLS = ["Product", "PaymentMethod", "OrderStatus", "ReferralSource", "CouponCode"]


def engineer_features(df):
    """
    All new columns derived from existing ones. Everything here is
    vectorized — no row-by-row loops, just pandas operating on whole
    columns at once the way it's supposed to.
    """
    df = df.copy()

    # how much each unit in an order is actually worth
    df["RevenuePerItem"] = df["TotalPrice"] / df["Quantity"]

    # what fraction of the cart they actually bought
    # values close to 1 = they came to buy, not just browse
    df["CartUtilizationRate"] = df["Quantity"] / df["ItemsInCart"]

    # flag orders in the top revenue quartile
    threshold = df["TotalPrice"].quantile(0.75)
    df["IsHighValueOrder"] = (df["TotalPrice"] >= threshold).astype(int)

    # month extracted from the order date — captures seasonal patterns
    # (WINTER15 coupon usage clusters in Q4, for example)
    df["OrderMonth"] = df["Date"].dt.month

    # simple binary: did this order use a coupon or not
    df["HasCoupon"] = df["CouponCode"].notna().astype(int)

    return df


def encode_categoricals(df, cols=CATEGORICAL_COLS):
    """
    One-hot encode all nominal columns. drop_first=True drops one dummy
    per group to avoid feeding the model a perfectly collinear column
    (the dummy variable trap).
    """
    df = pd.get_dummies(df, columns=cols, drop_first=True)
    return df


def drop_collinear_features(df, target_col="TotalPrice", threshold=0.80):
    """
    Finds pairs of numeric features with correlation above the threshold
    and drops whichever one has the weaker relationship with the target.

    Collinear features don't add new information — they just make
    the feature matrix harder to invert and inflate coefficient variance.
    """
    df = df.copy()
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr().abs()

    # only look at the upper triangle to avoid duplicates
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        for row in upper.index:
            if upper.loc[row, col] > threshold:
                # keep whichever has higher correlation with the target
                corr_col = abs(numeric[col].corr(df[target_col]))
                corr_row = abs(numeric[row].corr(df[target_col]))
                drop_candidate = col if corr_col < corr_row else row
                to_drop.add(drop_candidate)
                print(f"Dropping '{drop_candidate}' (corr with '{target_col}' was lower in pair [{col}, {row}])")

    df.drop(columns=list(to_drop), inplace=True)
    return df