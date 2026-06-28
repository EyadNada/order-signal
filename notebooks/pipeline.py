# -----------------------------------------------------------
# E-Commerce Order Data — EDA & Feature Engineering Pipeline
# -----------------------------------------------------------
# Run this top to bottom. Each section is a stage in the pipeline.
# Outputs land in data/processed/ and outputs/


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.imputation import check_missing, impute_coupon_code
from src.outliers import audit_outliers, winsorize
from src.features import engineer_features, encode_categoricals, drop_collinear_features
from src.validation import validate


# ── 0. Load ──────────────────────────────────────────────────────────────────

df = pd.read_excel("data/raw/Dataset_for_Data_Analytics.xlsx")
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
df.head()


# ── 1. Initial audit ─────────────────────────────────────────────────────────

print(df.info())
print("\n", df.describe())


# ── 2. Missing values ────────────────────────────────────────────────────────

print(check_missing(df))

# CouponCode: 25.75% missing → KNN
df = impute_coupon_code(df, n_neighbors=5)
print("\nAfter imputation:")
print(check_missing(df))  # should be empty


# ── 3. Outlier audit + Winsorization ─────────────────────────────────────────

print(audit_outliers(df))

# Visualise before capping
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, col in zip(axes, ["UnitPrice", "TotalPrice", "Quantity", "ItemsInCart"]):
    df.boxplot(column=col, ax=ax)
    ax.set_title(col)
plt.suptitle("Before Winsorization")
plt.tight_layout()
plt.savefig("outputs/boxplots_before.png", dpi=120)
plt.show()

df = winsorize(df)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, col in zip(axes, ["UnitPrice", "TotalPrice", "Quantity", "ItemsInCart"]):
    df.boxplot(column=col, ax=ax)
    ax.set_title(col)
plt.suptitle("After Winsorization")
plt.tight_layout()
plt.savefig("outputs/boxplots_after.png", dpi=120)
plt.show()


# ── 4. Feature engineering ───────────────────────────────────────────────────

df = engineer_features(df)
print("\nNew columns added:")
print(df[["RevenuePerItem", "CartUtilizationRate", "IsHighValueOrder", "OrderMonth", "HasCoupon"]].head())


# ── 5. Encode categoricals ───────────────────────────────────────────────────

df = encode_categoricals(df)
print(f"\nAfter OHE: {df.shape[1]} columns")


# ── 6. Correlation heatmap ───────────────────────────────────────────────────

numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, linewidths=0.3)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=120)
plt.show()


# ── 7. Drop collinear features ───────────────────────────────────────────────

df = drop_collinear_features(df, target_col="TotalPrice", threshold=0.80)
print(f"\nAfter collinearity removal: {df.shape[1]} columns")


# ── 8. Validate ──────────────────────────────────────────────────────────────

df = validate(df)


# ── 9. Export ────────────────────────────────────────────────────────────────

df.to_csv("data/processed/cleaned_dataset.csv", index=False)
df.to_parquet("data/processed/feature_store.parquet", index=False)

print(f"\nDone. Final shape: {df.shape}")
print("Exported to data/processed/")