# order-signal

**Turning messy e-commerce transaction logs into ML-ready features.**

A hands-on data engineering project covering the full preprocessing lifecycle — from raw Excel exports with missing values and outliers to a validated, schema-enforced feature store. Built as part of the DecodeLabs Data Science industrial training program (Batch 2026).

---

## What this is

Most ML tutorials hand you a clean dataset. Real work doesn't. This project takes a 1,200-row e-commerce orders table that has missing values, price outliers, raw categoricals, and no engineered features — and runs it through a structured three-stage pipeline until it's something a model can actually learn from.

The pipeline is split into modules (`src/`) so each piece can be tested, reused, or swapped out independently. The notebook in `notebooks/` runs them end to end.

---

## Dataset

| Property | Detail |
|---|---|
| Rows | 1,200 |
| Original columns | 14 |
| Date range | Jan 2023 – Jun 2025 |
| Domain | E-commerce orders |

Columns: `OrderID`, `Date`, `CustomerID`, `Product`, `Quantity`, `UnitPrice`, `ShippingAddress`, `PaymentMethod`, `OrderStatus`, `TrackingNumber`, `ItemsInCart`, `CouponCode`, `ReferralSource`, `TotalPrice`

> The raw dataset is not tracked in this repo. Drop your file at `data/raw/Dataset_for_Data_Analytics.xlsx` before running.

---

## Pipeline overview

```
PHASE 1 — INPUT                 PHASE 2 — PROCESS               PHASE 3 — OUTPUT
────────────────────            ────────────────────────        ─────────────────────────
Audit missing values        →   Vectorized feature creation →   Pandera schema validation
KNN imputation (CouponCode)     One-Hot Encoding                CSV + Parquet export
IQR outlier detection           Collinearity eradication        Feature store ready
Winsorization (np.clip)         Correlation heatmap
```

---

## Key decisions

### Missing data — why KNN and not median fill

`CouponCode` had 25.75% missing values. Median imputation works fine for numeric columns with moderate missingness, but CouponCode is categorical and the coupon someone uses tends to correlate with their referral source and what they bought. KNN can pick up on those relationships. A global fill would just stamp every missing row with the most common coupon regardless of context.

### Outlier treatment — why Winsorize instead of drop

The standard reflex is to drop rows with extreme values. With 1,200 records that's a meaningful chunk of training data, and the other 13 columns in those rows are perfectly valid. Capping at the IQR boundary with `np.clip()` keeps the row count at 1,200 throughout.

### Feature engineering

| Feature | Logic |
|---|---|
| `RevenuePerItem` | `TotalPrice / Quantity` — average value per unit ordered |
| `CartUtilizationRate` | `Quantity / ItemsInCart` — purchase intent signal |
| `IsHighValueOrder` | Binary flag for orders in the top revenue quartile |
| `OrderMonth` | Extracted from `Date` — seasonal coupon patterns visible here |
| `HasCoupon` | Binary flag for whether a discount code was applied |

All features computed with vectorized Pandas operations.

### Collinearity

After One-Hot Encoding, the absolute Pearson correlation matrix is checked for pairs above 0.80. When a pair is found, the feature with the lower correlation to `TotalPrice` gets dropped. No arbitrary deletions — every removal is justified by the target relationship.

---

## Outputs

After the pipeline runs:

```
outputs/
├── boxplots_before.png       ← outlier distribution pre-Winsorization
├── boxplots_after.png        ← distribution post-Winsorization
└── correlation_heatmap.png   ← full feature correlation matrix

data/processed/
├── cleaned_dataset.csv
└── feature_store.parquet
```

---

## Getting started

```bash
git clone https://github.com/YOUR_USERNAME/order-signal.git
cd order-signal

pip install -r requirements.txt

# drop your dataset at data/raw/Dataset_for_Data_Analytics.xlsx
# then run:
python notebooks/pipeline.py
```

---

## Project structure

```
order-signal/
├── data/
│   ├── raw/               ← not tracked (add your .xlsx here)
│   └── processed/         ← not tracked (pipeline outputs)
├── notebooks/
│   └── pipeline.py        ← runs the full pipeline end to end
├── src/
│   ├── imputation.py      ← missing value logic
│   ├── outliers.py        ← IQR detection + Winsorization
│   ├── features.py        ← feature engineering + encoding + collinearity
│   └── validation.py      ← Pandera schema contracts
├── outputs/               ← saved plots
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Stack

Python 3.11 · Pandas · NumPy · scikit-learn · Pandera · Matplotlib · Seaborn

---

## License

MIT