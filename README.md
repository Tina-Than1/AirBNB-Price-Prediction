# 🏠 Predicting Airbnb Listing Prices in Brisbane, Australia

A supervised regression project that predicts nightly Airbnb listing prices in Brisbane using host, property, location, and review data — built as part of **BUSA3020 (Predictive Analytics)** at Macquarie University. *(Result: High Distinction)*

---

## 📌 Overview

This project frames Airbnb pricing as a **supervised regression problem**: given a listing's physical characteristics, host attributes, location, and review history, predict its nightly price (AUD). The pipeline covers the full ML workflow — from raw, messy data to a tuned, cross-validated model — and mirrors a real Kaggle-style competition, evaluated on **Mean Absolute Percentage Error (MAPE)**.

```
Task 1: EDA & Feature Selection  →  Task 2: Cleaning & Feature Engineering  →  Task 3: Modelling & Tuning
```

## 🎯 Problem Statement

Given ~65 raw features per listing (property details, host attributes, location, review activity), predict the nightly price of unseen Brisbane listings. Accurate pricing supports:

- **Hosts** — benchmarking and dynamic pricing
- **Investors** — assessing short-term rental potential
- **Platforms** — smart pricing tools & anomaly detection
- **Travellers** — a fairer, better-calibrated market
- **Policymakers** — monitoring tourism/housing impacts

**Evaluation metric:** Mean Absolute Percentage Error (MAPE)

$$\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\%$$

MAPE was chosen for being scale-independent and business-intuitive, while acknowledging its limitations (undefined near zero-price listings, asymmetric penalty on under- vs. over-prediction).

## 📊 Dataset

- **3,735** training listings, **1,601** test listings, **65** raw features
- Target: `price` (nightly rate, AUD) — heavily right-skewed (mean ≈ $230, median ≈ $193, max $5,000); log-transformed for modelling
- 33 columns contained missing values, ranging from isolated gaps (e.g. `bathrooms`, `bedrooms`) to near-total missingness (`host_neighbourhood`, ~96%)

## 🔧 Approach

### 1. Exploratory Data Analysis
- Univariate analysis of price, room type, availability, and location distributions
- 23 features shortlisted based on correlation strength and domain relevance (e.g. `accommodates`, `bedrooms`, `bathrooms`, `room_type`, `neighbourhood_cleansed`)

### 2. Data Cleaning & Feature Engineering
- Parsed currency/percentage strings (`price`, `host_response_rate`, `bathrooms_text`) into clean numerics
- Engineered new features such as
  - `amenities_count` — number of amenities listed
  - `days_since_last_review` — recency of guest engagement
  - `host_tenure_days` — host experience on the platform
- Missing values imputed using **training-set-only** statistics (median for numeric, mode for categorical) to prevent data leakage
- High-cardinality categoricals capped at top 5 categories + "other", then one-hot encoded
- Skewed features log-transformed where appropriate, and scaled with `StandardScaler` (fit on train only)

### 3. Modelling & Evaluation
Three models were trained and compared using **5-fold cross-validation**, with MAPE (back-transformed from log-price) as the selection metric:

| Model | Type | CV MAPE |
|---|---|---|
| Ridge Regression | Regularised linear (L2) | 24.38% |
| Random Forest | Bagging ensemble | 12.90% |
| **XGBoost** | Gradient boosting | **10.53%** ✅ |

**XGBoost** (`learning_rate=0.05`, `max_depth=6`, `n_estimators=600`, `subsample=0.8`) was selected as the final model. Error analysis showed stable accuracy (~1.3–1.7% MAPE) across most price bands, with higher error (~2.7%) concentrated in the sparse $1,000+ listing segment — attributed to limited training data at the top of the price range.

## 🛠️ Tech Stack

- **Python** — pandas, NumPy
- **scikit-learn** — Ridge, Random Forest, preprocessing, cross-validation
- **XGBoost** — gradient boosting regressor
- **Matplotlib** — visualisation

## 📁 Repository Structure

```
1. AirBnb_Price_Prediction.ipynb   # Full analysis: EDA → feature engineering → modelling
2. README.md

## 🔑 Key Takeaways

- Non-linear models (Random Forest, XGBoost) substantially outperformed linear regression, confirming meaningful feature interactions in pricing behaviour
- Careful **leakage prevention** (fitting imputation/scaling/encoding on train only) was central to reliable evaluation
- Listing size/capacity (`accommodates`, `bedrooms`, `bathrooms`) were the strongest predictors; location and amenities added incremental signal
- Model performance degraded at the high end of the price range — a direct consequence of limited high-price training examples

## 🎓 Course Context

Completed as the final assignment for **BUSA3020: Predictive Analytics**, Macquarie University — **High Distinction**.

