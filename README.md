# Bike Sharing Demand Forecasting

**MSc Statistics – Machine Learning Competition (Kaggle, Fall 2024)**  
Team ABA: Arijan Seipi, Bastien Olivier Mutzner, Akshaan Murugesu

---

## Overview

Prediction of hourly bike rental counts for a Geneva-based bike-sharing company ("Mule"), using a dataset with temporal, meteorological and economic features. Goal: provide 3-month sales forecasts and business insights. Evaluated with **Mean Absolute Error (MAE)** on Kaggle.

---

## Dataset

- **15,211 hourly observations**, 15 features (temporal + weather + target `cnt`)
- Key features: `hr`, `temp`, `season`, `yr`, `workingday`, `weathersit`, `hum`, `windspeed`
- Target: `cnt` — total bike rentals (casual + registered)

---

## Data Preprocessing

- **Missing value imputation**: forward-fill for temporal/categorical columns, interpolation (neighbor average) for numerical columns (`temp`, `atemp`, `hum`, `windspeed`)
- **Feature encoding**: ordinal encoding for categorical variables (`season`, `weathersit`, etc.)
- **Dropped**: `Id`, `dteday`, `atemp` (near-perfect correlation with `temp`, r = 0.99)

---

## Exploratory Data Analysis

Key findings:
- **Rush hours** (7–9h, 16–18h) drive demand on weekdays; weekends show a smooth afternoon peak
- **Summer** has the highest rentals; **Winter** the lowest — temperature is a primary driver
- **Bike usage grew significantly** from 2011 to 2012, suggesting strong business growth trend
- Distribution of `cnt` is heavily right-skewed

---

## Feature Engineering

Cyclical sine/cosine encoding for periodic temporal features to capture their circular nature:

```python
train['hour_sin']    = np.sin(2 * np.pi * train['hr'] / 24)
train['hour_cos']    = np.cos(2 * np.pi * train['hr'] / 24)
train['weekday_sin'] = np.sin(2 * np.pi * train['weekday'] / 7)
train['weekday_cos'] = np.cos(2 * np.pi * train['weekday'] / 7)
train['month_sin']   = np.sin(2 * np.pi * train['mnth'] / 12)
train['month_cos']   = np.cos(2 * np.pi * train['mnth'] / 12)
```
---

## Models & Results

| Model | Train MAE | Test MAE | Kaggle MAE |
|---|---|---|---|
| Linear Regression | 91.28 | 91.08 | — |
| Lasso Regression | 91.23 | 91.03 | — |
| Decision Tree (depth=5) | 60.76 | 63.84 | — |
| Random Forest | 9.74 | 25.91 | 54.37 |
| Gradient Boosting | 12.54 | 23.99 | 43.32 |
| SVR (RBF kernel) | — | 43.22 | 65.85 |
| Deep Neural Network | — | — | **35.75**|

**Best model**: Deep Neural Network with LeakyReLU, Dropout, BatchNorm and early stopping.

### Model highlights

**Gradient Boosting** — best test-set MAE among classical models (23.99). Hyperparameters tuned with 5-fold CV grid search (`n_estimators`, `learning_rate`, `max_depth`). 1-SE rule applied to balance performance vs. generalization.

**Deep Neural Network** — 3 hidden layers (256 → 128 → 64), LeakyReLU activation, Dropout (0.2), BatchNormalization, EarlyStopping. Preprocessing pipeline: OneHotEncoding for categoricals + StandardScaler for numericals.

---

## Business Insights for Mule

- **Time of day** is the most important predictor (hour_cos, hour_sin dominate feature importance across all tree-based models)
- **Temperature** is the second most influential factor — plan fleet capacity seasonally
- **Weekday vs. weekend** usage patterns differ fundamentally — tailor fleet distribution accordingly
- Demand is projected to **grow significantly** in 2012 vs. 2011, even in off-season months

---
