# Bike Sharing Demand Forecasting  
**Time Series Modeling and Machine Learning for Demand Prediction**

Prediction of hourly bike rental demand using temporal, weather, and behavioral features. Project developed as part of a Kaggle competition (MSc Statistics, University of Geneva).

---

## Highlights

- 15,211 hourly observations with temporal and weather features
- Feature engineering using cyclical encoding (sin/cos transformations)
- Compared multiple model families (linear, tree-based, kernel, deep learning)
- Best performance achieved with **Deep Neural Network (MAE: 35.75 on Kaggle)**
- End-to-end pipeline: preprocessing → feature engineering → modeling → evaluation

---

## Project Overview

This project focuses on forecasting short-term bike rental demand for a bike-sharing system using historical usage and environmental data.

Key objectives:
- Predict hourly demand with high accuracy
- Capture temporal patterns and seasonality
- Extract actionable business insights

---

## Dataset

- **15,211 hourly observations**, 15 features
- Target: `cnt` (total rentals)

**Key features:**
- Temporal: `hr`, `weekday`, `mnth`, `yr`, `workingday`
- Weather: `temp`, `hum`, `windspeed`, `weathersit`

---

## Data Preprocessing

- Missing values:
  - Forward-fill for temporal/categorical variables
  - Interpolation for numerical variables

- Feature handling:
  - Ordinal encoding for categorical variables
  - Dropped:
    - `Id`, `dteday` (non-informative)
    - `atemp` (high correlation with `temp`, r ≈ 0.99)

---

## Exploratory Data Analysis

- Strong **hourly seasonality**:
  - Weekdays: peaks at commuting hours
  - Weekends: smoother afternoon demand

- **Temperature is a key driver** of demand

- Clear **growth trend** from 2011 to 2012

- Target variable is **right-skewed**

---

## Feature Engineering

Cyclical encoding of periodic variables:

```python
train['hour_sin']    = np.sin(2 * np.pi * train['hr'] / 24)
train['hour_cos']    = np.cos(2 * np.pi * train['hr'] / 24)
train['weekday_sin'] = np.sin(2 * np.pi * train['weekday'] / 7)
train['weekday_cos'] = np.cos(2 * np.pi * train['weekday'] / 7)
train['month_sin']   = np.sin(2 * np.pi * train['mnth'] / 12)
train['month_cos']   = np.cos(2 * np.pi * train['mnth'] / 12)
```

---

## Model Evaluation

### Local Evaluation (Train/Test)

| Model | Train MAE | Test MAE |
|------|----------:|---------:|
| Linear Regression | 91.28 | 91.08 |
| Lasso Regression | 91.23 | 91.03 |
| Decision Tree (depth=5) | 60.76 | 63.84 |
| Random Forest | 9.74 | 25.91 |
| Gradient Boosting | 12.54 | 23.99 |
| SVR (RBF kernel) | — | 43.22 |

### Kaggle Evaluation

| Model | Kaggle MAE |
|------|-----------:|
| Random Forest | 54.37 |
| Gradient Boosting | 43.32 |
| SVR | 65.85 |
| Deep Neural Network | **35.75** |

---

## Model Highlights

**Gradient Boosting**
- Best test-set MAE among classical models (23.99)
- Hyperparameter tuning via 5-fold cross-validation
- Balanced bias-variance tradeoff

**Deep Neural Network**
- Architecture: 256 → 128 → 64
- LeakyReLU activations
- Dropout (0.2) + Batch Normalization
- Early stopping

---

## Technical Contributions

- Built a full machine learning pipeline:
  - Data preprocessing
  - Feature engineering
  - Model training and evaluation

- Compared multiple model families
- Applied cross-validation and regularization techniques
- Translated model outputs into business insights

---

## Repository Structure

```text
├── bike_sharing_forecast.ipynb     # Full pipeline: preprocessing → modeling → evaluation
├── MachineLearning_Report_Team_ABA.pdf              # Final report
├── README.md
```

---

## Tech Stack

- Python
- scikit-learn
- TensorFlow / Keras
- pandas, numpy
- matplotlib, seaborn

---

## Business Insights

- **Time of day** is the strongest predictor
- **Temperature** drives seasonal demand
- **Weekday vs weekend patterns differ structurally**
- Evidence of **strong demand growth**

---

## Why This Project Matters

This project demonstrates the ability to:

- Model structured time series data
- Perform feature engineering for periodic signals
- Compare and tune machine learning models
- Connect predictive modeling with real-world decisions

---

## Authors

- Akshaan Murugesu
- Arijan Seipi
- Bastien Olivier Mutzner

MSc Statistics — University of Geneva
