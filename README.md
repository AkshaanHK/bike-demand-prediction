# Bike Sharing Demand Prediction

Predicting hourly bike rental counts using a dataset from a bike-sharing company — covering weather, temporal, and calendar features. Built as part of a Kaggle competition for the Machine Learning course (MSc Statistics, University of Geneva, Fall 2024).

**Best Kaggle leaderboard MAE: 35.75** (Deep Neural Network)

---

## Project Structure

```
├── bike_demand_notebook.ipynb      # Full pipeline: EDA → preprocessing → modelling → evaluation
├── MachineLearning_Report_Team_ABA.pdf   # Full written report
├── BikeSharing_Report.pdf          # Synthesized report
├── README.md
```

---

## Dataset

15,211 hourly observations with 15 features:

| Type | Features |
|------|----------|
| Temporal | `season`, `yr`, `mnth`, `hr`, `weekday`, `holiday`, `workingday` |
| Weather | `weathersit`, `temp`, `atemp`, `hum`, `windspeed` |
| Target | `cnt` (total hourly rental count) |

---

## Preprocessing & Feature Engineering

- **Missing value imputation**: forward-fill for categorical/temporal columns; interpolation (average of nearest non-NA neighbours) for numerical columns.
- **Dropped**: `Id`, `dteday` (redundant), `atemp` (corr = 0.99 with `temp` — removes multicollinearity).
- **Cyclic encoding**: `hr`, `weekday`, and `mnth` transformed with sine/cosine to preserve their circular structure (e.g. hour 23 is adjacent to hour 0).
- **Encoding**: ordinal for `season`, `weathersit`; binary for `holiday`, `workingday`, `yr`.

---

## Models & Results

| Model | Test MAE | Kaggle MAE |
|-------|----------|------------|
| Linear Regression | 91.09 | — |
| Lasso Regression | 91.31 | — |
| Decision Tree (depth=5) | 63.76 | — |
| Random Forest | 25.40 | 54.37 |
| Gradient Boosting | 24.65 | 43.32 |
| SVR (RBF kernel) | 43.19 | 65.85 |
| **Deep Neural Network** | **24.39** | **35.75** |

Key modelling choices:

- **Lasso** over Ridge: similar predictive performance, sparser and more interpretable coefficients.
- **Random Forest**: grid search over `n_estimators`, `max_depth`, `max_features` (heuristic m ≈ p/3). Best: 300 trees, depth 18, 5 features.
- **Gradient Boosting**: two-stage grid search with the 1-SE rule to favour generalisation. Best: 1000 estimators, lr = 0.1, depth 5.
- **DNN**: 3 hidden layers (256 → 128 → 64), LeakyReLU, Batch Normalisation, Dropout (0.2), Early Stopping + ReduceLROnPlateau. Architecture selected via 5-fold CV.

---

## Key Insights

- **Hour of day** is the strongest predictor: rush-hour peaks at 7:00–9:00 and 16:00–18:00 on workdays; smooth afternoon curve on weekends.
- **Temperature** is strongly positively correlated with rentals (r = 0.44) — seasonal demand planning is essential.
- **Weekday vs weekend** usage patterns are structurally different, requiring distinct operational strategies.
- Demand grew significantly from 2011 to 2012, with the forecast window (Fall/Winter 2012) projecting higher counts than the equivalent period in 2011.

---

## Stack

`Python` · `scikit-learn` · `TensorFlow / Keras` · `pandas` · `numpy` · `matplotlib`

---

## Authors

Arijan Seipi · Bastien Olivier Mutzner · Akshaan Murugesu
MSc Statistics — University of Geneva

