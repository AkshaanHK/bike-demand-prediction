# Bike Demand Prediction

## Overview

This project aims to forecast bike rental demand using machine learning models based on temporal and weather-related features.

The objective is to build a predictive model that helps a bike rental company anticipate demand over a three-month period, supporting operational planning and financial decision-making.

---

## Dataset

The dataset consists of historical bike rental data, including:

- Time-related variables (hour, weekday, month, year)
- Weather conditions (temperature, humidity, wind speed, weather category)
- Binary indicators (working day, holiday)
- Target variable: `cnt` (number of rented bikes)

The training dataset contains approximately 15,000 observations.

---

## Methodology

### 1. Data Preprocessing

- Removed non-informative features (`Id`, `dteday`)
- Handled missing values using context-aware imputation strategies
- Addressed multicollinearity by removing highly correlated variables (`atemp`)
- Converted categorical variables into suitable numerical formats

---

### 2. Feature Engineering

- Created cyclical features for time variables:
  - Hour → `sin(hour)`, `cos(hour)`
  - Month → `sin(month)`, `cos(month)`
  - Weekday → `sin(weekday)`, `cos(weekday)`

- These transformations allow models to capture periodic patterns such as:
  - Daily commuting cycles
  - Seasonal effects

---

### 3. Model Development

Several models were implemented and compared:

- Linear Regression (baseline)
- Lasso Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Regression (SVR)
- Neural Network (MLP)

---

### 4. Evaluation

Models were evaluated using:

- Mean Absolute Error (MAE) on validation/test data
- External leaderboard performance (Kaggle-style evaluation)

---

## Results

| Model                | Test MAE | Leaderboard MAE |
|---------------------|----------|------------------|
| Linear Regression   | ~91      | ~96              |
| Lasso               | ~91      | ~96              |
| Decision Tree       | ~64      | ~71              |
| Random Forest       | ~26      | ~54              |
| Gradient Boosting   | ~24      | ~43              |
| SVR                 | ~43      | ~66              |
| Neural Network      | -        | ~36              |

Key observations:

- Tree-based models significantly outperform linear models
- Gradient Boosting achieves the best test performance
- Neural networks show strong generalization after tuning
- Some models overfit despite strong validation scores

---

## Key Insights

- **Time variables (hour, weekday, month)** are the most predictive features
- Demand shows strong daily and seasonal patterns
- Weather conditions (especially temperature) strongly influence usage
- Working days exhibit different demand patterns compared to weekends


---

## Code

The main implementation is available in:

- `src/`

Key components:

- Data preprocessing pipeline
- Feature engineering functions
- Model training and evaluation
- Prediction pipeline

The notebook (`notebooks/`) provides a complete walkthrough of the analysis.

---

## Reproducibility

- Python 3.x
- Libraries: scikit-learn, pandas, numpy, matplotlib


---

## Applications

This project demonstrates skills relevant to:

- Demand forecasting
- Time series feature engineering
- Machine learning model selection
- Business-oriented data analysis

---

## Author

Akshaan Murugesu
Arijan Seipi
Bastien Olivier Mutzner
Master’s Students in Statistics – University of Geneva
