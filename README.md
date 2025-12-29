# Medical Insurance Charges Prediction (Regression)

End-to-end regression workflow on the classic **medical insurance cost** dataset (`insurance.csv`).  
The notebook covers **EDA**, **outlier handling**, **feature preprocessing with scikit-learn pipelines**, and **model selection** across several regressors (including ensembles and boosting).

---

## Project Goals

- Understand which factors drive insurance charges (EDA + visualizations)
- Build a robust preprocessing pipeline for mixed numerical + categorical features
- Train and compare multiple regression models
- Tune key models using `GridSearchCV`
- Evaluate performance using MSE / RMSE (and extendable to MAE/R²)

---

## Dataset

**File:** `data/insurance.csv`

Typical columns in this dataset:
- Numerical: `age`, `bmi`, `children`
- Categorical: `sex`, `smoker`, `region`
- Target: `charges`

> Note: The notebook applies a **log transform** to the target:  
`charges = log1p(charges)` to reduce skew.  
To convert predictions back to the original scale, use: `expm1(pred)`.

---

## What’s Inside the Notebook (`main.ipynb`)

### 1) Exploratory Data Analysis (EDA)
- Scatter plots: `age` vs `charges`, `bmi` vs `charges`, `children` vs `charges`
- Category boxplots: `sex`, `smoker`, `region` vs `charges`
- Correlation heatmap (numeric features)
- Target distribution before and after `log1p`

### 2) Data Cleaning / Outliers
The notebook identifies extreme / suspicious points and drops specific rows by index:
- BMI outliers with unexpectedly low charges
- High-charge outliers

(You can replace this manual approach with a reproducible rule-based strategy—see **Improvements**.)

### 3) Preprocessing (scikit-learn)
A `ColumnTransformer` is used to apply different transformations by feature type:

- **Numeric pipeline:** `StandardScaler`
- **Ordinal encoding:** `OrdinalEncoder` for binary categories (`sex`, `smoker`)
- **One-hot encoding:** `OneHotEncoder` for `region`

### 4) Models Trained / Compared
The notebook includes (and tunes) several models:

- `LinearRegression`
- `Ridge` (tuned via `GridSearchCV`)
- `RandomForestRegressor` (tuned)
- `GradientBoostingRegressor` (tuned)
- `XGBRegressor` (tuned)
- `LGBMRegressor` (tuned)
- `CatBoostRegressor` (tuned)
- `VotingRegressor` (ensemble)
- `StackingRegressor` (ensemble)

---

## Results

> **Important:** These scores are computed on the **log-transformed target** (`charges = log1p(charges)`), exactly as in the notebook.

| Rank | Model                      | CV RMSE (5-fold) ↓ | Test MSE ↓  | Test RMSE ↓ |
|:---:|----------------------------|-------------------:|------------:|------------:|
|  1  | Stacking Regressor         |                 —  | 0.120956    | 0.347787    |
|  2  | XGBoost Regressor          | 0.368870           | 0.123103    | 0.350860    |
|  3  | CatBoost Regressor         | 0.370614           | 0.127070    | 0.356469    |
|  4  | Gradient Boosting Regressor| 0.378147           | 0.130413    | 0.361127    |
|  5  | Voting Regressor           |                 —  | 0.135313    | 0.367849    |
|  6  | LightGBM Regressor         | 0.379881           | 0.140840    | 0.375287    |
|  7  | Random Forest Regressor    | 0.381959           | 0.145974    | 0.382065    |
|  8  | Linear Regression          |                 —  | 0.188107    | 0.433713    |
|  9  | Ridge Regression           | 0.447313           | 0.188219    | 0.433842    |

---

## Improvements & Next Steps

### 1) Eliminate Data Leakage
- Split the data **before** any scaling or encoding.
- Wrap preprocessing and the model in a single `Pipeline`.
- Run `GridSearchCV` on the full pipeline so transformers are fit only on training folds.

### 2) Standardize Evaluation
- Report **MAE, RMSE, and R²** for all models.
- Clearly state whether metrics are on the **log scale** or **original charge scale**.
- Inverse-transform predictions (`expm1`) and report at least one metric on the original scale.

### 3) Reproducible Outlier Handling
- Replace manual row drops with rule-based methods (IQR, robust z-score).
- Compare results with and without outlier removal.
- Consider robust losses (Huber) instead of deleting data.

### 4) Stronger Cross-Validation
- Use nested CV for hyperparameter tuning where feasible.
- Increase folds (e.g., 10-fold) for more stable estimates on small datasets.
- Try out Optuna.

### 5) Feature Engineering
- Add interaction terms (e.g., `age × smoker`, `bmi × smoker`).
- Try polynomial features for linear models.
- Evaluate log / Box–Cox transforms for highly skewed predictors.

### 6) Model Simplification & Selection
- Remove dominated models that underperform consistently.
- Prefer simpler models when performance differences are marginal.
- Analyze bias–variance trade-offs explicitly.

### 7) Interpretability
- Plot feature importance for tree-based models.
- Use SHAP values to explain individual predictions.
- Compare explanations across models to validate consistency.

### 8) Deep Learning with PyTorch
- Implement a compact MLP regressor with:
  - Normalized numeric features
  - Embeddings for categorical variables (`sex`, `smoker`, `region`)
- Train with `AdamW`, early stopping, and learning-rate scheduling.
- Compare neural results against the best boosting model on the same splits.

### 9) Reproducibility & Production Readiness
- Fix random seeds across NumPy, scikit-learn, and PyTorch.
- Save the best model (`joblib` / `torch.save`).
- Export a `requirements.txt` or `environment.yml`.
- Refactor the notebook into a clean `src/` training script.


