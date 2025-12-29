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

## Getting Started

### 1) Clone
```bash
git clone <your-repo-url>
cd <your-repo-folder>
