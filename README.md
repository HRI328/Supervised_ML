# ML Model Builder

## What Is It?

ML Model Builder is an end-to-end supervised machine learning pipeline that automates the entire process from raw data to final predictions. It handles data preparation, feature selection, model training, hyperparameter optimization, model comparison, and prediction — all in one clean, flexible framework.

---

## Why Use It?

Building a machine learning model involves many steps, and it's easy to make mistakes along the way — especially when handling mixed data types, choosing the right features, or ensuring the prediction pipeline matches the training pipeline exactly. ML Model Builder solves this by:

- **Reducing boilerplate code** — no need to manually write encoding, scaling, or splitting logic every time.
- **Enforcing consistency** — the prediction pipeline mirrors the training pipeline exactly, so there are no shape mismatches or data leakage issues.
- **Making feature selection flexible** — users choose how to handle numerical and categorical features independently.
- **Comparing models fairly** — all models are trained and evaluated using the same data and preprocessing steps.

---

## What Does It Support?

### Tasks
- **Classification** — predict a category or label
- **Regression** — predict a continuous value

### Models
| Classification | Regression |
|---|---|
| Logistic Regression | Linear Regression |
| Decision Tree | Ridge |
| Random Forest | Lasso |
| Gradient Boosting | Decision Tree |
| SVM | Random Forest |
| KNN | Gradient Boosting |
| XGBoost* | SVR |
| CatBoost* | XGBoost* |
| | CatBoost* |

\* requires separate installation: `pip install xgboost catboost`

### Feature Selection Modes
One of the most powerful features of this pipeline is the flexible feature selection system. Users can independently control how numerical and categorical features are handled:

| Mode | Numerical Features | Categorical Features |
|---|---|---|
| **A** | Keep all | Keep all |
| **B** | Keep all | Chi2 selection |
| **C** | PCA reduction | Keep all |
| **D** | PCA reduction | Chi2 selection |

- **PCA (Principal Component Analysis)** reduces the dimensionality of numerical features by projecting them into a smaller space while preserving the most important variance. It is only applied to numerical features because it relies on continuous values.
- **Chi2 (Chi-Square Test)** measures the statistical relationship between each categorical feature and the target variable. Features with the strongest relationship are kept, and the rest are removed. It is only applied to categorical features because it works on discrete, non-negative values.

---

## How Does It Work?

The pipeline follows five clear steps:

### Step 1 — Prepare Data
Split the data into training and test sets. Automatically detect and encode categorical features using one-hot encoding (for features with 10 or fewer unique values) or label encoding (for high-cardinality features).

### Step 2 — Select Features
Choose a feature selection mode (A, B, C, or D). The pipeline separates numerical and categorical features, applies the selected transformations independently, then combines them back together before scaling.

### Step 3 — Train & Compare
Train all available models using cross-validation. Compare their performance side by side using accuracy (classification) or R² (regression). Optionally optimize a specific model's hyperparameters using Grid Search or Random Search.

### Step 4 — Predict
Make predictions on new data. The prediction function automatically applies the exact same preprocessing steps as training — encoding, feature selection, and scaling — in the correct order, so predictions are always consistent.

### Step 5 — Evaluate & Visualize
Evaluate model performance with detailed metrics. Generate visualizations including model comparison charts, cross-validation scores, confusion matrices, feature importance plots, and PCA analysis.

---

## Quick Start

```python
from ml_model_builder import MLModelBuilder

# 1. Initialize
builder = MLModelBuilder(task='classification')

# 2. Prepare data
builder.prepare_data(X, y)

# 3. Select features (choose mode A, B, C, or D)
builder.select_features(mode='D', pca_variance=0.95, chi2_k=5)

# 4. Train and compare
builder.train_all_models(cv=5)
builder.compare_models()

# 5. Optimize and select best
builder.optimize_model('Random Forest', param_grid)
builder.select_best_model()

# 6. Predict
predictions = builder.predict(X_new)
predictions, probabilities = builder.predict(X_new, return_proba=True)

# 7. Save and reload
builder.save_model('model.pkl')
builder.load_model('model.pkl')
```

---

## Dependencies

**Required:**
```
scikit-learn, numpy, pandas, matplotlib, seaborn, scipy
```

**Optional (for boosting models):**
```
xgboost, catboost
```
