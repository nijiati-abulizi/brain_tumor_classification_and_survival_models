# Feature Selection for Methylation Data using Bayesian Optimization

## Overview
This script uses Bayesian optimization with XGBoost to select important features from methylation data. The goal is to optimize hyperparameters and improve classification accuracy. Bayesian optimization efficiently explores the search space, avoiding exhaustive grid searches.

## Comparison of Models for p >> n
When dealing with high-dimensional data (p >> n, where the number of features far exceeds the number of samples), different models have strengths and weaknesses. The following table summarizes the comparison:

| Model                  | Strengths | Weaknesses |
|------------------------|--------------------------------|------------------------------------------------|
| **LogisticRegressionCV** | Works well for small datasets, interpretable, no tuning needed | Struggles with high-dimensional data, assumes linear relationships |
| **RandomForest**        | Handles high-dimensional data, robust to noise, non-linear relationships | Can overfit if not tuned properly, lacks interpretability |
| **XGBoost**            | Handles high-dimensional data well, feature importance, regularization, fast with GPU support | Requires hyperparameter tuning, can be complex to optimize |

## Why XGBoost?
1. **Handling High-Dimensional Data**: Unlike LogisticRegressionCV, XGBoost uses boosting techniques that effectively handle p >> n problems.
2. **Feature Selection**: XGBoost naturally ranks feature importance, making it suitable for selecting the most relevant methylation markers.
3. **Regularization**: Built-in L1/L2 regularization helps prevent overfitting.
4. **Scalability**: Optimized for GPU acceleration and large-scale data.
5. **Non-Linearity**: Unlike LogisticRegressionCV, it captures complex feature interactions.


## Code Implementation

```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import xgboost as xgb

# Define Bayesian Optimization Search Space
search_spaces = {
    'reg_alpha': Real(10, 100, prior='log-uniform'),
    'reg_lambda': Real(10, 100, prior='log-uniform'),
    'max_depth': Integer(3, 5),
    'learning_rate': Real(0.001, 0.05, prior='log-uniform'),
    'n_estimators': Integer(50, 100),
    'colsample_bytree': Real(0.1, 0.3),
    'subsample': Real(0.5, 0.8)
}

# Initialize XGBoost Classifier
xgb_clf = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=len(y.unique()),
    eval_metric="mlogloss",
    tree_method="hist",
    n_jobs=-1
)

# Perform Bayesian Optimization
bayes_search = BayesSearchCV(
    xgb_clf,
    search_spaces,
    n_iter=15,
    scoring='accuracy',
    cv=2,
    n_jobs=-1,
    verbose=3
)

# Fit Bayesian Search
bayes_search.fit(X_train, y_train)

print("\n✅ Best Parameters Found:", bayes_search.best_params_)
```

By using Bayesian optimization, we efficiently fine-tune hyperparameters, improving feature selection and predictive performance in methylation classification tasks.


