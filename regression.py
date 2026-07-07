"""XGBoost regression on structural attributes only.

Predicts log(sold_price) from cleaned_sold.csv and reports the loss (RMSE on
log price) along with the median percent difference computed on regular price.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Structural attributes only
NUMERIC_FEATURES = ['bedrooms', 'bathrooms', 'area-sqft', 'built_in', 'lot_area', 'days_old']
CATEGORICAL_FEATURES = ['type', 'neighborhood']
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = 'sold_price'

df = pd.read_csv('data/cleaned_sold.csv')

# Drop rows without a sale price and build the feature matrix / target
df = df[df[TARGET] > 0].copy()

X = df[FEATURES].copy()
for col in CATEGORICAL_FEATURES:
    X[col] = X[col].astype('category')

# Train on log price so the target is closer to normal
y_log = np.log(df[TARGET].to_numpy())
y_price = df[TARGET].to_numpy()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_rmse = []          # loss: RMSE on log price
fold_median_pct = []    # median percent difference on regular price
fold_importances = []   # gain-based feature importance per fold

for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train = y_log[train_idx]
    y_test_log = y_log[test_idx]
    y_test_price = y_price[test_idx]

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        enable_categorical=True,
        tree_method='hist',
        random_state=42,
    )
    model.fit(X_train, y_train)

    pred_log = model.predict(X_test)

    # Loss on log price
    rmse = np.sqrt(mean_squared_error(y_test_log, pred_log))
    fold_rmse.append(rmse)

    # Median percent difference on regular price
    pred_price = np.exp(pred_log)
    pct_diff = np.abs(pred_price - y_test_price) / y_test_price * 100
    median_pct = np.median(pct_diff)
    fold_median_pct.append(median_pct)

    # Gain-based feature importance (0 for features never used in a split)
    gain = model.get_booster().get_score(importance_type='gain')
    fold_importances.append(pd.Series({f: gain.get(f, 0.0) for f in FEATURES}))

    print(f"Fold {fold}: log-RMSE = {rmse:.4f} | median % diff = {median_pct:.2f}%")

print("-" * 50)
print(f"Loss (log-price RMSE):      {np.mean(fold_rmse):.4f} (+/- {np.std(fold_rmse):.4f})")
print(f"Median percent difference:  {np.mean(fold_median_pct):.2f}% (+/- {np.std(fold_median_pct):.2f}%)")

# Feature importance averaged across folds, normalized to sum to 1
importance = pd.concat(fold_importances, axis=1).mean(axis=1)
importance = (importance / importance.sum()).sort_values(ascending=False)
print("-" * 50)
print("Feature importance (gain, averaged across folds):")
for feature, score in importance.items():
    print(f"  {feature:<14} {score:.3f}")