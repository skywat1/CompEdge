"""XGBoost regression on structural attributes and POI counts.

Predicts log(sold_price) from cleaned_sold.csv joined with pois.csv and reports
the loss (RMSE on log price) along with the median percent difference computed
on regular price.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# Structural Attributes
NUMERIC_FEATURES = ['bedrooms', 'bathrooms', 'area-sqft', 'built_in', 'lot_area', 'days_old']
CATEGORICAL_FEATURES = ['type', 'neighborhood']
TARGET = 'sold_price'

# PLUTO attributes (data/pluto.csv from get_pluto.py), each sale bucketed to its
# historical PLUTO version. histdist/landmark are binary flags (1/0/NaN); the
# size/dimension/year/FAR fields are numeric; the rest are categorical codes.
# Load-bearing set, kept after permutation-importance testing. bldgclass carries
# almost all of PLUTO's value (beats Zillow `type`); the rest are small but
# nonzero structural contributors. Dropped as ~0 / redundant: landmark, commfar,
# histdist, landuse (redundant w/ neighborhood/bldgclass), lotdepth, yearalter1,
# bldgfront (noise-level), yearbuilt (redundant w/ Zillow built_in).
PLUTO_NUMERIC = ['bldgarea', 'resarea', 'builtfar', 'bldgdepth', 'lotfront', 'residfar',
                 'assesstot', 'assessland']
PLUTO_CATEGORICAL = ['bldgclass']

df = pd.read_csv('data/cleaned_sold.csv')

# POIs
pois = pd.read_csv('data/pois.csv')
POI_FEATURES = [c for c in pois.columns if c != 'zpid']
df = df.merge(pois, on='zpid', how='left')

# PLUTO (already cleaned in get_pluto.py; just join and register the features)
pluto = pd.read_csv(
    'data/pluto.csv',
    usecols=['zpid'] + PLUTO_NUMERIC + PLUTO_CATEGORICAL,
    dtype={c: 'string' for c in PLUTO_CATEGORICAL},
).drop_duplicates('zpid')
df = df.merge(pluto, on='zpid', how='left')

CATEGORICAL_FEATURES = CATEGORICAL_FEATURES + PLUTO_CATEGORICAL
FEATURES = NUMERIC_FEATURES + PLUTO_NUMERIC + POI_FEATURES + CATEGORICAL_FEATURES

# Drop rows without a sale price and build the feature matrix / target
df = df[df[TARGET] > 0].copy()

X = df[FEATURES].copy()
for col in CATEGORICAL_FEATURES:
    X[col] = X[col].astype('category')

# Train on log price so the target is closer to normal
y_log = np.log(df[TARGET].to_numpy())
y_price = df[TARGET].to_numpy()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
N_SEEDS = 5  # models per fold; predictions are averaged to smooth out subsampling noise

fold_rmse = []          # loss: RMSE on log price
fold_median_pct = []    # median percent difference on regular price
fold_importances = []   # gain-based feature importance per fold

pbar = tqdm(total=kf.get_n_splits() * N_SEEDS, desc='Training', unit='model')
for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train = y_log[train_idx]
    y_train_price = y_price[train_idx]
    y_test_log = y_log[test_idx]
    y_test_price = y_price[test_idx]

    seed_preds = []
    seed_train_preds = []
    seed_gains = []
    for seed in range(N_SEEDS):
        model = xgb.XGBRegressor(
            n_estimators=2000,
            learning_rate=0.02,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            # optimize typical error (matches median % diff) rather than squared
            objective='reg:absoluteerror',
            enable_categorical=True,
            tree_method='hist',
            random_state=42 + seed,
        )
        model.fit(X_train, y_train)
        seed_preds.append(model.predict(X_test))
        seed_train_preds.append(model.predict(X_train))
        seed_gains.append(model.get_booster().get_score(importance_type='gain'))
        pbar.update(1)

    pred_log = np.mean(seed_preds, axis=0)
    pred_train_log = np.mean(seed_train_preds, axis=0)

    # Loss on log price
    rmse = np.sqrt(mean_squared_error(y_test_log, pred_log))
    fold_rmse.append(rmse)
    train_rmse = np.sqrt(mean_squared_error(y_train, pred_train_log))

    # Median percent difference on regular price
    pred_price = np.exp(pred_log)
    pct_diff = np.abs(pred_price - y_test_price) / y_test_price * 100
    median_pct = np.median(pct_diff)
    fold_median_pct.append(median_pct)
    train_median_pct = np.median(
        np.abs(np.exp(pred_train_log) - y_train_price) / y_train_price * 100)

    # Gain-based feature importance (0 for features never used in a split),
    # averaged across seeds
    fold_importances.append(pd.Series({
        f: np.mean([g.get(f, 0.0) for g in seed_gains]) for f in FEATURES}))

    pbar.write(
        f"Fold {fold}: log-RMSE = {rmse:.4f} (train {train_rmse:.4f}) | "
        f"median % diff = {median_pct:.2f}% (train {train_median_pct:.2f}%)")

pbar.close()
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