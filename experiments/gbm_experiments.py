"""Compare GBM libraries (XGBoost, LightGBM, CatBoost) and their blends.

Same setup as regression.py config 7: 5-fold CV, 5 seeds per fold averaged in
log space, MAE objective on log(sold_price). Each library is trained once and
its per-fold test predictions are cached to disk, so blends are computed from
the saved predictions without retraining.
"""

import time

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

NUMERIC_FEATURES = ['bedrooms', 'bathrooms', 'area-sqft', 'built_in', 'lot_area', 'days_old']
CATEGORICAL_FEATURES = ['type', 'neighborhood']
TARGET = 'sold_price'
N_SEEDS = 5
PRED_CACHE = 'data/gbm_experiment_preds.npz'

df = pd.read_csv('data/cleaned_sold.csv')
pois = pd.read_csv('data/pois.csv')
POI_FEATURES = [c for c in pois.columns if c != 'zpid']
df = df.merge(pois, on='zpid', how='left')

FEATURES = NUMERIC_FEATURES + POI_FEATURES + CATEGORICAL_FEATURES
df = df[df[TARGET] > 0].copy()

# XGBoost / LightGBM take pandas categoricals; CatBoost needs strings with no NaN
X = df[FEATURES].copy()
for col in CATEGORICAL_FEATURES:
    X[col] = X[col].astype('category')
X_cb = df[FEATURES].copy()
for col in CATEGORICAL_FEATURES:
    X_cb[col] = X_cb[col].astype(str).fillna('missing')

y_log = np.log(df[TARGET].to_numpy())
y_price = df[TARGET].to_numpy()


def make_xgb(seed):
    return xgb.XGBRegressor(
        n_estimators=2000, learning_rate=0.02, max_depth=7,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        objective='reg:absoluteerror', enable_categorical=True,
        tree_method='hist', random_state=seed)


def make_lgb(seed):
    return lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.02, max_depth=7, num_leaves=127,
        min_child_samples=5, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, objective='regression_l1',
        random_state=seed, verbose=-1)


def make_cb(seed):
    return CatBoostRegressor(
        iterations=2000, learning_rate=0.02, depth=7,
        subsample=0.8, bootstrap_type='Bernoulli', rsm=0.8,
        loss_function='MAE', cat_features=CATEGORICAL_FEATURES,
        random_seed=seed, verbose=0, allow_writing_files=False)


LIBRARIES = {
    'xgb': (make_xgb, X),
    'lgb': (make_lgb, X),
    'cb': (make_cb, X_cb),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds = list(kf.split(X))

# preds[lib] = list of seed-averaged log-price predictions, one array per fold
preds = {lib: [] for lib in LIBRARIES}
pbar = tqdm(total=len(LIBRARIES) * len(folds) * N_SEEDS, desc='Training', unit='model')
for lib, (make_model, X_lib) in LIBRARIES.items():
    start = time.time()
    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        seed_preds = []
        for seed in range(N_SEEDS):
            pbar.set_postfix_str(f'{lib} fold {fold}/5 seed {seed + 1}/{N_SEEDS}')
            model = make_model(42 + seed)
            model.fit(X_lib.iloc[train_idx], y_log[train_idx])
            seed_preds.append(model.predict(X_lib.iloc[test_idx]))
            pbar.update(1)
        preds[lib].append(np.mean(seed_preds, axis=0))
    pbar.write(f"{lib}: trained {len(folds) * N_SEEDS} models in {time.time() - start:.0f}s")
pbar.close()

np.savez(PRED_CACHE,
         **{f'{lib}_fold{i}': p for lib, ps in preds.items() for i, p in enumerate(ps)},
         **{f'test_idx_fold{i}': test_idx for i, (_, test_idx) in enumerate(folds)})
print(f"Cached predictions to {PRED_CACHE}", flush=True)


def evaluate(name, fold_preds):
    """fold_preds: list of log-price prediction arrays, one per fold."""
    rmses, med_pcts = [], []
    for (_, test_idx), pred_log in zip(folds, fold_preds):
        rmses.append(np.sqrt(mean_squared_error(y_log[test_idx], pred_log)))
        pct = np.abs(np.exp(pred_log) - y_price[test_idx]) / y_price[test_idx] * 100
        med_pcts.append(np.median(pct))
    print(f"\n{name}")
    for i, (rmse, pct) in enumerate(zip(rmses, med_pcts), start=1):
        print(f"Fold {i}: log-RMSE = {rmse:.4f} | median % diff = {pct:.2f}%")
    print(f"Loss (log-price RMSE):      {np.mean(rmses):.4f} (+/- {np.std(rmses):.4f})")
    print(f"Median percent difference:  {np.mean(med_pcts):.2f}% (+/- {np.std(med_pcts):.2f}%)")


BLENDS = [('xgb',), ('lgb',), ('cb',),
          ('xgb', 'lgb'), ('xgb', 'cb'), ('lgb', 'cb'),
          ('xgb', 'lgb', 'cb')]
for combo in BLENDS:
    fold_preds = [np.mean([preds[lib][i] for lib in combo], axis=0)
                  for i in range(len(folds))]
    evaluate(' + '.join(combo), fold_preds)
