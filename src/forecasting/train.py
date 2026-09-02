import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import joblib

def train_baseline(features_parquet, target_col, model_out_path, features_exclude=None):
    df = pd.read_parquet(features_parquet)
    df = df.sort_values('round_start')
    if features_exclude is None:
        features_exclude = [
            'round_id', 'user_id', 'round_start', 'round_end', 'prev_round_end',
            'target', 'next_event_count',
            # String/categorical columns — model only predicts event_count (numeric target)
            'trend_direction_label',
            # Internal helpers from compute_round_features
            '_gap_raw',
        ]
    feature_cols = [c for c in df.columns if c not in features_exclude]
    
    # Validate no duplicate rows exist in the feature matrix before splitting
    dup_mask = df.duplicated(subset=feature_cols, keep=False)
    if dup_mask.any():
        dup_ids = df[dup_mask][['round_id', 'user_id']].drop_duplicates()
        raise ValueError(
            f"Duplicate rows detected in the feature matrix:\n{dup_ids.to_string(index=False)}\n"
            f"Total duplicated sample keys: {dup_ids.shape[0]}"
        )
    
    X = df[feature_cols].fillna(0)
    y = df[target_col].fillna(0)
    # Adaptive n_splits: ensure enough samples for TimeSeriesSplit
    n_samples = len(X)
    n_splits = min(5, max(2, n_samples // 2))
    if n_samples < n_splits:
        raise ValueError(
            f"Need at least {n_splits} samples for TimeSeriesSplit with "
            f"{n_splits} folds, got {n_samples}"
        )
    tscv = TimeSeriesSplit(n_splits=n_splits)
    models = []
    scores = []
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        print(f"[TRAIN] fold {i}: {len(X_train)} train / {len(X_test)} test")
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_test, label=y_test)
        params = {'objective':'regression','metric':'mae','learning_rate':0.05}
        model = lgb.train(
            params,
            train_set=dtrain,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)],
            num_boost_round=1000,
        )
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        scores.append(mae)
        models.append(model)
    # choose last model to save (or average)
    joblib.dump({'model':models[-1], 'feature_cols': feature_cols}, model_out_path)
    return {'cv_mae_mean': sum(scores)/len(scores), 'cv_maes': scores}
