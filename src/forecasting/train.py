import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import joblib

def train_baseline(features_parquet, target_col, model_out_path, features_exclude=None):
    df = pd.read_parquet(features_parquet)
    df = df.sort_values('round_start')
    if features_exclude is None:
        features_exclude = ['round_id','user_id','round_start','round_end', target_col]
    feature_cols = [c for c in df.columns if c not in features_exclude]
    X = df[feature_cols].fillna(0)
    y = df[target_col].fillna(0)
    tscv = TimeSeriesSplit(n_splits=5)
    models = []
    scores = []
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_test, label=y_test)
        params = {'objective':'regression','metric':'mae','learning_rate':0.05}
        model = lgb.train(params, dtrain, valid_sets=[dval], early_stopping_rounds=50, num_boost_round=1000, verbose_eval=False)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        scores.append(mae)
        models.append(model)
    # choose last model to save (or average)
    joblib.dump({'model':models[-1], 'feature_cols': feature_cols}, model_out_path)
    return {'cv_mae_mean': sum(scores)/len(scores), 'cv_maes': scores}
