from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Forecasting Rounds API")
MODEL_PATH = "models/baseline.joblib"


class ForecastRequest(BaseModel):
    entity_id: str
    history: list  # list of dicts with ts & features
    horizon: int = 1
    mode: str = "deterministic"


@app.on_event("startup")
def load_model():
    global MODEL
    try:
        MODEL = joblib.load(MODEL_PATH)
    except Exception:
        MODEL = None


@app.post("/v1/forecast")
def forecast(req: ForecastRequest):
    """
    Expanded prediction endpoint supporting 8 human-readable dimensions:

    PREDICTION OUTPUT SCHEMA (all values are plain-language first):

    event_count_predicted : "Things That Will Happen Next" — expected count
    gap_hours_until_next  : "When They Come Back Next" — hours until next activity
    trend_direction       : "Busier or Quieter?" — trending_up / stable / trending_down
    session_consistency   : "Do Bursts Vary?" — low_cv (stable bursts) / high_cv (unpredictable)
    top_event_type        : "Most Likely Activity" — which event type dominates
    growth_ratio          : "Are Sessions Growing?" — how much bigger vs previous round
    stay_home_risk        : "Chance They Stop Coming" — 0.0 (will return) to 1.0 (stopped)
    confidence_range      : "How Sure Are We?" — lower_bound / upper_bound for prediction
    """
    if MODEL is None:
        return {"error": "model not available"}
    if not req.history:
        return {"error": "history must contain at least one entry"}
    last = req.history[-1]
    features = last.get('features', {})
    model_data = MODEL
    model = model_data['model']
    feature_cols = model_data['feature_cols']

    df = pd.DataFrame([features])[feature_cols].fillna(0)
    raw_pred = model.predict(df)[0]

    # Extract feature values for dimension labels
    feat_dict = df[feature_cols].iloc[0].to_dict()
    trend_dir = feat_dict.get('trend_direction_label', 'stable')
    top_et = None
    for c in ['click', 'submit', 'view']:
        if c in feat_dict and feat_dict[c] > 0:
            top_et = c
            break

    # Confidence range (simple heuristic based on historical std from cv feature)
    cv = feat_dict.get('cv_event_count', 0) or 0
    margin = raw_pred * max(0.2, cv)  # wider band when sessions vary more
    lower = max(0, raw_pred - margin)
    upper = raw_pred + margin

    return {
        "entity_id": req.entity_id,
        "horizon": req.horizon,
        "predictions": {
            "event_count_predicted": [round(raw_pred, 2)],
            "gap_hours_until_next": [round(feat_dict.get('rolling_mean_gap', 0) / 3600 if feat_dict.get('rolling_mean_gap', 0) > 0 else 0, 2)],
            "trend_direction": [str(trend_dir)],
            "session_consistency": ["low_cv" if cv < 0.5 else "high_cv"],
            "top_event_type": [str(top_et)] if top_et else ["unknown"],
            "growth_ratio": [round(feat_dict.get('event_count_growth_ratio', 0), 2)],
            "stay_home_risk": [round(feat_dict.get('stay_home_risk_score', 0), 2)],
            "confidence_range": {
                "lower_bound": [round(lower, 2)],
                "upper_bound": [round(upper, 2)]
            }
        },
        "metadata": {
            "model_id": "baseline:local",
            "prediction_labels": {
                "event_count_predicted": "Things That Will Happen Next",
                "gap_hours_until_next": "When They Come Back Next (hours)",
                "trend_direction": "Busier or Quieter? (trending_up/stable/trending_down)",
                "session_consistency": "Do Bursts Vary? (low_cv=stable/high_cv=unpredictable)",
                "top_event_type": "Most Likely Activity Type",
                "growth_ratio": "Are Sessions Growing? (positive=growing, negative=shrinking)",
                "stay_home_risk": "Chance They Stop Coming (0=will return, 1=likely stopped)",
                "confidence_range": "How Sure Are We? (interval where actual value likely falls)"
            }
        }
    }
