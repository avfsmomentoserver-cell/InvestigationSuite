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
    Minimal prediction: expects last-round aggregated features to be provided in history[-1]['features'].
    This is a demo; replace with feature-store lookup / full featurization in production.
    """
    if MODEL is None:
        return {"error": "model not available"}
    if not req.history:
        return {"error": "history must contain at least one entry"}
    last = req.history[-1]
    features = last.get('features', {})
    df = pd.DataFrame([features])[MODEL['feature_cols']]
    preds = MODEL['model'].predict(df.fillna(0))
    return {
        "entity_id": req.entity_id,
        "horizon": req.horizon,
        "predictions": {"0.5": preds.tolist()},
        "metadata": {"model_id": "baseline:local"}
    }
