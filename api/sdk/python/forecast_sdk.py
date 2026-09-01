import requests
from typing import List, Dict

class ForecastSDK:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url.rstrip('/')

    def forecast(self, entity_id: str, history: List[Dict], horizon: int=1, mode: str="deterministic"):
        payload = {"entity_id": entity_id, "history": history, "horizon": horizon, "mode": mode}
        r = requests.post(f"{self.base_url}/v1/forecast", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
