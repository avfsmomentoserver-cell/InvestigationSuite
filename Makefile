install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

test:
	pytest -q

run:
	uvicorn src.forecasting.serve:app --reload --port 8000
