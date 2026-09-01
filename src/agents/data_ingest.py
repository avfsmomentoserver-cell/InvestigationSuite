"""
Agent: Data ingestion watcher. Polls sources (S3, DB) for new recordings and triggers ETL.
"""
import time
import subprocess

def poll():
    # Example pseudo-code
    while True:
        print("Polling data sources for new recordings...")
        # check S3 / DB for new files, then call ETL
        # e.g., run subprocess: python -m src.forecasting.etl ...
        time.sleep(60)

if __name__ == "__main__":
    poll()
