"""
Agent manager: coordinates multiple agents (data ingestion, retrain, monitor, notifications).
This is a simple orchestrator that launches worker processes or schedules tasks.
For production, replace with a proper orchestration (Celery, Airflow, or a custom async scheduler).
"""
import subprocess
import time
import os
from multiprocessing import Process

AGENTS = [
    ("data_ingest", "src/agents/data_ingest.py"),
    ("monitor", "src/agents/monitor.py"),
    ("retrain", "src/agents/retrain.py")
]

def run_agent(path):
    while True:
        try:
            subprocess.run(["python3", path], check=True)
        except Exception as e:
            print(f"Agent {path} failed: {e}. Restarting in 5s.")
            time.sleep(5)

if __name__ == "__main__":
    procs = []
    for name, path in AGENTS:
        p = Process(target=run_agent, args=(path,))
        p.start()
        procs.append(p)
    try:
        while True:
            time.sleep(10)
            for p in procs:
                if not p.is_alive():
                    print("Restarting dead agent")
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()
