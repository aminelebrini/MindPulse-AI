import os
import runpy
from datetime import datetime
from pathlib import Path
from airflow import DAG
from airflow.operators.python import PythonOperator


ROOT = Path(__file__).resolve().parents[1]


def train_model() -> None:
    """Run the training script from the project root."""
    previous_directory = Path.cwd()
    try:
        os.chdir(ROOT)
        runpy.run_path(str(ROOT / "main.py"), run_name="__main__")
    finally:
        os.chdir(previous_directory)


with DAG(
    dag_id="mindpulseai_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["mindpulse"],
) as dag:
    ml = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )