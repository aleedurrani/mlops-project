from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

# Add project root to Python path
sys.path.append("/opt/airflow")
from collect_data import collect_weather_data
from preprocess_data import preprocess_data

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 5, 1),
    "retries": 1,
}

with DAG(
    "weather_pipeline",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
) as dag:
    collect_task = PythonOperator(
        task_id="collect_data",
        python_callable=collect_weather_data,
    )

    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    collect_task >> preprocess_task