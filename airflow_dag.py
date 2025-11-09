"""
Apache Airflow DAG for AQI Prediction Pipeline
Run this with: airflow dags list
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.pipeline.feature_pipeline import run_feature_pipeline
from src.pipeline.training_pipeline import run_training_pipeline

# Default arguments
default_args = {
    'owner': 'aqi-prediction',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# Create DAG
dag = DAG(
    'aqi_prediction_pipeline',
    default_args=default_args,
    description='AQI Prediction Pipeline - Feature and Training',
    schedule_interval=None,  # Can be set to '@hourly' or '@daily'
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['aqi', 'ml', 'prediction']
)

# Feature Pipeline Task (runs hourly)
feature_pipeline_task = PythonOperator(
    task_id='run_feature_pipeline',
    python_callable=run_feature_pipeline,
    op_kwargs={'use_hopsworks': False},
    dag=dag
)

# Training Pipeline Task (runs daily)
training_pipeline_task = PythonOperator(
    task_id='run_training_pipeline',
    python_callable=run_training_pipeline,
    op_kwargs={
        'use_feature_store': True,
        'use_model_registry': True,
        'use_hopsworks': False,
        'use_mlflow': False
    },
    dag=dag
)

# Set task dependencies
feature_pipeline_task >> training_pipeline_task

