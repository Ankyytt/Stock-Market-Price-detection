from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def evaluate_models():
    from stock_forecast_project.evaluate import evaluate_arima, evaluate_lstm
    evaluate_arima()
    evaluate_lstm()

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    dag_id='evaluate_forecast_models',
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval='@weekly',
    catchup=False,
    tags=['evaluate'],
) as dag:

    eval_task = PythonOperator(
        task_id='evaluate_models',
        python_callable=evaluate_models
    )
