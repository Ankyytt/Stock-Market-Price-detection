from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    dag_id='daily_model_retrain',
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily',
    catchup=False,
    tags=['retrain', 'stock'],
) as dag:

    retrain_arima = BashOperator(
        task_id='train_arima',
        bash_command='python stock_forecast_project/train_arima.py'
    )

    retrain_lstm = BashOperator(
        task_id='train_lstm',
        bash_command='python stock_forecast_project/train_lstm.py'
    )

    retrain_arima >> retrain_lstm
