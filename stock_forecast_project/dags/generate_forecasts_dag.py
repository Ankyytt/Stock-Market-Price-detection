from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd

def run_forecast():
    from stock_forecast_project.forecast import forecast_model
    predictions = forecast_model('AAPL')
    pd.Series(predictions).to_csv('outputs/forecast_aapl.csv')

default_args = {
    'owner': 'airflow',
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
}

with DAG(
    dag_id='generate_forecasts',
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily',
    catchup=False,
    tags=['forecast'],
) as dag:

    forecast_task = PythonOperator(
        task_id='generate_forecast_csv',
        python_callable=run_forecast
    )
