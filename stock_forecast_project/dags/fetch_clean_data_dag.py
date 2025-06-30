from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

def fetch_and_clean_data(ticker='AAPL', start_date='2022-01-01', end_date=None):
    import datetime
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start_date, end=end_date)
    df.dropna(inplace=True)
    df.to_csv('data/cleaned_aapl.csv')

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id='fetch_and_clean_stock_data',
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval='@weekly',
    catchup=False,
    tags=['data', 'stock'],
) as dag:

    fetch_clean = PythonOperator(
        task_id='fetch_and_clean',
        python_callable=fetch_and_clean_data
    )
