from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

with DAG(
    dag_id='backup_trained_models',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily',
    catchup=False,
    tags=['backup'],
) as dag:

    backup = BashOperator(
        task_id='backup_models',
        bash_command='tar -czf backups/models_$(date +%Y%m%d).tar.gz stock_forecast_project/models'
    )
