@echo off
REM This script installs Apache Airflow 2.6.3 with constraints compatible with Python 3.10

REM Activate virtual environment
call venv\Scripts\activate

REM Uninstall conflicting Airflow and related packages
pip uninstall -y apache-airflow apache-airflow-core apache-airflow-task-sdk ^
    apache-airflow-providers-common-compat apache-airflow-providers-common-io ^
    apache-airflow-providers-common-sql apache-airflow-providers-fab ^
    apache-airflow-providers-ftp apache-airflow-providers-http ^
    apache-airflow-providers-imap apache-airflow-providers-smtp ^
    apache-airflow-providers-sqlite apache-airflow-providers-standard ^
    opentelemetry-exporter-otlp opentelemetry-proto colorlog

REM Install Airflow 2.6.3 with constraints for Python 3.10
pip install "apache-airflow==2.6.3" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.6.3/constraints-3.10.txt"

REM Freeze working environment to requirements_cleaned.txt
pip freeze > requirements_cleaned.txt

echo Airflow 2.6.3 installation complete.
