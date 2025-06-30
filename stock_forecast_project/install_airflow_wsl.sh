#!/bin/bash
# Script to install and run Apache Airflow in WSL (Ubuntu) for full DAG testing

# Update and install dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv

# Create and activate virtual environment
python3 -m venv airflow_venv
source airflow_venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Apache Airflow with constraints for version 2.6.3
AIRFLOW_VERSION=2.6.3
PYTHON_VERSION=3.10
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

# Initialize Airflow database
export AIRFLOW_HOME=~/airflow
airflow db init

# Create user for Airflow UI login
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Start Airflow webserver and scheduler in background
airflow webserver -p 8080 &
airflow scheduler &

echo "Airflow installation and startup complete."
echo "Access the Airflow UI at http://localhost:8080"
