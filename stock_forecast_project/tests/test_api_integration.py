import sys
import os
import pytest
from fastapi.testclient import TestClient

# Add the project root directory to sys.path to import app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_forecast_lstm_valid():
    response = client.get("/forecast/lstm?ticker=TSLA&start_date=2023-12-01&end_date=2024-06-01")
    assert response.status_code == 200
    json_data = response.json()
    assert "historical" in json_data
    assert "forecast" in json_data
    assert isinstance(json_data["historical"], list)
    assert isinstance(json_data["forecast"], list)

def test_forecast_lstm_invalid_ticker():
    response = client.get("/forecast/lstm?ticker=INVALIDTICKER&start_date=2023-12-01&end_date=2024-06-01")
    assert response.status_code == 200
    json_data = response.json()
    assert "error" in json_data

def test_forecast_arima_valid():
    response = client.get("/forecast/arima?ticker=TSLA&start_date=2023-12-01&end_date=2024-06-01")
    assert response.status_code == 200
    json_data = response.json()
    assert "historical" in json_data
    assert "forecast" in json_data

def test_forecast_arima_invalid_ticker():
    response = client.get("/forecast/arima?ticker=INVALIDTICKER&start_date=2023-12-01&end_date=2024-06-01")
    assert response.status_code == 200
    json_data = response.json()
    assert "error" in json_data
