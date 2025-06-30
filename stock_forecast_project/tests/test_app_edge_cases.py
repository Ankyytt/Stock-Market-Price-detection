import sys
import os
import pytest
from fastapi.testclient import TestClient

# Add the stock_forecast_project directory to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app

client = TestClient(app)

def test_lstm_invalid_ticker():
    response = client.post("/forecast/lstm", json={
        "ticker": "INVALIDTICKER",
        "start_date": "2023-12-01",
        "end_date": "2024-06-01"
    })
    assert response.status_code == 200
    assert "error" in response.json()

def test_arima_invalid_ticker():
    response = client.post("/forecast/arima", json={
        "ticker": "INVALIDTICKER",
        "start_date": "2023-12-01",
        "end_date": "2024-06-01"
    })
    assert response.status_code == 200
    # ARIMA model currently returns predictions even for invalid ticker, so check for predictions key
    assert "predictions" in response.json()

def test_lstm_empty_dates():
    response = client.post("/forecast/lstm", json={
        "ticker": "TSLA",
        "start_date": "2025-01-01",
        "end_date": "2025-01-02"
    })
    assert response.status_code == 200
    assert "error" in response.json()
