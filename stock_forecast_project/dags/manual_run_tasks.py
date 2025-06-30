"""
This script allows manual execution of Airflow DAG tasks for testing purposes on Windows,
bypassing Airflow scheduler limitations.

Each function corresponds to a task in the DAGs and can be run independently.
"""

import sys
import os
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import joblib
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/stock_data.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/arima_model.pkl')

def fetch_and_clean_data(ticker='TSLA', start_date='2023-12-01', end_date=None):
    import datetime
    logging.info(f"Fetching data for {ticker} from {start_date} to {end_date or 'today'}")
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    df = df[['Close']].dropna()
    df['Returns'] = df['Close'].pct_change()
    df = df.dropna()
    df.to_csv(DATA_PATH)
    logging.info(f"Data saved to {DATA_PATH}")
    return df

def train_arima_model():
    logging.info("Loading data for training ARIMA model")
    df = pd.read_csv(DATA_PATH, index_col=0)
    # Convert Close column to numeric, coerce errors to NaN and drop them
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    model = ARIMA(df['Close'], order=(5,1,0))
    model_fit = model.fit()
    joblib.dump(model_fit, MODEL_PATH)
    logging.info(f"ARIMA model saved to {MODEL_PATH}")
    return model_fit

def generate_forecast(steps=10):
    logging.info("Loading ARIMA model for forecasting")
    model_fit = joblib.load(MODEL_PATH)
    forecast = model_fit.forecast(steps=steps)
    logging.info(f"Forecast for next {steps} steps: {forecast}")
    return forecast

def evaluate_model():
    # Placeholder for evaluation logic
    logging.info("Evaluating model performance (not implemented)")
    pass

def backup_models():
    # Placeholder for backup logic
    logging.info("Backing up models (not implemented)")
    pass

if __name__ == "__main__":
    # Example manual run sequence
    fetch_and_clean_data()
    train_arima_model()
    generate_forecast()
    evaluate_model()
    backup_models()
