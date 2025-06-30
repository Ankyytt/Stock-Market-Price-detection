"""
Forecasting script to load trained models and make predictions.
"""

import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
try:
    from tensorflow.keras.models import load_model
except ImportError:
    from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def fetch_data(ticker='TSLA', start_date='2023-12-01', end_date='2024-06-01'):
    df = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    return df

def preprocess_data(df):
    df = df[['Close']].dropna()
    return df

def forecast_arima(model_path, steps=10):
    model = joblib.load(model_path)
    forecast = model.forecast(steps=steps)
    return forecast

def create_lstm_dataset(data, time_steps=10):
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
    return np.array(X)

def forecast_lstm(model_path, scaler_path, df, time_steps=10):
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    scaled_data = scaler.transform(df[['Close']].values)
    X = create_lstm_dataset(scaled_data, time_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    predictions = model.predict(X)
    predictions_rescaled = scaler.inverse_transform(predictions)
    return predictions_rescaled

def plot_forecast(df, forecast, title='Forecast'):
    plt.figure(figsize=(10,6))
    plt.plot(df.index, df['Close'], label='Actual')
    plt.plot(forecast.index, forecast.values, label='Forecast')
    plt.title(title)
    plt.legend()
    plt.savefig('forecast_plot.png')
    plt.show()

def main():
    print("Fetching data...")
    df = fetch_data()
    df_processed = preprocess_data(df)

    print("Loading and forecasting with ARIMA model...")
    arima_forecast = forecast_arima('models/arima_model.pkl')
    print(arima_forecast)

    print("Loading and forecasting with LSTM model...")
    lstm_forecast = forecast_lstm('models/lstm_model.h5', 'models/scaler.save', df_processed)
    print(lstm_forecast)

if __name__ == "__main__":
    main()
