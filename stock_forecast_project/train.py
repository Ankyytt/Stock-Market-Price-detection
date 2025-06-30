"""
Time Series Forecasting Training Script with ARIMA and LSTM models and MLflow integration.
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import mlflow
#import mlflow.keras
import joblib

from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def fetch_data(ticker='TSLA', start_date='2023-12-01', end_date='2024-06-01'):
    df = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    return df

def preprocess_data(df):
    df = df[['Close']].dropna()
    df['Returns'] = df['Close'].pct_change()
    df = df.dropna()
    return df

def train_arima(df):
    model = ARIMA(df['Close'], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=10)
    return model_fit, forecast

def create_lstm_dataset(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

def train_lstm(df):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df[['Close']].values)

    time_steps = 10
    X, y = create_lstm_dataset(scaled_data, time_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(time_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=50, verbose=1)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return model, scaler, mse

def plot_forecast(df, forecast, title='Forecast'):
    plt.figure(figsize=(10,6))
    plt.plot(df.index, df['Close'], label='Actual')
    plt.plot(forecast.index, forecast.values, label='Forecast')
    plt.title(title)
    plt.legend()
    plt.savefig('forecast_plot.png')
    plt.close()

def main():
    mlflow.set_experiment("Stock_Price_Forecasting")

    print("Fetching data...")
    df = fetch_data()
    print("Preprocessing data...")
    df_processed = preprocess_data(df)
    print(df_processed.head())

    # ARIMA model training
    with mlflow.start_run(run_name="ARIMA"):
        print("Training ARIMA model...")
        arima_model, arima_forecast = train_arima(df_processed)
        # Convert forecast and actual to numpy arrays for mse calculation
        actual = df_processed['Close'][-10:].values
        forecast_values = arima_forecast.values if hasattr(arima_forecast, 'values') else arima_forecast
        mse_arima = ((forecast_values - actual) ** 2).mean()
        print(f"ARIMA MSE: {mse_arima}")

        mlflow.log_param("model_type", "ARIMA")
        mlflow.log_metric("mse", float(mse_arima))
        arima_model_path = "models/arima_model.pkl"
        joblib.dump(arima_model, arima_model_path)
        mlflow.log_artifact(arima_model_path)

    # LSTM model training
    with mlflow.start_run(run_name="LSTM"):
        print("Training LSTM model...")
        lstm_model, scaler, mse_lstm = train_lstm(df_processed)
        print(f"LSTM MSE: {mse_lstm}")

        mlflow.log_param("model_type", "LSTM")
        mlflow.log_metric("mse", mse_lstm)
        lstm_model_path = "models/lstm_model.h5"
        scaler_path = "models/scaler.save"
        lstm_model.save(lstm_model_path)
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(lstm_model_path)
        mlflow.log_artifact(scaler_path)

    print("Training complete.")

if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models")
    main()
