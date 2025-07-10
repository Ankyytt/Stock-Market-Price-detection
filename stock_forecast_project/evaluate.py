import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
import os

MODEL_DIR = r"C:\Users\OmniXXX\Desktop\Project\models"
ARIMA_MODEL_PATH = os.path.join(MODEL_DIR, "arima_model.pkl")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.save")

def evaluate_arima():
    # Load ARIMA model
    if not os.path.exists(ARIMA_MODEL_PATH):
        print("ARIMA model not found.")
        return

    model = joblib.load(ARIMA_MODEL_PATH)

    # Fetch test data
    df = yf.download("TSLA", start="2023-01-01", end="2024-06-01")
    if df.empty:
        print("No data for evaluation.")
        return

    actual = df['Close'].values

    # Forecast
    forecast = model.forecast(steps=len(actual))

    mse = mean_squared_error(actual, forecast)
    mae = mean_absolute_error(actual, forecast)

    print(f"ARIMA Evaluation - MSE: {mse:.4f}, MAE: {mae:.4f}")

def evaluate_lstm():
    # Load LSTM model and scaler
    if not os.path.exists(LSTM_MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("LSTM model or scaler not found.")
        return

    model = tf.keras.models.load_model(LSTM_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Fetch test data
    df = yf.download("TSLA", start="2023-01-01", end="2024-06-01")
    if df.empty:
        print("No data for evaluation.")
        return

    close_prices = df['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(close_prices)

    TIME_STEPS = 20
    X_test = []
    y_test = []

    for i in range(len(scaled_data) - TIME_STEPS):
        X_test.append(scaled_data[i:i+TIME_STEPS, 0])
        y_test.append(scaled_data[i+TIME_STEPS, 0])

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"LSTM Evaluation - MSE: {mse:.4f}, MAE: {mae:.4f}")
