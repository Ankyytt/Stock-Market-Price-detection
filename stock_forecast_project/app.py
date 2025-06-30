from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import yfinance as yf
import pandas as pd
try:
    from tensorflow.keras.models import load_model
except ImportError:
    from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = FastAPI()

class ForecastRequest(BaseModel):
    ticker: str = "TSLA"
    start_date: str = "2023-12-01"
    end_date: str = "2024-06-01"

def fetch_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    return df[['Close']].dropna()

def create_lstm_dataset(data, time_steps=10):
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
    return np.array(X)

@app.get("/")
def root():
    return {"message": "Stock Forecasting API is running."}

@app.post("/forecast/lstm")
def forecast_lstm(request: ForecastRequest):
    try:
        df = fetch_data(request.ticker, request.start_date, request.end_date)
        if df.empty:
            return {"error": "No data found for the given ticker and date range."}
        scaler = joblib.load("models/scaler.save")
        model = load_model("models/lstm_model.h5")

        scaled_data = scaler.transform(df.values)
        X = create_lstm_dataset(scaled_data)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        predictions = model.predict(X)
        predictions_rescaled = scaler.inverse_transform(predictions)

        return {"predictions": predictions_rescaled.flatten().tolist()}
    except Exception as e:
        return {"error": str(e)}

@app.post("/forecast/arima")
def forecast_arima(request: ForecastRequest):
    try:
        import joblib
        model = joblib.load("models/arima_model.pkl")
        forecast = model.forecast(steps=10)
        return {"predictions": forecast.tolist()}
    except Exception as e:
        return {"error": str(e)}
