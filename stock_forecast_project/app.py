import os
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf
import uvicorn

app = FastAPI()

# Mount static files for frontend
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "frontend")), name="static")

# CORS middleware for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redirect root to frontend index.html with JSON message for API test
from fastapi.responses import JSONResponse

@app.get("/", response_class=JSONResponse)
async def root():
    return {"message": "Stock Price Forecasting API is running"}

class ForecastRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str

# Update model path to absolute path as confirmed by existing files
MODEL_PATH = r"C:\Users\OmniXXX\Desktop\Project\models\lstm_model.h5"
SCALER_PATH = r"C:\Users\OmniXXX\Desktop\Project\models\scaler.save"

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

TIME_STEPS = 20

def create_sequences(data, time_steps=TIME_STEPS):
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
    return np.array(X)

from fastapi import Request

@app.post("/forecast/lstm")
async def forecast_lstm(request: Request):
    import yfinance as yf
    from fastapi import HTTPException
    import pandas as pd
    from datetime import timedelta

    data = await request.json()
    ticker = data.get("ticker")
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    forecast_days = data.get("forecast_days", 14)  # default 14 days forecast

    # Validate inputs
    if not ticker or not start_date or not end_date:
        raise HTTPException(status_code=400, detail="Missing required parameters: ticker, start_date, end_date")

    try:
        start_date_parsed = pd.to_datetime(start_date)
        end_date_parsed = pd.to_datetime(end_date)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    if start_date_parsed >= end_date_parsed:
        raise HTTPException(status_code=400, detail="Start date must be before end date.")

    # Fetch historical data for requested ticker and date range
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found for given ticker and date range.")

    close_prices = df['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(close_prices)

    # Use last TIME_STEPS data points as seed for prediction
    input_seq = scaled_data[-TIME_STEPS:].reshape(1, TIME_STEPS, 1)

    preds_scaled = []
    current_seq = input_seq

    for _ in range(forecast_days):
        pred = model.predict(current_seq)[0, 0]
        # Clip prediction to be non-negative to avoid negative prices
        pred = max(pred, 0)
        preds_scaled.append(pred)
        # Append prediction and remove oldest value to maintain sequence length
        current_seq = np.append(current_seq[:, 1:, :], [[[pred]]], axis=1)

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled).flatten()

    # Prepare future dates for predictions
    last_date = df.index[-1]
    pred_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]

    # Prepare historic data for chart
    historic_data = [{"date": str(date.date()), "close": float(price)} for date, price in zip(df.index, df['Close'].values)]

    result = {
        "ticker": ticker,
        "predictions": [{"date": str(date.date()), "predicted_close": float(pred)} for date, pred in zip(pred_dates, preds)],
        "historic": historic_data
    }

    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
