import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

MODEL_SAVE_PATH = r"C:\Users\OmniXXX\Desktop\Project\models\lstm_model.h5"
SCALER_SAVE_PATH = r"C:\Users\OmniXXX\Desktop\Project\models\scaler.save"

TIME_STEPS = 20
EPOCHS = 120
BATCH_SIZE = 32

def fetch_data(ticker='TSLA', start_date=None, end_date=None, interval='1d'):
    if start_date and end_date:
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    else:
        df = yf.download(ticker, period='5y', interval=interval)
    return df

def preprocess_data(df):
    df = df[['Close']].dropna()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values)
    return scaled_data, scaler, df

def create_sequences(data, time_steps=TIME_STEPS):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps, 0])
        y.append(data[i+time_steps, 0])
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, activation='tanh', return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(50, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    ticker = 'TSLA'
    start_date = '2018-01-01'
    end_date = '2024-06-01'
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    df = fetch_data(ticker, start_date=start_date, end_date=end_date)

    print("Preprocessing data...")
    scaled_data, scaler, df_clean = preprocess_data(df)

    print("Creating sequences...")
    X, y = create_sequences(scaled_data, TIME_STEPS)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Train-test split (last 10% as test)
    split_index = int(len(X)*0.9)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print("Building model...")
    model = build_model((TIME_STEPS, 1))

    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[early_stop]
    )

    print(f"Saving model to {MODEL_SAVE_PATH}...")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)

    print(f"Saving scaler to {SCALER_SAVE_PATH}...")
    joblib.dump(scaler, SCALER_SAVE_PATH)

    # Predict on test set
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot actual vs predicted
    plt.figure(figsize=(12,6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title('LSTM Predictions vs Actual')
    plt.xlabel('Time Step')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

    print("Training complete.")

if __name__ == "__main__":
    main()
