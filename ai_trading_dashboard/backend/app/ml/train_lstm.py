from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from app.services.indicators import add_indicators

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "lstm_model.keras"
SCALER_PATH = MODEL_DIR / "lstm_scaler.joblib"


def build_dataset(ticker: str, period: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    df = add_indicators(df).dropna().copy()
    return df


def train(ticker: str, period: str) -> None:
    df = build_dataset(ticker, period)

    features = ["Close", "SMA_20", "EMA_20", "RSI", "MACD", "MACD_SIGNAL"]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])

    joblib.dump(scaler, SCALER_PATH)

    lookback = 30
    X, y = [], []
    target_idx = features.index("Close")
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i])
        y.append(scaled[i, target_idx])

    X = np.array(X)
    y = np.array(y)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=32, verbose=1)
    model.save(MODEL_PATH)

    print(f"Saved LSTM model to {MODEL_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--period", default="5y")
    args = parser.parse_args()
    train(args.ticker.upper(), args.period)


if __name__ == "__main__":
    main()
