from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from app.services.indicators import add_indicators

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "xgboost_model.joblib"

FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA_20",
    "EMA_20",
    "RSI",
    "MACD",
    "MACD_SIGNAL",
    "VOLATILITY_20",
    "RETURN_1D",
    "RETURN_5D",
]


def build_dataset(ticker: str, period: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    df = add_indicators(df)
    df["Target"] = df["Close"].shift(-1)
    df = df.dropna().copy()
    return df


def train(ticker: str, period: str) -> dict[str, float]:
    df = build_dataset(ticker, period)
    X = df[FEATURE_COLUMNS]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = float(mean_squared_error(y_test, preds) ** 0.5)
    mae = float(mean_absolute_error(y_test, preds))

    residual_std = float(np.std(y_test - preds))
    model._residual_std = residual_std  # type: ignore[attr-defined]

    joblib.dump(model, MODEL_PATH)

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "residual_std": residual_std,
    }

    print(f"Saved model to {MODEL_PATH}")
    print(metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--period", default="5y")
    args = parser.parse_args()
    train(args.ticker.upper(), args.period)


if __name__ == "__main__":
    main()
