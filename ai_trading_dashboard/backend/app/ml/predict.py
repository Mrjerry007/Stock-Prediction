from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "models"
XGB_MODEL_PATH = MODEL_DIR / "xgboost_model.joblib"

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


def model_status() -> dict[str, Any]:
    return {
        "xgboost_model_exists": XGB_MODEL_PATH.exists(),
        "xgboost_model_path": str(XGB_MODEL_PATH),
    }


def _latest_row(df: pd.DataFrame) -> pd.Series:
    clean = df.dropna()
    if clean.empty:
        raise ValueError("Not enough data after indicator calculation.")
    return clean.iloc[-1]


def _fallback_prediction(df: pd.DataFrame) -> dict[str, Any]:
    latest = _latest_row(df)
    returns = df["Close"].pct_change().dropna()
    vol = float(returns.tail(20).std() or 0.01)
    trend = float((df["Close"].tail(5).mean() / df["Close"].tail(20).mean()) - 1.0) if len(df) >= 20 else 0.0
    predicted = float(latest["Close"] * (1 + np.clip(trend * 0.5, -0.03, 0.03)))
    interval = max(latest["Close"] * vol * 1.96, latest["Close"] * 0.01)
    direction = "UP" if predicted > latest["Close"] * 1.002 else "DOWN" if predicted < latest["Close"] * 0.998 else "FLAT"
    confidence = float(np.clip(1 - min(vol * 5, 0.85), 0.15, 0.85))
    return {
        "latest": latest,
        "predicted_next_close": predicted,
        "predicted_return_pct": ((predicted / float(latest["Close"])) - 1) * 100,
        "interval_low": predicted - interval,
        "interval_high": predicted + interval,
        "direction": direction,
        "confidence": confidence,
        "model_used": "heuristic_fallback",
    }


def predict_bundle(df: pd.DataFrame, ticker: str | None = None) -> dict[str, Any]:
    clean = df.copy().dropna()
    if clean.empty:
        return _fallback_prediction(df)

    latest = _latest_row(clean)

    if XGB_MODEL_PATH.exists():
        model = joblib.load(XGB_MODEL_PATH)
        X = clean[FEATURE_COLUMNS].tail(1)
        predicted = float(model.predict(X)[0])

        residual_std = float((clean["Target"] - model.predict(clean[FEATURE_COLUMNS])).std() or clean["Close"].pct_change().std() * float(latest["Close"]))
        predicted_return = ((predicted / float(latest["Close"])) - 1) * 100
        interval = max(1.96 * residual_std, float(latest["Close"]) * 0.01)
        direction = "UP" if predicted > latest["Close"] * 1.002 else "DOWN" if predicted < latest["Close"] * 0.998 else "FLAT"
        confidence = float(np.clip(1 - min(abs(predicted_return) / 10, 0.8), 0.2, 0.95))
        return {
            "latest": latest,
            "predicted_next_close": predicted,
            "predicted_return_pct": predicted_return,
            "interval_low": predicted - interval,
            "interval_high": predicted + interval,
            "direction": direction,
            "confidence": confidence,
            "model_used": "xgboost",
        }

    return _fallback_prediction(clean)
