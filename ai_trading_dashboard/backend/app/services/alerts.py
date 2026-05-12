from __future__ import annotations

from typing import Any

import pandas as pd


def generate_alerts(
    df: pd.DataFrame,
    ticker: str,
    price_drop_pct: float = 2.0,
    price_rise_pct: float = 2.0,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
) -> list[dict[str, Any]]:
    df = df.copy().dropna()
    if df.empty:
        return []

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    alerts: list[dict[str, Any]] = []

    day_return = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100 if prev["Close"] else 0.0

    if day_return <= -abs(price_drop_pct):
        alerts.append({
            "type": "price_drop",
            "severity": "high" if day_return <= -abs(price_drop_pct) * 1.5 else "medium",
            "message": f"{ticker} dropped {day_return:.2f}% from the previous session.",
        })

    if day_return >= abs(price_rise_pct):
        alerts.append({
            "type": "price_rise",
            "severity": "high" if day_return >= abs(price_rise_pct) * 1.5 else "medium",
            "message": f"{ticker} rose {day_return:.2f}% from the previous session.",
        })

    rsi = float(latest.get("RSI", 50.0))
    if rsi >= rsi_overbought:
        alerts.append({
            "type": "rsi_overbought",
            "severity": "medium",
            "message": f"{ticker} RSI is {rsi:.2f}, which is in the overbought zone.",
        })

    if rsi <= rsi_oversold:
        alerts.append({
            "type": "rsi_oversold",
            "severity": "medium",
            "message": f"{ticker} RSI is {rsi:.2f}, which is in the oversold zone.",
        })

    vol = float(latest.get("VOLATILITY_20", 0.0) or 0.0)
    if vol >= 0.35:
        alerts.append({
            "type": "high_volatility",
            "severity": "medium",
            "message": f"{ticker} is showing elevated 20-day volatility ({vol:.2f}).",
        })

    if latest.get("Close", 0) > latest.get("SMA_20", latest.get("Close", 0)):
        alerts.append({
            "type": "trend_up",
            "severity": "low",
            "message": f"{ticker} is trading above its 20-day SMA.",
        })
    else:
        alerts.append({
            "type": "trend_down",
            "severity": "low",
            "message": f"{ticker} is trading below its 20-day SMA.",
        })

    return alerts
