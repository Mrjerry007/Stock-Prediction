from __future__ import annotations

import pandas as pd
import ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["EMA_20"] = ta.trend.ema_indicator(df["Close"], window=20)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    df["MACD"] = ta.trend.macd(df["Close"])
    df["MACD_SIGNAL"] = ta.trend.macd_signal(df["Close"])
    df["VOLATILITY_20"] = df["Close"].pct_change().rolling(window=20).std() * (252 ** 0.5)
    df["RETURN_1D"] = df["Close"].pct_change()
    df["RETURN_5D"] = df["Close"].pct_change(5)
    df["ROLLING_HIGH_20"] = df["Close"].rolling(window=20).max()
    df["ROLLING_LOW_20"] = df["Close"].rolling(window=20).min()

    return df
