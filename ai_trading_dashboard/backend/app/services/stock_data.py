from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_history(ticker: str, period: str = "6mo") -> pd.DataFrame:
    ticker = ticker.upper().strip()
    df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.index.name = "Date"
    return df
