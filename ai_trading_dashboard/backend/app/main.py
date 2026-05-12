from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.ml.predict import model_status, predict_bundle
from app.services.alerts import generate_alerts
from app.services.indicators import add_indicators
from app.services.schemas import AlertsResponse, PredictionResponse, StockResponse
from app.services.stock_data import fetch_history

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent.parent / "frontend"

app = FastAPI(title="AI Trading Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home() -> dict[str, str]:
    return {"message": "AI Trading Dashboard API", "status": "running"}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "model_status": model_status()}


@app.get("/stock/{ticker}", response_model=StockResponse)
def get_stock(ticker: str, period: str = "6mo") -> StockResponse:
    df = fetch_history(ticker, period=period)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No market data found for {ticker}")

    df = add_indicators(df)
    df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    records = df.replace({np.nan: None}).to_dict(orient="records")
    return StockResponse(ticker=ticker.upper(), period=period, data=records)


@app.get("/predict/{ticker}", response_model=PredictionResponse)
def predict_stock(ticker: str, period: str = "2y") -> PredictionResponse:
    df = fetch_history(ticker, period=period)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No market data found for {ticker}")

    df = add_indicators(df)
    df["Target"] = df["Close"].shift(-1)
    bundle = predict_bundle(df, ticker=ticker.upper())

    latest = bundle["latest"]
    return PredictionResponse(
        ticker=ticker.upper(),
        current_price=float(latest["Close"]),
        predicted_next_close=float(bundle["predicted_next_close"]),
        predicted_return_pct=float(bundle["predicted_return_pct"]),
        interval_low=float(bundle["interval_low"]),
        interval_high=float(bundle["interval_high"]),
        direction=bundle["direction"],
        confidence=float(bundle["confidence"]),
        model_used=bundle["model_used"],
    )


@app.get("/alerts/{ticker}", response_model=AlertsResponse)
def get_alerts(
    ticker: str,
    period: str = "6mo",
    price_drop_pct: float = 2.0,
    price_rise_pct: float = 2.0,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
) -> AlertsResponse:
    df = fetch_history(ticker, period=period)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No market data found for {ticker}")

    df = add_indicators(df)
    alerts = generate_alerts(
        df,
        ticker=ticker.upper(),
        price_drop_pct=price_drop_pct,
        price_rise_pct=price_rise_pct,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
    )
    return AlertsResponse(ticker=ticker.upper(), alerts=alerts)


@app.websocket("/ws/{ticker}")
async def stream_stock(websocket: WebSocket, ticker: str) -> None:
    await websocket.accept()
    try:
        while True:
            df = fetch_history(ticker, period="1mo")
            if df.empty:
                await websocket.send_json({"error": f"No market data found for {ticker}"})
            else:
                df = add_indicators(df)
                latest = df.iloc[-1]
                payload = {
                    "ticker": ticker.upper(),
                    "timestamp": pd.Timestamp.utcnow().isoformat(),
                    "close": float(latest["Close"]),
                    "volume": float(latest["Volume"]),
                    "rsi": float(latest.get("RSI", 0.0)),
                    "sma_20": float(latest.get("SMA_20", 0.0)),
                    "ema_20": float(latest.get("EMA_20", 0.0)),
                }
                await websocket.send_json(payload)
            import asyncio
            await asyncio.sleep(15)
    except WebSocketDisconnect:
        return
