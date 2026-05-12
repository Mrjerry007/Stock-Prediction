from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class StockResponse(BaseModel):
    ticker: str
    period: str
    data: list[dict[str, Any]]


class PredictionResponse(BaseModel):
    ticker: str
    current_price: float
    predicted_next_close: float
    predicted_return_pct: float
    interval_low: float
    interval_high: float
    direction: Literal["UP", "DOWN", "FLAT"]
    confidence: float
    model_used: str


class AlertsResponse(BaseModel):
    ticker: str
    alerts: list[dict[str, Any]]
