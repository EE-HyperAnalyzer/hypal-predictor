from typing import Optional

from hypal_utils.candles import Candle_OHLC
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    source: str
    sensor: str
    axis: str
    predictions: dict[str, list[Candle_OHLC]]  # tf → list[Candle_OHLC]


class SignalResponse(BaseModel):
    source: str
    sensor: str
    axis: str
    is_critical: bool
    triggered_timeframes: list[str]
    timestamp: Optional[int] = None
    predictions: dict[str, list[Candle_OHLC]]
