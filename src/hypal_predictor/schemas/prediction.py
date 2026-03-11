from hypal_utils.candles import Candle_OHLC
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    source: str
    sensor: str
    axis: str
    predictions: dict[str, list[Candle_OHLC]]
