from hypal_utils.sensor_data import SensorData
from pydantic import BaseModel


class CandleIngestRequest(BaseModel):
    """Запрос на приём батча свечей."""

    candles: list[SensorData]


class CandleIngestResponse(BaseModel):
    received: int
