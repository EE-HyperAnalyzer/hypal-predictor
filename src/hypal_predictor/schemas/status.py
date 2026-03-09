from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from hypal_predictor.core.buffer import ModelState


class TimeframeMetrics(BaseModel):
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None


class TimeframeStatus(BaseModel):
    state: ModelState
    candles_gathered: int
    num_train_samples: int
    last_trained_at: Optional[datetime] = None
    last_predicted_at: Optional[datetime] = None
    is_in_critical_zone: bool = False
    metrics: Optional[TimeframeMetrics] = None


class SensorStatusResponse(BaseModel):
    source: str
    sensor: str
    axis: str
    timeframes: dict[str, TimeframeStatus]
