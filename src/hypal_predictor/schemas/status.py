from typing import Optional

from pydantic import BaseModel

from hypal_predictor.core.buffer import ModelState


class TimeframeMetrics(BaseModel):
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None


class TimeframeStatus(BaseModel):
    state: ModelState
    model_type: str
    input_horizon: int
    output_horizon: int
    rollout_multiplier: int
    critical_zone: str
    candles_gathered: int
    num_train_samples: int
    is_in_critical_zone: bool


class SensorStatusResponse(BaseModel):
    source: str
    sensor: str
    axis: str
    timeframes: dict[str, TimeframeStatus]
