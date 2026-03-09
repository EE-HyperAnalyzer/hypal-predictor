from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SensorConfigRequest(BaseModel):
    timeframes: list[str] = Field(
        description='Список таймфреймов, например ["1:m", "5:m", "1:h"]',
        examples=[["1:m", "5:m"]],
    )
    input_horizon: int = Field(ge=1, description="Длина входного окна в свечах")
    output_horizon: int = Field(ge=1, description="Горизонт предсказания в свечах")
    rollout_multiplier: int = Field(default=1, ge=1)
    num_train_samples: int = Field(ge=10, description="Минимум агрегированных свечей для обучения")
    model_type: Literal["linear", "catboost", "transformer"] = "linear"
    core_api_url: str | None = None


class SensorConfigResponse(BaseModel):
    source: str
    sensor: str
    axis: str
    timeframes: list[str]
    input_horizon: int
    output_horizon: int
    rollout_multiplier: int
    num_train_samples: int
    model_type: str
    core_api_url: str | None
    created_at: datetime
    updated_at: datetime
