from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TimeframeSettings(BaseModel):
    input_horizon: int = Field(ge=1, description="Длина входного окна в свечах")
    output_horizon: int = Field(ge=1, description="Горизонт предсказания в свечах")
    rollout_multiplier: int = Field(default=1, ge=1)
    num_train_samples: int = Field(ge=10, description="Минимум агрегированных свечей для обучения")
    model_type: Literal["linear", "catboost", "transformer"] = "linear"
    critical_zone: str = Field(description="Критическая зона в виде системы неравенств")


class SensorConfigRequest(BaseModel):
    timeframes: dict[str, TimeframeSettings] = Field(
        description='Словарь таймфреймов с настройками, например {"1:m": {"input_horizon": 10, ...}, "5:m": {...}}',
        examples=[
            {
                "1:m": {
                    "input_horizon": 10,
                    "output_horizon": 5,
                    "rollout_multiplier": 1,
                    "num_train_samples": 500,
                    "model_type": "linear",
                    "critical_zone": "x < 5 or x > 35",
                }
            }
        ],
    )


class SensorConfigResponse(BaseModel):
    source: str
    sensor: str
    axis: str
    timeframes: dict[str, TimeframeSettings]
    created_at: datetime
    updated_at: datetime
