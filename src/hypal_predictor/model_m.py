from pathlib import Path
from typing import Optional

from hypal_utils.sensor_data import SensorData

from hypal_predictor.model import Model
from hypal_predictor.model.builtin import LinearModel
from hypal_predictor.model.config import ModelConfig
from hypal_predictor.timeframe import Timeframe
from hypal_predictor.timeframe_m import TimeframeManager, TimeframeResult


class ModelManager:
    models: dict[str, TimeframeManager]

    def __init__(self):
        self.models = {}

    def add_timeframe(self, sensor_id: str, timeframe: Timeframe):
        if sensor_id not in self.models:
            tm = TimeframeManager(model_t=LinearModel, timeframe=timeframe)
            self.models[self._get_model_id(sensor_id, timeframe)] = tm

    def update_model_config(self, sensor_id: str, timeframe: Timeframe, config: ModelConfig):
        self.models[self._get_model_id(sensor_id, timeframe)].update_model_config(config)

    def _load_from_cache(self, sensor_id: str, timeframe: str) -> Optional[Model]:
        name = f"{sensor_id}-{timeframe.replace(':', '_')}"
        path = Path("model_cache") / name
        if path.exists() and path.is_file():
            raise NotImplementedError

    def consume(self, data: SensorData) -> dict[str, TimeframeResult]:
        sensor_id = f"{data.source}:{data.sensor}:{data.axis}"
        results = {}
        for t in self.models.keys():
            if sensor_id not in t:
                continue
            tm = self.models[t]
            r = tm.consume(data)
            results[t] = r

        return results

    @staticmethod
    def _get_model_id(sensor_id: str, timeframe: Timeframe) -> str:
        return f"{sensor_id}:{timeframe}"


_model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    return _model_manager
