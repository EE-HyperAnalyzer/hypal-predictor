from pathlib import Path
from typing import Optional

from hypal_utils.sensor_data import SensorData

from src.model import Model
from src.model.builtin import LinearModel
from src.normalizer import MinMaxNormalizer
from src.timeframe_m import TimeframeManager


class ModelManager:
    models: dict[str, TimeframeManager]

    def __init__(self):
        self.models = {}

    def add_timeframe(self, sensor_id: str, timeframe: str):
        assert timeframe.count(":") == 1
        timeframe_n, timeframe_det = timeframe.split(":")
        timeframe_n = int(timeframe_n)

        if sensor_id not in self.models:
            tm = TimeframeManager(model_id=sensor_id, model_t=LinearModel)
            self.models[sensor_id] = tm
            tm.add_timeframe(
                timeframe=timeframe,
                normalizer=MinMaxNormalizer(),
                input_horizon_length=2,
            )

    def _load_from_cache(self, sensor_id: str, timeframe: str) -> Optional[Model]:
        name = f"{sensor_id}-{timeframe.replace(':', '_')}"
        path = Path("model_cache") / name
        if path.exists() and path.is_file():
            raise NotImplementedError

    def consume(self, data: SensorData):
        sensor_id = f"{data.source}:{data.sensor}:{data.axis}"
        tm = self.models[sensor_id]
        tm.consume(data)


_model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    return _model_manager
