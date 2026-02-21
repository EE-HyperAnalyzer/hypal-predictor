from hypal_utils.candles import Candle_OHLC
from hypal_utils.logger import log_info
from hypal_utils.sensor_data import SensorData

from src.model.base import Model
from src.normalizer import Normalizer


class TimeframeManager:
    model_id: str
    gathered_data: list[Candle_OHLC]

    _model: Model
    _model_factory: type[Model]

    def __init__(self, model_id: str, model_t: type[Model]):
        self._timeframe_to_model: dict[int, Model] = {}
        self.gathered_data = []
        self._model_id = model_id
        self._model_factory = model_t

    def add_timeframe(self, timeframe: str, normalizer: Normalizer, input_horizon_length: int):
        self._model = self._model_factory(normalizer=normalizer, input_horizon_length=input_horizon_length)

    def get_timeframes(self) -> list[str]:
        return ["1:s"]

    def consume(self, data: SensorData):
        match self._model.get_state():
            case Model.ModelState.GATHERING:
                log_info(f"gathering: {self._model_id} [{len(self.gathered_data)}/20]")
                self.gathered_data.append(data.candle)
                if len(self.gathered_data) > 20:
                    self._model.fit(self.gathered_data)
                    self._model.change_state(Model.ModelState.READY)

            case Model.ModelState.READY:
                ...

            case Model.ModelState.TRAINING:
                ...
