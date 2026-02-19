from hypal_utils.candles import Candle_OHLC

from hypal_predictor.model.base import Model
from hypal_predictor.normalizer import Normalizer
from hypal_predictor.utils import timeframe_to_sec


class ModelManager:
    model_id: str
    gathered_data: list[Candle_OHLC]

    _timeframe_to_model: dict[int, Model]
    _model_factory: type[Model]

    def __init__(self, model_id: str, model_t: type[Model]):
        self._timeframe_to_model: dict[int, Model] = {}
        self._model_id = model_id
        self._model_factory = model_t

    def add_timeframe(self, timeframe: str, normalizer: Normalizer, input_horizon_lenght: int):
        timeframe_sec = timeframe_to_sec(timeframe)
        if timeframe_sec in self._timeframe_to_model:
            raise RuntimeError("You have passed to different timeframes with the same amount of delta-seconds")

        self._timeframe_to_model[timeframe_sec] = self._model_factory(
            normalizer=normalizer, input_horizon_lenght=input_horizon_lenght
        )

    def get_timeframes(self) -> list[int]:
        return list(self._timeframe_to_model.keys())
