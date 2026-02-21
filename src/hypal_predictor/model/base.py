from abc import ABC, abstractmethod

from hypal_utils.candles import Candle_OHLC

from hypal_predictor.normalizer import Normalizer


class Model(ABC):
    _input_horizon_length: int
    _normalizer: Normalizer

    def __init__(self, normalizer: Normalizer, input_horizon_length: int):
        self._normalizer = normalizer
        self._input_horizon_length = input_horizon_length

    @abstractmethod
    def fit(self, x: list[Candle_OHLC]) -> "Model":
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: list[Candle_OHLC]) -> Candle_OHLC:
        raise NotImplementedError

    def get_context_length(self) -> int:
        return self._input_horizon_length
