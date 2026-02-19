from abc import ABC, abstractmethod
from enum import Enum, auto

from hypal_utils.candles import Candle_OHLC
from hypal_utils.logger import log_info

from hypal_predictor.normalizer import Normalizer


class Model(ABC):
    class ModelState(Enum):
        GATHERING = auto()
        TRAINING = auto()
        READY = auto()

    _input_horizon_length: int
    _normalizer: Normalizer
    _state: ModelState

    def __init__(self, normalizer: Normalizer, input_horizon_length: int):
        self._normalizer = normalizer
        self._input_horizon_length = input_horizon_length
        self._state = self.ModelState.GATHERING

    @abstractmethod
    def fit(self, x: list[Candle_OHLC]) -> "Model":
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: list[Candle_OHLC]) -> Candle_OHLC:
        raise NotImplementedError

    def get_context_length(self) -> int:
        return self._input_horizon_length

    def get_state(self) -> ModelState:
        return self._state

    def change_state(self, to: ModelState):
        log_info(f"Changed state: {self._state} -> {to}")
        self._state = to
