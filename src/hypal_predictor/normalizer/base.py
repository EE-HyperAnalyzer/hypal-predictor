from abc import ABC, abstractmethod
from dataclasses import dataclass

from hypal_utils.candles import Candle_OHLC


@dataclass
class Normalizer(ABC):
    _is_fitted: bool = False

    @abstractmethod
    def fit(self, data: list[Candle_OHLC]) -> "Normalizer":
        raise NotImplementedError

    @abstractmethod
    def transform(self, data: list[Candle_OHLC]) -> list[Candle_OHLC]:
        raise NotImplementedError

    @abstractmethod
    def reverse(self, candle: Candle_OHLC) -> Candle_OHLC:
        raise NotImplementedError

    def fit_transform(self, data: list[Candle_OHLC]) -> list[Candle_OHLC]:
        self.fit(data)
        return self.transform(data)
