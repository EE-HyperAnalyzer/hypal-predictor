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
    def reverse(self, data: list[Candle_OHLC]) -> list[Candle_OHLC]:
        raise NotImplementedError

    def fit_transform(self, data: list[Candle_OHLC]) -> list[Candle_OHLC]:
        self.fit(data)
        return self.transform(data)


@dataclass
class MinMaxNormalizer(Normalizer):
    min_value: float | None = None
    max_value: float | None = None

    def fit(self, data: list[Candle_OHLC]) -> "MinMaxNormalizer":
        self.min_value = min(candle.low for candle in data)
        self.max_value = max(candle.high for candle in data)
        self._is_fitted = True
        return self

    def transform(self, data: list[Candle_OHLC]) -> list[Candle_OHLC]:
        if not self._is_fitted:
            raise ValueError("Normalizer not fitted")

        assert self.min_value is not None, "min_value is None"
        assert self.max_value is not None, "max_value is None"

        result = []
        delta = self.max_value - self.min_value
        for candle in data:
            result.append(
                Candle_OHLC(
                    open=(candle.open - self.min_value) / delta,
                    high=(candle.high - self.min_value) / delta,
                    low=(candle.low - self.min_value) / delta,
                    close=(candle.close - self.min_value) / delta,
                )
            )

        return result

    def reverse(self, data: list[Candle_OHLC]) -> list[Candle_OHLC]:
        if not self._is_fitted:
            raise ValueError("Normalizer not fitted")

        assert self.min_value is not None, "min_value is None"
        assert self.max_value is not None, "max_value is None"

        result = []
        delta = self.max_value - self.min_value
        for candle in data:
            result.append(
                Candle_OHLC(
                    open=candle.open * delta + self.min_value,
                    high=candle.high * delta + self.min_value,
                    low=candle.low * delta + self.min_value,
                    close=candle.close * delta + self.min_value,
                )
            )

        return result
