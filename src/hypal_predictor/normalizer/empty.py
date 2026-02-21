from hypal_utils.candles import Candle_OHLC

from .base import Normalizer


class EmptyNormalizer(Normalizer):
    def fit(self, data: list[Candle_OHLC]) -> "EmptyNormalizer":
        self._is_fitted = True
        return self

    def transform(self, data: list[Candle_OHLC]) -> list[Candle_OHLC]:
        if not self._is_fitted:
            raise ValueError("Normalizer is not fitted")

        return data

    def reverse(self, candle: Candle_OHLC) -> Candle_OHLC:
        if not self._is_fitted:
            raise ValueError("Normalizer is not fitted")

        return candle
