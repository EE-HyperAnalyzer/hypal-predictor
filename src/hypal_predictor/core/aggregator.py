from dataclasses import dataclass, field

from hypal_utils.candles import Candle_OHLC
from hypal_utils.sensor_data import SensorData
from hypal_utils.timeframe import Timeframe


@dataclass
class CandleAggregator:
    timeframe: Timeframe
    _window_start: int | None = field(default=None)
    _candles_in_window: list[SensorData] = field(default_factory=list)

    def push(self, data: SensorData) -> SensorData | None:
        tf_sec = self.timeframe.as_seconds()
        bucket_start = (data.timestamp // tf_sec) * tf_sec

        if self._window_start is None:
            self._window_start = bucket_start
            self._candles_in_window = [data]
            return None

        if bucket_start == self._window_start:
            self._candles_in_window.append(data)
            return None

        result = self._finalize()
        self._window_start = bucket_start
        self._candles_in_window = [data]
        return result

    def flush(self) -> SensorData | None:
        if not self._candles_in_window:
            return None
        return self._finalize()

    def _finalize(self) -> SensorData:
        candles = self._candles_in_window
        first = candles[0]
        agg = Candle_OHLC(
            open=first.candle.open,
            high=max(c.candle.high for c in candles),
            low=min(c.candle.low for c in candles),
            close=candles[-1].candle.close,
        )
        return SensorData(
            source=first.source,
            sensor=first.sensor,
            axis=first.axis,
            candle=agg,
            timestamp=self._window_start,  # type: ignore[arg-type]
        )
