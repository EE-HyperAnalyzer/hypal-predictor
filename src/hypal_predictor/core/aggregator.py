from dataclasses import dataclass, field

from hypal_utils.candles import Candle_OHLC
from hypal_utils.sensor_data import SensorData

from hypal_predictor.timeframe import Timeframe


@dataclass
class CandleAggregator:
    """
    Накапливает сырые свечи и выдаёт агрегированные свечи таймфрейма.

    Принцип работы:
    - Каждая входящая SensorData имеет timestamp (unix seconds).
    - Свечи группируются в окна размером timeframe.as_seconds().
    - Когда приходит свеча с timestamp >= window_start + tf_seconds,
      текущее окно финализируется и возвращается как агрегированная SensorData.
    - Не хранит данные о source/sensor/axis — это ответственность вызывающей стороны.
    """

    timeframe: Timeframe
    _window_start: int | None = field(default=None)
    _candles_in_window: list[SensorData] = field(default_factory=list)

    def push(self, data: SensorData) -> SensorData | None:
        """
        Принимает новую сырую свечу.
        Возвращает агрегированную SensorData, если текущее окно завершилось, иначе None.
        """
        tf_sec = self.timeframe.as_seconds()

        if self._window_start is None:
            self._window_start = data.timestamp
            self._candles_in_window = [data]
            return None

        if data.timestamp < self._window_start + tf_sec:
            # Свеча принадлежит текущему окну
            self._candles_in_window.append(data)
            return None

        # Текущее окно завершилось — финализируем
        result = self._finalize()
        # Начинаем новое окно с текущей свечой
        self._window_start = data.timestamp
        self._candles_in_window = [data]
        return result

    def flush(self) -> SensorData | None:
        """Принудительно финализировать текущее окно (если есть данные)."""
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
