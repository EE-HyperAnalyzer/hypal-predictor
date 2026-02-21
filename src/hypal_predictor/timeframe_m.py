from enum import Enum
from typing import Optional

from hypal_utils.candles import Candle_OHLC
from hypal_utils.logger import log_debug, log_info
from hypal_utils.sensor_data import SensorData

from hypal_predictor.model.base import Model
from hypal_predictor.model.config import ModelConfig
from hypal_predictor.model_state import ModelState
from hypal_predictor.timeframe import Timeframe
from hypal_predictor.utils import rollout


class TimeframeResult(Enum):
    NONE = 0
    DETECTED = 1
    RESOLVED = 2


class TimeframeManager:
    """Minimal abstraction on model"""

    _buffer: list[Candle_OHLC]
    _config: Optional[ModelConfig]
    _t: int

    _last_usage: int
    _model: Model
    _model_factory: type[Model]
    _timeframe: Timeframe
    _state: ModelState
    _anomaly_detected: bool

    def __init__(self, model_t: type[Model], timeframe: Timeframe):
        self._timeframe_to_model: dict[int, Model] = {}
        self._buffer = [Candle_OHLC(open=0.0, high=0.0, low=0.0, close=0.0)]
        self._t = 0
        self._model_factory = model_t
        self._timeframe = timeframe
        self._config = None
        self._state = ModelState.INITIALIZING
        self._anomaly_detected = False

    def get_timeframe(self) -> Timeframe:
        return self._timeframe

    def update_model_config(self, config: ModelConfig):
        if self._config is None:
            self._config = config
            self._change_state(ModelState.GATHERING)
            self._model = self._model_factory(
                normalizer=self._config.normalizer, input_horizon_length=self._config.input_horizont_length
            )
        else:
            retrain_attributes = ["candles_to_train", "input_horizon_length", "normalizer", "train_size"]
            if any(getattr(self._config, attr) != getattr(config, attr) for attr in retrain_attributes):
                self._change_state(ModelState.GATHERING)
                self._model = self._model_factory(
                    normalizer=config.normalizer, input_horizon_length=config.input_horizont_length
                )
            self._config = config

    def consume(self, data: SensorData) -> TimeframeResult:
        if self._config is None and self._state != ModelState.INITIALIZING:
            self._change_state(ModelState.INITIALIZING)

        if self._config and self._state == ModelState.INITIALIZING:
            self._buffer = [Candle_OHLC(open=0.0, high=0.0, low=0.0, close=0.0)]
            self._t = 0
            self._change_state(ModelState.GATHERING)

        match self._state:
            case ModelState.INITIALIZING:
                log_info("Model not initialized, skipping this candle...")

            case ModelState.READY:
                assert self._config and self._model
                self._buffer.append(data.candle)
                if len(self._buffer) == self._model.get_context_length():
                    y_pred = rollout(self._model, self._buffer, self._config.output_horizont_length)
                    self._buffer.pop(0)
                    if self._detect_anomaly(y_pred):
                        if self._anomaly_detected is False:
                            log_debug("Anomaly detected!")
                            self._anomaly_detected = True
                            return TimeframeResult.DETECTED
                    else:
                        if self._anomaly_detected:
                            log_debug("Anomaly resolved!")
                            self._anomaly_detected = False
                            return TimeframeResult.RESOLVED

            case ModelState.TRAINING:
                raise RuntimeError("unk")

            case ModelState.GATHERING:
                assert self._config

                self._buffer[-1] = Candle_OHLC(
                    open=self._buffer[-1].open,
                    high=max(self._buffer[-1].high, data.candle.high),
                    low=min(self._buffer[-1].low, data.candle.low),
                    close=data.candle.close,
                )
                self._t += 1

                if self._t % self._timeframe.as_seconds() == 0:
                    self._buffer.append(Candle_OHLC(open=0.0, high=0.0, low=0.0, close=0.0))
                    self._t = 0

                if len(self._buffer) > self._config.candles_to_train:
                    self._model.fit(self._buffer)
                    self._change_state(ModelState.READY)
                    self._buffer = []

        return TimeframeResult.NONE

    def _change_state(self, state: ModelState):
        log_info(f"Changing state: {self._state} -> {state}")
        self._state = state

    def _detect_anomaly(self, y_pred: list[Candle_OHLC]):
        assert self._config
        for candle in y_pred:
            if candle.high > self._config.anomaly_threshold:
                return True
        return False
