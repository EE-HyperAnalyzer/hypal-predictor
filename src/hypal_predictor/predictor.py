from collections import deque
from dataclasses import dataclass

import torch
from hypal_utils.candles import Candle_OHLC
from numpy import sign

from hypal_predictor.ema import EMA
from hypal_predictor.model import Model


@dataclass
class PredictResult: ...


@dataclass
class Ok(PredictResult):
    horizont: list[Candle_OHLC]


@dataclass
class Gather(PredictResult): ...


class PredictorStream:
    _model: Model
    _output_horizont_size: int
    _timeframe_s: int
    _loss_ema: EMA

    def __init__(self, model: Model, output_horizont_size: int, timeframe_s: int):
        self.buffer: deque[Candle_OHLC] = deque()

        self._model = model
        self._optimizer = torch.optim.Adam(self._model.parameters())
        self._loss_fn = torch.nn.MSELoss()

        self._output_horizont_size = output_horizont_size
        self._timeframe_s = timeframe_s
        self._loss_ema = EMA(0.9)

    def step(self, new_candle: Candle_OHLC) -> PredictResult:
        self.buffer.append(new_candle)

        if len(self.buffer) < self._model.get_context_length() + 2:
            return Gather()

        # Online step
        online_base_candle = self.buffer.popleft()
        temp_vec_norm = self._normalize_candles(online_base_candle, list(self.buffer))
        *x_vec_norm, y_norm_true = temp_vec_norm
        x_vec_norm = torch.flatten(torch.FloatTensor([[c.open, c.high, c.low, c.close] for c in x_vec_norm]))
        y_norm_true = torch.FloatTensor([y_norm_true.open, y_norm_true.high, y_norm_true.low, y_norm_true.close])

        self._optimizer.zero_grad()
        y_norm_pred = self._model(x_vec_norm)
        loss = self._loss_fn(y_norm_pred, y_norm_true)
        loss_val = self._loss_ema.update(loss.item())
        print(loss_val)
        loss.backward()
        self._optimizer.step()

        # Predict step
        base_candle, *input_horizont = list(self.buffer)

        input_horizont_normalized = self._normalize_candles(base_candle, input_horizont)
        predicted_horizont_normalized = self.predict(input_horizont_normalized)
        predicted_horizont = self._denormalize_candles(base_candle, predicted_horizont_normalized)

        return Ok(predicted_horizont)

    def predict(self, input_horizont: list[Candle_OHLC]) -> list[Candle_OHLC]:
        assert len(input_horizont) == self._model.get_context_length()
        current_input = deque(input_horizont)
        pred_horizont = []

        for i in range(self._output_horizont_size):
            x = torch.flatten(torch.FloatTensor([[c.open, c.high, c.low, c.close] for c in current_input]))
            pred = self._model(x)
            # print(i, [c.close for c in current_input], pred[3].item())

            last_candle = current_input[-1]
            pred_candle = Candle_OHLC(
                name=last_candle.name,
                open=pred[0],
                high=pred[1],
                low=pred[2],
                close=pred[3],
            )
            pred_horizont.append(pred_candle)
            current_input.popleft()
            current_input.append(pred_candle)

        return pred_horizont

    @staticmethod
    def _normalize_candles(base_candle: Candle_OHLC, candles: list[Candle_OHLC]) -> list[Candle_OHLC]:
        result: list[Candle_OHLC] = []

        last_candle = base_candle
        for curr_candle in candles:
            c = max(abs(last_candle.close), 1e-12) * sign(last_candle.close)

            norm_candle = Candle_OHLC(
                name=curr_candle.name,
                open=(curr_candle.open - c) / c,
                high=(curr_candle.high - c) / c,
                low=(curr_candle.low - c) / c,
                close=(curr_candle.close - c) / c,
            )

            result.append(norm_candle)

        return result

    @staticmethod
    def _denormalize_candles(base_candle: Candle_OHLC, candles: list[Candle_OHLC]) -> list[Candle_OHLC]:
        result: list[Candle_OHLC] = []

        last_candle = base_candle
        for curr_candle in candles:
            c = max(abs(last_candle.close), 1e-12) * sign(last_candle.close)

            denorm_candle = Candle_OHLC(
                name=curr_candle.name,
                open=c * curr_candle.open + c,
                high=c * curr_candle.high + c,
                low=c * curr_candle.low + c,
                close=c * curr_candle.close + c,
            )

            result.append(denorm_candle)

        return result
