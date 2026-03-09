from abc import ABCMeta

import numpy as np
from hypal_utils.candles import Candle_OHLC

from hypal_predictor.model.scikit_compat.base import ScikitLearnCompatibleModel
from hypal_predictor.utils import candle_to_array


class ScikitLearnCompatibleNumericModel(ScikitLearnCompatibleModel, metaclass=ABCMeta):
    def _train_eval_split(self, candle_seq: list[Candle_OHLC]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        k = self.input_horizon + self.output_horizon
        sequence = [candle_to_array(c) for c in candle_seq]
        windows_list = [sequence[i : i + k] for i in range(len(sequence) - k + 1)]
        windows = np.array(windows_list)

        x = windows[:, : self.input_horizon, :]
        y = np.mean(windows[:, self.input_horizon :, :], axis=2, keepdims=True)

        tr_x, tr_y, ev_x, ev_y = self._split(x, y)

        self._x_mus = tr_x.mean(axis=0)
        self._y_mus = tr_y.mean(axis=0)
        self._x_std = tr_x.std(axis=0)
        self._y_std = tr_y.std(axis=0)

        tr_x = (tr_x - self._x_mus) / self._x_std
        tr_y = (tr_y - self._y_mus) / self._y_std
        ev_x = (ev_x - self._x_mus) / self._x_std
        ev_y = (ev_y - self._y_mus) / self._y_std

        tr_x = tr_x.reshape(tr_x.shape[0], -1)
        tr_y = tr_y.reshape(tr_y.shape[0], -1)
        ev_x = ev_x.reshape(ev_x.shape[0], -1)
        ev_y = ev_y.reshape(ev_y.shape[0], -1)

        return tr_x, tr_y, ev_x, ev_y

    def _normalize_x(self, x: list[Candle_OHLC]) -> np.ndarray:
        x_np = np.array([[candle_to_array(candle) for candle in x]])
        x_np = (x_np - self._x_mus) / self._x_std
        return x_np

    def _inverse_z(self, z: np.ndarray) -> list[Candle_OHLC]:
        z_inv = (self._y_std * z + self._y_mus)[0]
        pred = [Candle_OHLC(open=z_inv[0], high=z_inv[0], low=z_inv[0], close=z_inv[0])]
        for i in range(1, self.output_horizon):
            pred.append(Candle_OHLC(open=pred[-1].close, high=z_inv[i], low=z_inv[i], close=z_inv[i]))
        return pred
