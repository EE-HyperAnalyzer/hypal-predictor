from abc import ABCMeta

import numpy as np
import torch
from hypal_utils.candles import Candle_OHLC

from hypal_predictor.model.torch_compat.base import TorchCompatibleModel
from hypal_predictor.utils import candle_to_array


class TorchCompatibleShrinkedCandleModel(TorchCompatibleModel, metaclass=ABCMeta):
    def _train_eval_split(
        self, candle_seq: list[Candle_OHLC]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        k = self.input_horizon + self.output_horizon
        sequence = [candle_to_array(c) for c in candle_seq]
        windows_list = np.array([sequence[i : i + k] for i in range(len(sequence) - k + 1)])
        windows = torch.tensor(windows_list, dtype=torch.float32)

        x = windows[:, : self.input_horizon, :]
        _y = windows[:, self.input_horizon :, :]
        _h = _y[:, :, 1] - _y[:, :, 0]
        _l = _y[:, :, 0] - _y[:, :, 1]
        _d = torch.clip(_h + _l, min=1e-6)

        y = torch.stack((_h, _l, _d), dim=2)

        tr_x, tr_y, ev_x, ev_y = self._split(x, y)

        self._x_mus = tr_x.mean(dim=0)
        self._y_mus = tr_y.mean(dim=0)
        self._x_std = tr_x.std(dim=0)
        self._y_std = tr_y.std(dim=0)

        tr_x = (tr_x - self._x_mus) / self._x_std
        tr_y = (tr_y - self._y_mus) / self._y_std
        ev_x = (ev_x - self._x_mus) / self._x_std
        ev_y = (ev_y - self._y_mus) / self._y_std

        return tr_x, tr_y, ev_x, ev_y

    def _inverse_z(self, lc: Candle_OHLC, z: torch.Tensor) -> list[Candle_OHLC]:
        z_inv = (self._y_std * z + self._y_mus)[0].flatten()
        pred = [lc]
        for i in range(self.output_horizon):
            open_ = pred[-1].close
            high_ = open_ + z_inv[3 * i]
            low_ = open_ - z_inv[3 * i + 1]
            close_ = z_inv[3 * i + 2] * (low_ - high_) + high_
            pred.append(
                Candle_OHLC(
                    open=open_,
                    high=float(high_),
                    low=float(low_),
                    close=float(close_),
                )
            )
        return pred[1:]
