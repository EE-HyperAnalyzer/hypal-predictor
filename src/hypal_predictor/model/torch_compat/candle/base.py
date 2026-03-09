from abc import ABCMeta

import numpy as np
import torch
from hypal_utils.candles import Candle_OHLC

from hypal_predictor.model.torch_compat.base import TorchCompatibleModel
from hypal_predictor.utils import candle_to_array


class TorchCompatibleCandleModel(TorchCompatibleModel, metaclass=ABCMeta):
    def _train_eval_split(
        self, candle_seq: list[Candle_OHLC]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        k = self.input_horizon + self.output_horizon
        sequence = [candle_to_array(c) for c in candle_seq]
        windows_list = np.array([sequence[i : i + k] for i in range(len(sequence) - k + 1)])
        windows = torch.tensor(windows_list, dtype=torch.float32)

        x = windows[:, : self.input_horizon, :]
        y = windows[:, self.input_horizon :, :]

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

    def _normalize_x(self, x: list[Candle_OHLC]) -> torch.Tensor:
        x_np = np.array([[candle_to_array(candle) for candle in x]])
        x_t = torch.tensor(x_np, dtype=torch.float32)
        x_t = (x_t - self._x_mus) / self._x_std
        return x_t

    def _inverse_z(self, z: torch.Tensor) -> list[Candle_OHLC]:
        z_inv = (self._y_std * z + self._y_mus)[0]
        pred = []
        for i in range(self.output_horizon):
            pred.append(
                Candle_OHLC(
                    open=float(z_inv[i][0]), high=float(z_inv[i][1]), low=float(z_inv[i][2]), close=float(z_inv[i][3])
                )
            )
        return pred
