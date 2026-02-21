from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from hypal_utils.candles import Candle_OHLC

from hypal_predictor.utils import candle_to_array


@dataclass
class Metric(ABC):
    y_true: list[Candle_OHLC]
    y_pred: list[Candle_OHLC]

    def __post_init__(self):
        assert len(self.y_true) == len(self.y_pred), "y_true and y_pred must have the same length"

    @abstractmethod
    def calculate(self) -> float:
        raise NotImplementedError

    @property
    def y_true_np(self) -> np.ndarray:
        return np.array([candle_to_array(c) for c in self.y_true])

    @property
    def y_pred_np(self) -> np.ndarray:
        return np.array([candle_to_array(c) for c in self.y_pred])


class MAE(Metric):
    def calculate(self) -> float:
        return float(np.abs(self.y_true_np[:, 3] - self.y_pred_np[:, 3]).mean())


class MSE(Metric):
    def calculate(self) -> float:
        return float(((self.y_true_np[:, 3] - self.y_pred_np[:, 3]) ** 2).mean())
