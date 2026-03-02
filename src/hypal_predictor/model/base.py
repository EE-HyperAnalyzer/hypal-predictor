from abc import ABC, abstractmethod

import numpy as np
from hypal_utils.candles import Candle_OHLC
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score

from hypal_predictor.critical_zone import ZoneRule
from hypal_predictor.utils import candle_to_array


class Model(ABC):
    input_horizon: int
    output_horizon: int

    def __init__(self, input_horizon: int, output_horizon: int) -> None:
        self.input_horizon = input_horizon
        self.output_horizon = output_horizon

    @abstractmethod
    def fit(
        self,
        train_x: list[list[Candle_OHLC]],
        train_y: list[list[Candle_OHLC]],
        valid_x: list[list[Candle_OHLC]],
        valid_y: list[list[Candle_OHLC]],
    ) -> "Model":
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: list[Candle_OHLC]) -> list[Candle_OHLC]:
        raise NotImplementedError

    def eval_regression(self, val_x: list[list[Candle_OHLC]], val_y: list[list[Candle_OHLC]]) -> dict[str, float]:
        true_ = []
        pred_ = []
        for x, y in zip(val_x, val_y, strict=True):
            z = self.predict(x)
            true_rgr = np.array([candle_to_array(d) for d in y])
            pred_rgr = np.array([candle_to_array(d) for d in z])
            true_.append(true_rgr)
            pred_.append(pred_rgr)
        true_ = np.concatenate(true_)
        pred_ = np.concatenate(pred_)

        r2 = r2_score(true_, pred_)
        mse = mean_squared_error(true_, pred_)
        mae = mean_absolute_error(true_, pred_)
        return {"r2": r2, "mse": mse, "mae": mae}

    def eval_classification(
        self, val_x: list[list[Candle_OHLC]], val_y: list[list[Candle_OHLC]], critical_zone: ZoneRule
    ) -> dict[str, object]:
        true_: list[bool] = []
        pred_: list[bool] = []
        for x, y in zip(val_x, val_y, strict=True):
            z = self.predict(x)

            true_cls = any(critical_zone.is_satisfied(candle) for candle in y)
            pred_cls = any(critical_zone.is_satisfied(candle) for candle in z)

            true_.append(true_cls)
            pred_.append(pred_cls)

        report = classification_report(true_, pred_, output_dict=True, zero_division=0.0)
        return report
