import numpy as np
from catboost import CatBoostRegressor
from hypal_utils.candles import Candle_OHLC

from hypal_predictor.model.base import Model


class CatBoostRegressorModel(Model):
    def __init__(
        self,
        input_horizon: int,
        output_horizon: int,
    ):
        super().__init__(input_horizon, output_horizon)
        self.model = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3,
            random_seed=42,
            loss_function="MultiRMSE",
            eval_metric="MultiRMSE",
            verbose=5,
            early_stopping_rounds=100,
            task_type="CPU",  # 'GPU' если есть видеокарта
        )

    def fit(
        self,
        train_x: list[list[Candle_OHLC]],
        train_y: list[list[Candle_OHLC]],
        valid_x: list[list[Candle_OHLC]],
        valid_y: list[list[Candle_OHLC]],
    ) -> "Model":
        train_x_ = np.array([[candle.close for candle in candles] for candles in train_x])
        train_y_ = np.array([[candle.close for candle in candles] for candles in train_y])
        valid_x_ = np.array([[candle.close for candle in candles] for candles in valid_x])
        valid_y_ = np.array([[candle.close for candle in candles] for candles in valid_y])

        self.model.fit(
            train_x_,
            train_y_,
            eval_set=(valid_x_, valid_y_),
            use_best_model=True,
            verbose=False,
        )
        return self

    def predict(self, x: list[Candle_OHLC]) -> list[Candle_OHLC]:
        pred = self.model.predict([[candle.close for candle in x]])
        if self.output_horizon > 1:
            pred = pred[0]
        pred_candles = [Candle_OHLC(open=pred[0], high=pred[0], low=pred[0], close=pred[0])]
        for z in pred[1:]:
            pred_candles.append(
                Candle_OHLC(
                    open=pred_candles[-1].close,
                    high=z,
                    low=z,
                    close=z,
                )
            )
        return pred_candles
