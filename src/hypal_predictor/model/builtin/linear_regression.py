import numpy as np
from hypal_utils.candles import Candle_OHLC
from sklearn.linear_model import LinearRegression

from hypal_predictor.model.base import Model


class LinearRegressionModel(Model):
    def __init__(self, input_horizon: int, output_horizon: int):
        super().__init__(input_horizon, output_horizon)
        self.model = LinearRegression()

    def fit(
        self,
        train_x: list[list[Candle_OHLC]],
        train_y: list[list[Candle_OHLC]],
        valid_x: list[list[Candle_OHLC]],
        valid_y: list[list[Candle_OHLC]],
    ) -> "Model":
        train_x_ = np.array([[candle.close for candle in candles] for candles in train_x])
        train_y_ = np.array([[candle.close for candle in candles] for candles in train_y])
        self.model.fit(train_x_, train_y_)
        return self

    def predict(self, x: list[Candle_OHLC]) -> list[Candle_OHLC]:
        pred = self.model.predict([[candle.close for candle in x]])[0]
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
