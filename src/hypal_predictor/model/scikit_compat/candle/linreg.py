from hypal_utils.candles import Candle_OHLC
from sklearn.linear_model import LinearRegression

from .base import ScikitLearnCompatibleCandleModel


class SLCC_LinearRegressionModel(ScikitLearnCompatibleCandleModel):
    model: LinearRegression

    def fit(self, data_seq: list[Candle_OHLC]) -> "SLCC_LinearRegressionModel":
        self.model = LinearRegression()
        train_x, train_y, *_ = self._train_eval_split(data_seq)
        self.model.fit(train_x, train_y)
        return self

    def _predict_raw(self, x: list[Candle_OHLC]) -> list[Candle_OHLC]:
        x_np = self._normalize_x(x).reshape(-1, 4 * self.input_horizon)
        pred_norm = self.model.predict(x_np).reshape(-1, self.output_horizon, 4)
        return self._inverse_z(pred_norm)
