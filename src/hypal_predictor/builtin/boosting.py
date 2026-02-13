import numpy as np
from catboost import CatBoostRegressor
from hypal_utils.candles import Candle_OHLC

from hypal_predictor.model import Model
from hypal_predictor.normalizer import MinMaxNormalizer, Normalizer
from hypal_predictor.utils import candle_to_array, create_sequences


class BoostingModel(Model):
    normalizer: Normalizer
    input_horizont: int

    def __init__(self, input_horizont: int, **kwargs):
        super().__init__()
        self.model = CatBoostRegressor(
            loss_function="MultiRMSE",
            objective="MultiRMSE",
            eval_metric="MultiRMSE",
            **kwargs,
        )
        self.normalizer = MinMaxNormalizer()
        self.input_horizont = input_horizont

    def fit(self, x: list[Candle_OHLC]) -> "BoostingModel":
        x_norm = self.normalizer.fit_transform(x)
        x_train_seq, y_train_seq = create_sequences(
            data=x_norm, inp_seq_len=self.get_context_length(), out_seq_len=1, flatten=True
        )
        self.model.fit(x_train_seq, y_train_seq)
        self.is_fitted = True
        return self

    def predict(self, x: list[Candle_OHLC]) -> Candle_OHLC:
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        if len(x) != self.get_context_length():
            raise ValueError("Input length does not match model context length")

        x_norm = self.normalizer.transform(x)
        x_mat = np.array([candle_to_array(candle) for candle in x_norm]).reshape(-1)
        res = self.model.predict(x_mat)
        return self.normalizer.reverse(Candle_OHLC(open=res[0], high=res[1], low=res[2], close=res[3]))

    def get_context_length(self) -> int:
        return self.input_horizont
