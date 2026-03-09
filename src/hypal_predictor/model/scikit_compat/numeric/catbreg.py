from catboost import CatBoostRegressor
from hypal_utils.candles import Candle_OHLC


from .base import ScikitLearnCompatibleNumericModel


class SLCN_CatBoostRegressorModel(ScikitLearnCompatibleNumericModel):
    def fit(self, data_seq: list[Candle_OHLC]) -> "SLCN_CatBoostRegressorModel":
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
            task_type="CPU",
        )
        train_x, train_y, eval_x, eval_y = self._train_eval_split(data_seq)
        self.model.fit(
            train_x,
            train_y,
            eval_set=(eval_x, eval_y),
            use_best_model=True,
            verbose=False,
        )
        return self

    def _predict_raw(self, x: list[Candle_OHLC]) -> list[Candle_OHLC]:
        x_np = self._normalize_x(x).reshape(1, 4 * self.input_horizon)
        pred_norm = self.model.predict(x_np).reshape(1, self.output_horizon)
        return self._inverse_z(pred_norm)
