from dataclasses import dataclass

from hypal_utils.candles import Candle_OHLC

from hypal_predictor.metrics import MAE, MSE, Metric
from hypal_predictor.model import Model
from hypal_predictor.normalizer import MinMaxNormalizer, Normalizer
from hypal_predictor.utils import create_sequences, rollout


@dataclass
class PredictResult: ...


@dataclass
class Ok(PredictResult):
    horizont: list[Candle_OHLC]


@dataclass
class Gather(PredictResult): ...


class PredictorStream:
    model: Model
    output_horizont_size: int
    is_fitted: bool = False
    scaler: Normalizer

    def __init__(self, model: Model, output_horizont_size: int):
        self.model = model
        self.output_horizont_size = output_horizont_size

    def fit(
        self,
        data: list[Candle_OHLC],
        train_size: float = 0.8,
        train_steps: int = 100,
        batch_size: int = 32,
        lr: float = 3e-3,
        metrics: tuple[type[Metric], ...] = (MSE, MAE),
    ) -> dict[str, list[float]]:
        self.scaler = MinMaxNormalizer()
        candle_scaled_data = self.scaler.fit_transform(data)

        n = len(candle_scaled_data)
        train_size = int(n * train_size)
        train_data = candle_scaled_data[:train_size]

        self.model.fit(train_data)

        test_data = candle_scaled_data[train_size:]
        X_test, y_test = create_sequences(
            data=test_data, inp_seq_len=self.model.get_context_length(), out_seq_len=self.output_horizont_size
        )

        metric_values: dict[str, list[float]] = {}
        for x, y in zip(X_test, y_test):
            x = [Candle_OHLC(open=v[0], high=v[1], low=v[2], close=v[3]) for v in x]
            y = [Candle_OHLC(open=v[0], high=v[1], low=v[2], close=v[3]) for v in y]

            y_pred = rollout(self.model, x, self.output_horizont_size)

            for metric in metrics:
                metric_values[metric.__name__] = metric_values.get(metric.__name__, []) + [
                    metric(y, y_pred).calculate()
                ]

        self.is_fitted = True
        return metric_values

    def predict(self, data: list[Candle_OHLC]) -> list[Candle_OHLC]:
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        return rollout(self.model, data, self.output_horizont_size)
