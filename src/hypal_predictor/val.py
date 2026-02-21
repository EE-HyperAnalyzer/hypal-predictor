from hypal_utils.candles import Candle_OHLC

from hypal_predictor.metrics.metrics import MAE, MSE
from hypal_predictor.model.base import Model
from hypal_predictor.utils import rollout
from hypal_predictor.utils.utils import create_sequences


def val_model(model: Model, data: list[Candle_OHLC], out_horizont: int) -> dict[str, list[float]]:
    X_test, Y_test = create_sequences(data, inp_seq_len=model.get_context_length(), out_seq_len=out_horizont)

    metrics: dict[str, list[float]] = {}
    for X, Y in zip(X_test, Y_test):
        x = [Candle_OHLC(open=x[0], close=x[1], high=x[2], low=x[3]) for x in X]
        y = [Candle_OHLC(open=y[0], close=y[1], high=y[2], low=y[3]) for y in Y]

        y_pred = rollout(model, x, out_horizont)
        for metric in [MSE, MAE]:
            m = metric(y_true=y, y_pred=y_pred)
            metrics[m.__class__.__name__] = metrics.get(m.__class__.__name__, []) + [m.calculate()]
    return metrics
