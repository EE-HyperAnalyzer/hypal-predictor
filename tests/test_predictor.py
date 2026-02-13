import random

from hypal_utils.candles import Candle_OHLC

from hypal_predictor.predictor import PredictorStream


def get_data(k: int) -> list[Candle_OHLC]:
    random.seed(52)

    return [
        Candle_OHLC(
            open=1 + random.random(), high=2 + random.random(), low=3 + random.random(), close=4 + random.random()
        )
        for _ in range(k)
    ]


def test_predictor_with_linear_model():
    from hypal_predictor.builtin import LinearModel

    context = 5
    pred = PredictorStream(
        model=LinearModel(context),
        output_horizont_size=2,
    )
    data = get_data(100)
    pred.fit(data)
    pred.predict(data[:context])


def test_predictor_with_transformer_model():
    from hypal_predictor.builtin import TimeSeriesTransformerModel

    context = 5
    pred = PredictorStream(
        model=TimeSeriesTransformerModel(context),
        output_horizont_size=2,
    )
    data = get_data(100)
    pred.fit(data)
    pred.predict(data[:context])


def test_predictor_with_boosting_model():
    from hypal_predictor.builtin import BoostingModel

    context = 5
    pred = PredictorStream(
        model=BoostingModel(context),
        output_horizont_size=2,
    )
    data = get_data(100)
    pred.fit(data)
    pred.predict(data[:context])
