import random

from hypal_utils.candles import Candle_OHLC

from hypal_predictor.builtin import BoostingModel, LinearModel, TimeSeriesTransformerModel


def test_linear_model():
    random.seed(52)

    data = [
        Candle_OHLC(
            open=1 + random.random(), high=2 + random.random(), low=3 + random.random(), close=4 + random.random()
        )
        for _ in range(10)
    ]
    context = 5
    model = LinearModel(input_size=context)
    model.fit(data)
    model.predict(data[:context])
    assert model.get_context_length() == context


def test_transformer_model():
    random.seed(52)

    data = [
        Candle_OHLC(
            open=1 + random.random(), high=2 + random.random(), low=3 + random.random(), close=4 + random.random()
        )
        for _ in range(10)
    ]
    context = 5
    model = TimeSeriesTransformerModel(input_size=context)
    model.fit(data)
    model.predict(data[:context])
    assert model.get_context_length() == context


def test_boosting_model():
    random.seed(52)

    data = [
        Candle_OHLC(
            open=1 + random.random(), high=2 + random.random(), low=3 + random.random(), close=4 + random.random()
        )
        for _ in range(8)
    ]
    context = 5
    model = BoostingModel(input_horizont=context)
    model.fit(data)
    model.predict(data[:context])
    assert model.get_context_length() == context
