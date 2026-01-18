import numpy as np
from hypal_utils.candles import Candle_OHLC

from hypal_predictor.builtin import LinearModel
from hypal_predictor.predictor import PredictorStream


def main():
    model = LinearModel(3)

    predictor = PredictorStream(
        model=model,
        output_horizont_size=5,
        timeframe_s=1,
    )

    for i in range(5000):
        candle = Candle_OHLC(
            name="unknown",
            unit="unknown",
            open=10.0 + np.random.standard_normal(),
            high=20.0 + np.random.standard_normal(),
            low=5.0 + np.random.standard_normal(),
            close=15.5 + np.random.standard_normal(),
            timestamp=i,
        )
        print(f"====[ i={i} ]====")
        print(predictor.step(candle))


if __name__ == "__main__":
    main()
