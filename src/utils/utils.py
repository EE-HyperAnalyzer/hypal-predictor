from collections import deque

import numpy as np
from hypal_utils.candles import Candle_OHLC

from src.model import Model


def create_sequences(
    data: list[Candle_OHLC], inp_seq_len: int, out_seq_len: int, flatten: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(len(data) - inp_seq_len - out_seq_len):
        x = data[i : i + inp_seq_len]
        y = data[i + inp_seq_len : i + inp_seq_len + out_seq_len]
        xs.append([[d.open, d.high, d.low, d.close] for d in x])
        ys.append([[d.open, d.high, d.low, d.close] for d in y])

    x, y = np.array(xs), np.array(ys)
    if flatten:
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)

    return x, y


def candle_to_array(candle: Candle_OHLC) -> np.ndarray:
    return np.array([candle.open, candle.high, candle.low, candle.close])


def rollout(model: Model, x0: list[Candle_OHLC], k: int) -> list[Candle_OHLC]:
    x = deque(x0)
    preds = []

    for _ in range(k):
        y = model.predict(list(x))
        preds.append(y)
        x.popleft()
        x.append(y)

    return preds


def timeframe_to_sec(timeframe: str) -> int:
    assert timeframe.count(":") == 1

    n, t = timeframe.split(":")
    n = int(n)

    match t:
        case "s":
            return n * 1
        case "m":
            return n * 60
        case "h":
            return n * 3600
        case _:
            raise RuntimeError(f"Unknown timeframe: {t}")
