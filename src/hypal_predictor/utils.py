import numpy as np
from hypal_utils.candles import Candle_OHLC
from hypal_utils.sensor_data import SensorData


def create_sequences(
    data: list[Candle_OHLC], inp_seq_len: int, out_seq_len: int, flatten: bool = False
) -> tuple[list[list[Candle_OHLC]], list[list[Candle_OHLC]]]:
    xs, ys = [], []
    for i in range(len(data) - inp_seq_len - out_seq_len):
        xs.append(data[i : i + inp_seq_len])
        ys.append(data[i + inp_seq_len : i + inp_seq_len + out_seq_len])

    return xs, ys


def candle_to_array(candle: Candle_OHLC) -> np.ndarray:
    return np.array([candle.open, candle.high, candle.low, candle.close])


def array_to_candle(arr: np.ndarray) -> Candle_OHLC:
    return Candle_OHLC(open=arr[0], high=arr[1], low=arr[2], close=arr[3])


# def rollout(model: Model, x0: list[Candle_OHLC], k: int) -> list[Candle_OHLC]:
#     x = deque(x0)
#     preds = []

#     for _ in range(k):
#         y = model.predict(list(x))
#         preds.append(y)
#         x.popleft()
#         x.append(y)

#     return preds


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


def to_train_valid_split(
    data: list[SensorData], inp_hor: int, out_hor: int, train_ratio: float = 0.8
) -> tuple[list[list[Candle_OHLC]], list[list[Candle_OHLC]], list[list[Candle_OHLC]], list[list[Candle_OHLC]]]:
    train_size = int(len(data) * train_ratio)
    train_data = [x.candle for x in data[:train_size]]
    valid_data = [x.candle for x in data[train_size:]]

    train_x, train_y = create_sequences(train_data, inp_hor, out_hor)
    valid_x, valid_y = create_sequences(valid_data, inp_hor, out_hor)

    return train_x, train_y, valid_x, valid_y
