import numpy as np
import torch
import torch.nn as nn
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]  # ty:ignore[not-subscriptable]


class CandleTransformer(nn.Module):
    def __init__(
        self, num_features=4, d_model=64, nhead=4, num_layers=2, output_horizon: int = 1, dim_feedforward: int = 256
    ):
        super().__init__()
        self.num_features = num_features
        self.embedding = nn.Linear(4, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # ← главное изменение
        self.fc = nn.Linear(d_model, num_features * output_horizon)

    def forward(self, src):
        # src: [batch, input_horizon, 4]
        src = self.embedding(src)  # → [b, seq, d]
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)  # [b, seq, d]

        # Берём последний скрытый вектор и проецируем сразу на весь горизонт
        out = self.fc(memory[:, -1, :])  # [batch, 4 * output_horizon]
        out = out.view(out.size(0), -1, self.num_features)  # [batch, output_horizon, self.num_features]
        return out
