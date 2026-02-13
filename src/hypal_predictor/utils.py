import numpy as np
import torch
from hypal_utils.candles import Candle_OHLC

from hypal_predictor.model import Model


def create_sequences(data: list[Candle_OHLC], inp_seq_len: int, out_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for i in range(len(data) - inp_seq_len - out_seq_len):
        x = data[i : i + inp_seq_len]
        y = data[i + inp_seq_len : i + inp_seq_len + out_seq_len]
        xs.append([[d.open, d.high, d.low, d.close] for d in x])
        ys.append([[d.open, d.high, d.low, d.close] for d in y])
    return torch.tensor(np.array(xs), dtype=torch.float32), torch.tensor(np.array(ys), dtype=torch.float32)


def rollout(model: Model, x0: torch.Tensor, k: int) -> torch.Tensor:
    x = x0.clone()
    preds = []

    for _ in range(k):
        y = model(x)
        preds.append(y)
        x = torch.cat([x[:, 1:], y.unsqueeze(1)], dim=1)

    return torch.stack(preds, dim=1)
