import numpy as np
import torch
import torch.nn as nn
from hypal_utils.candles import Candle_OHLC

from src.model import TorchModel
from src.normalizer import MinMaxNormalizer, Normalizer
from src.utils import candle_to_array


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


class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features: int, d_model=64, nhead=4, num_layers=2, dim_feedforward=256):
        super().__init__()
        self.embedding = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers, enable_nested_tensor=True)
        self.fc = nn.Linear(d_model, num_features)

    def forward(self, src):
        src = self.embedding(src)  # [batch, seq_len, d_model]
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)  # [seq_len, batch, d_model]
        output = self.transformer_encoder(src)
        output = output[-1]  # Последний токен
        return self.fc(output)  # [batch, num_features]


class TimeSeriesTransformerModel(TorchModel):
    def __init__(
        self,
        input_size: int,
        train_steps: int = 10,
        batch_size: int = 32,
        normalizer: Normalizer = MinMaxNormalizer(),
        device: str = "cpu",
    ):
        self.model = TimeSeriesTransformer(num_features=4)
        super().__init__(
            model=self.model,
            input_horizon_length=input_size,
            train_steps=train_steps,
            batch_size=batch_size,
            normalizer=normalizer,
            device=device,
        )

    def predict(self, x: list[Candle_OHLC]) -> Candle_OHLC:
        if not self.is_fitted:
            raise ValueError("Model is not fitted")

        if len(x) != self.get_context_length():
            raise ValueError("Input length does not match model context length")

        x_norm = self._normalizer.transform(x)
        x_vec = torch.tensor(np.array([candle_to_array(candle) for candle in x_norm]), dtype=torch.float32).unsqueeze(0)
        res = self.model(x_vec).detach().squeeze(0)

        return self._normalizer.reverse(Candle_OHLC(open=res[0], high=res[1], low=res[2], close=res[3]))
