import numpy as np
import torch
import torch.nn as nn
from hypal_utils.candles import Candle_OHLC

from hypal_predictor.model import TorchModel
from hypal_predictor.utils import candle_to_array


class LinearModel(TorchModel):
    def __init__(self, input_size: int, train_steps: int = 10, batch_size: int = 32):
        self.model = nn.Sequential(nn.Linear(input_size * 4, 4))
        super().__init__(self.model, input_size, train_steps, batch_size)

    def predict(self, x: list[Candle_OHLC]) -> Candle_OHLC:
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        if len(x) != self.get_context_length():
            raise ValueError("Input length does not match model context length")

        x_norm = self.normalizer.transform(x)
        x_vec = torch.tensor(np.array([candle_to_array(candle) for candle in x_norm]), dtype=torch.float32).view(-1)
        res = self.model(x_vec).detach()
        return self.normalizer.reverse(Candle_OHLC(open=res[0], high=res[1], low=res[2], close=res[3]))

    def get_context_length(self) -> int:
        return self.input_size

    @staticmethod
    def _process_batch(x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)
