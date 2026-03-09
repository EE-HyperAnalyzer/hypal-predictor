import torch
from hypal_utils.candles import Candle_OHLC

from hypal_predictor.utils import CandleTransformer

from .base import TorchCompatibleCandleModel


class TCC_Transformer(TorchCompatibleCandleModel):
    def __init__(
        self,
        input_horizon: int,
        output_horizon: int,
        batch_size: int = 32,
        max_epochs: int = 300,
        train_size: float = 0.8,
        tolerance: float = 1e-5,
        lr_scheduler_gamma: float = 0.998,
        device: str = "cpu" if torch.cuda.is_available() else "cpu",
    ):
        model = CandleTransformer(num_features=4, output_horizon=output_horizon)
        super().__init__(
            model,
            input_horizon,
            output_horizon,
            batch_size,
            max_epochs,
            train_size,
            tolerance,
            lr_scheduler_gamma,
            device,
        )

    def _predict_raw(self, x: list[Candle_OHLC]) -> list[Candle_OHLC]:
        with torch.no_grad():
            x_t = self._normalize_x(x)
            z = self.model(x_t).cpu()
            return self._inverse_z(z)
