from abc import ABCMeta

import numpy as np
import torch
from hypal_utils.candles import Candle_OHLC
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from hypal_predictor.model.base import BaseModel
from hypal_predictor.utils import candle_to_array


class TorchCompatibleModel(BaseModel[torch.Tensor], metaclass=ABCMeta):
    model: torch.nn.Module
    batch_size: int
    lr_scheduler_gamma: float
    max_epochs: int
    tolerance: float

    def __init__(
        self,
        model: torch.nn.Module,
        input_horizon: int,
        output_horizon: int,
        batch_size: int = 256,
        max_epochs: int = 100,
        train_size: float = 0.8,
        tolerance: float = 1e-5,
        lr_scheduler_gamma: float = 0.998,
        device: str = "cpu" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(input_horizon, output_horizon, train_size)
        self.model = model
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.device = device

    def _split(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(x) == len(y)
        _n = int(len(x) * self.train_size)
        train_x = x[:_n, ...]
        eval_x = x[_n:, ...]
        train_y = y[:_n, ...]
        eval_y = y[_n:, ...]
        return train_x, train_y, eval_x, eval_y

    def fit(
        self,
        data_seq: list[Candle_OHLC],
    ) -> "BaseModel":
        train_x, train_y, eval_x, eval_y = self._train_eval_split(data_seq)

        train_dataset = TensorDataset(train_x, train_y)
        eval_dataset = TensorDataset(eval_x, eval_y)

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters())
        loss_fn = torch.nn.MSELoss()
        lr_s = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.lr_scheduler_gamma)

        train_pbar = tqdm(range(self.max_epochs), leave=False)
        for epoch in train_pbar:
            self.model.train()
            total_loss = []
            for batch_X, batch_y in train_dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = loss_fn(output, batch_y)
                total_loss.append(loss.item())
                loss.backward()
                optimizer.step()

            train_loss_mean = np.mean(total_loss)

            if train_loss_mean < self.tolerance:
                train_pbar.set_description("Early stopping")
                break

            lr_s.step()

            self.model.eval()
            eval_total_loss = []
            with torch.no_grad():
                for batch_X, batch_y in eval_dataloader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    output = self.model(batch_X)

                    loss = loss_fn(output, batch_y)
                    eval_total_loss.append(loss.item())

            eval_mean_loss = np.mean(eval_total_loss)

            train_pbar.set_description(
                f"Epoch {epoch + 1}. Train Loss: {train_loss_mean:.7f}. Eval Loss: {eval_mean_loss:.7f}"
            )

        self.model.eval()
        return self

    def _normalize_x(self, x: list[Candle_OHLC]) -> torch.Tensor:
        x_np = np.array([[candle_to_array(candle) for candle in x]])
        x_t = torch.tensor(x_np, dtype=torch.float32)
        x_t = (x_t - self._x_mus) / self._x_std
        return x_t
