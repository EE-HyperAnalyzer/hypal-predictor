from abc import ABC, abstractmethod

import torch
import tqdm
from hypal_utils.candles import Candle_OHLC
from torch.utils.data import DataLoader

from hypal_predictor.dataset import TimeSeriesDataset
from hypal_predictor.normalizer import MinMaxNormalizer, Normalizer


class Model(ABC):
    is_fitted: bool = False

    @abstractmethod
    def fit(self, x: list[Candle_OHLC]) -> "Model":
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: list[Candle_OHLC]) -> Candle_OHLC:
        raise NotImplementedError

    @abstractmethod
    def get_context_length(self) -> int:
        raise NotImplementedError


class TorchModel(Model):
    normalizer: Normalizer

    def __init__(self, model: torch.nn.Module, input_size: int, train_steps: int = 10, batch_size: int = 32):
        super().__init__()
        self.model = model
        self.input_size = input_size
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.normalizer = MinMaxNormalizer()

    def fit(self, x: list[Candle_OHLC]) -> "TorchModel":
        from hypal_predictor.utils import create_sequences

        x_norm = self.normalizer.fit_transform(x)

        X_train, y_train = create_sequences(data=x_norm, inp_seq_len=self.get_context_length(), out_seq_len=1)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        train_dataset = TimeSeriesDataset(X_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters())
        loss_fn = torch.nn.MSELoss()

        train_pbar = tqdm.tqdm(range(self.train_steps), leave=False)
        for epoch in train_pbar:
            self.model.train()
            for batch_X, batch_y in train_dataloader:
                optimizer.zero_grad()
                pb = self._process_batch(batch_X)
                output = self.model(pb).unsqueeze(1)
                loss = loss_fn(output, batch_y)
                loss.backward()
                optimizer.step()
            train_pbar.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.5f}")

        self.is_fitted = True
        return self

    @staticmethod
    def _process_batch(x: torch.Tensor) -> torch.Tensor:
        return x
