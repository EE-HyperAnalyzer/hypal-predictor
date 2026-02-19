import numpy as np
import torch
from hypal_utils.candles import Candle_OHLC
from torch.utils.data import DataLoader
from tqdm import tqdm

from hypal_predictor.dataset import TimeSeriesDataset
from hypal_predictor.normalizer import MinMaxNormalizer, Normalizer
from hypal_predictor.utils import create_sequences

from .base import Model


class TorchModel(Model):
    def __init__(
        self,
        model: torch.nn.Module,
        input_horizon_length: int,
        train_steps: int = 10,
        batch_size: int = 32,
        normalizer: Normalizer = MinMaxNormalizer(),
        device: str = "cuda",
    ):
        super().__init__(normalizer=normalizer, input_horizon_length=input_horizon_length)
        self.model = model
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.device = device

    def fit(self, x: list[Candle_OHLC], es_tol: float = 1e-5) -> "TorchModel":
        self.model.to(self.device)
        self.model.train()

        x_norm = self._normalizer.fit_transform(x)

        X_train, y_train = create_sequences(data=x_norm, inp_seq_len=self.get_context_length(), out_seq_len=1)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        train_dataset = TimeSeriesDataset(X_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters())
        loss_fn = torch.nn.MSELoss()

        train_pbar = tqdm(range(self.train_steps), leave=False)
        for epoch in train_pbar:
            self.model.train()
            total_loss = []
            for batch_X, batch_y in train_dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                pb = self._process_batch(batch_X)
                output = self.model(pb).unsqueeze(1)
                loss = loss_fn(output, batch_y)
                total_loss.append(loss.item())
                loss.backward()
                optimizer.step()

            loss_mean = np.mean(total_loss)
            train_pbar.set_description(f"Epoch {epoch + 1}, Loss: {loss_mean:.5f}")

            if loss_mean < es_tol:
                train_pbar.set_description("Early stopping")
                break

        self.is_fitted = True
        self.model.eval()
        self.model.cpu()
        return self

    @staticmethod
    def _process_batch(x: torch.Tensor) -> torch.Tensor:
        return x
