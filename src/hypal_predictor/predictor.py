from dataclasses import dataclass

import torch
import tqdm
from hypal_utils.candles import Candle_OHLC
from torch.utils.data import DataLoader

from hypal_predictor.dataset import TimeSeriesDataset
from hypal_predictor.metrics import MAE, MSE, Metric
from hypal_predictor.model import Model
from hypal_predictor.normalizer import MinMaxNormalizer, Normalizer
from hypal_predictor.utils import create_sequences, rollout


@dataclass
class PredictResult: ...


@dataclass
class Ok(PredictResult):
    horizont: list[Candle_OHLC]


@dataclass
class Gather(PredictResult): ...


class PredictorStream:
    model: Model
    output_horizont_size: int
    is_fitted: bool = False
    scaler: Normalizer

    def __init__(self, model: Model, output_horizont_size: int):
        self.model = model
        self.output_horizont_size = output_horizont_size

    def fit(
        self,
        data: list[Candle_OHLC],
        train_size: float = 0.8,
        train_steps: int = 100,
        batch_size: int = 32,
        lr: float = 3e-3,
        metrics: tuple[type[Metric], ...] = (MSE, MAE),
    ) -> dict[str, list[float]]:
        self.scaler = MinMaxNormalizer()
        candle_scaled_data = self.scaler.fit_transform(data)

        n = len(candle_scaled_data)
        train_size = int(n * train_size)
        train_data = candle_scaled_data[:train_size]
        test_data = candle_scaled_data[train_size:]

        X_train, y_train = create_sequences(data=train_data, inp_seq_len=self.model.get_context_length(), out_seq_len=1)
        X_test, y_test = create_sequences(
            data=test_data, inp_seq_len=self.model.get_context_length(), out_seq_len=self.output_horizont_size
        )

        train_dataset = TimeSeriesDataset(X_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = TimeSeriesDataset(X_test, y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        train_pbar = tqdm.tqdm(range(train_steps), leave=False)
        for epoch in train_pbar:
            self.model.train()
            for batch_X, batch_y in train_dataloader:
                optimizer.zero_grad()
                print(batch_X.shape)
                output = self.model(batch_X).unsqueeze(1)
                loss = loss_fn(output, batch_y)
                loss.backward()
                optimizer.step()
            train_pbar.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.5f}")

        metric_values: dict[str, list[float]] = {}
        self.model.eval()
        with torch.no_grad():
            for batch_X, batch_y in test_dataloader:
                y_pred = rollout(self.model, batch_X, self.output_horizont_size)

                y_pred_unbatched = y_pred.view(-1, y_pred.shape[-1])
                batch_y_unbatched = batch_y.view(-1, batch_y.shape[-1])

                for metric in metrics:
                    metric_values[metric.__name__] = metric_values.get(metric.__name__, []) + [
                        metric(batch_y_unbatched, y_pred_unbatched).calculate()
                    ]

        self.is_fitted = True
        return metric_values

    def predict(self, data: list[Candle_OHLC]) -> list[Candle_OHLC]:
        assert len(data) == self.model.get_context_length()
        data_scaled = self.scaler.transform(data)
        x = torch.tensor([[d.open, d.high, d.low, d.close] for d in data_scaled])
        pred_horizont = rollout(self.model, x, self.output_horizont_size).detach().numpy()
        pred_candles: list[Candle_OHLC] = [
            Candle_OHLC(
                open=pred_horizont[i][0], high=pred_horizont[i][1], low=pred_horizont[i][2], close=pred_horizont[i][3]
            )
            for i in range(len(pred_horizont))
        ]
        result = self.scaler.reverse(pred_candles)
        return result
