from abc import ABC, abstractmethod
from collections import deque

from hypal_utils.candles import Candle_OHLC


class BaseModel[T](ABC):
    input_horizon: int
    output_horizon: int
    train_size: float

    _initialized: bool
    _x_mus: T
    _y_mus: T
    _x_std: T
    _y_std: T

    def __init__(self, input_horizon: int, output_horizon: int, train_size: float):
        self.input_horizon = input_horizon
        self.output_horizon = output_horizon
        self.train_size = train_size

    @abstractmethod
    def fit(
        self,
        data_seq: list[Candle_OHLC],
    ) -> "BaseModel":
        raise NotImplementedError

    def predict(self, x: list[Candle_OHLC], rollout_multiplier: int = 1) -> list[Candle_OHLC]:
        queue = deque(x)
        result = []
        for _ in range(rollout_multiplier):
            prediction = self._predict_raw(list(queue))
            result.extend(prediction)
            queue.extend(prediction)
            for _ in range(self.output_horizon):
                queue.popleft()
        return result

    @abstractmethod
    def _predict_raw(self, x: list[Candle_OHLC]) -> list[Candle_OHLC]:
        raise NotImplementedError

    @abstractmethod
    def _train_eval_split(self, candle_seq: list[Candle_OHLC]) -> tuple[T, T, T, T]:
        raise NotImplementedError


# @dataclass
# class ScikitLearnModel(BaseModel, metaclass=ABCMeta):
#     def _split(self, x: pd.DataFrame, y: pd.DataFrame) -> tuple[]


#     def _init_as_numeric_model(
#         self: BaseModel, candle_seq: list[Candle_OHLC]
#     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#         assert not self._initialized

#         raw_df = {"open": [], "high": [], "low": [], "close": []}
#         for candle in candle_seq:
#             raw_df["open"].append(candle.open)
#             raw_df["high"].append(candle.high)
#             raw_df["low"].append(candle.low)
#             raw_df["close"].append(candle.close)
#         raw_df = pd.DataFrame(raw_df)

#         df = pd.DataFrame()
#         for i in range(1, self.input_horizon + 1):
#             columns = [f"open_i_{i}", f"high_i_{i}", f"low_i_{i}", f"close_i_{i}"]
#             extra_df = raw_df.shift(-i)[["open", "high", "low", "close"]]
#             extra_df.columns = columns
#             df = pd.concat([df, extra_df], axis=1)

#         for i in range(1, self.output_horizon + 1):
#             columns = [f"open_o_{i}", f"high_o_{i}", f"low_o_{i}", f"close_o_{i}"]
#             extra_df = raw_df.shift(-i - self.input_horizon)[["open", "high", "low", "close"]].mean(axis=1)
#             df = pd.concat([df, extra_df], axis=1)
#             df.columns = df.columns[:-1].tolist() + [f"o_{i}"]

#         return self._train_eval_split(df)

#     def _get_candle_dataset(
#         self: BaseModel, candle_seq: list[Candle_OHLC]
#     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#         assert not self._initialized

#         raw_df = {"open": [], "high": [], "low": [], "close": []}
#         for candle in candle_seq:
#             raw_df["open"].append(candle.open)
#             raw_df["high"].append(candle.high)
#             raw_df["low"].append(candle.low)
#             raw_df["close"].append(candle.close)
#         raw_df = pd.DataFrame(raw_df)

#         df = pd.DataFrame()
#         for i in range(1, self.input_horizon + 1):
#             columns = [f"open_i_{i}", f"high_i_{i}", f"low_i_{i}", f"close_i_{i}"]
#             extra_df = raw_df.shift(-i)[["open", "high", "low", "close"]]
#             extra_df.columns = columns
#             df = pd.concat([df, extra_df], axis=1)

#         for i in range(1, self.output_horizon + 1):
#             columns = [f"open_o_{i}", f"high_o_{i}", f"low_o_{i}", f"close_o_{i}"]
#             extra_df = raw_df.shift(-i - self.input_horizon)[["open", "high", "low", "close"]]
#             extra_df.columns = columns
#             df = pd.concat([df, extra_df], axis=1)

#         return self._train_eval_split(df)

#     def _get_shrinked_dataset(
#         self: BaseModel, candle_seq: list[Candle_OHLC]
#     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#         assert not self._initialized

#         raw_df = {"open": [], "high": [], "low": [], "close": []}
#         for candle in candle_seq:
#             raw_df["open"].append(candle.open)
#             raw_df["high"].append(candle.high)
#             raw_df["low"].append(candle.low)
#             raw_df["close"].append(candle.close)
#         raw_df = pd.DataFrame(raw_df)

#         df = pd.DataFrame()
#         for i in range(1, self.input_horizon + 1):
#             columns = [f"open_i_{i}", f"high_i_{i}", f"low_i_{i}", f"close_i_{i}"]
#             extra_df = raw_df.shift(-i)[["open", "high", "low", "close"]]
#             extra_df.columns = columns
#             df = pd.concat([df, extra_df], axis=1)

#         for i in range(1, self.output_horizon + 1):
#             columns = [f"h_o_{i}", f"l_o_{i}", f"c_o_{i}", f"close_o_{i}"]
#             shifted_df = raw_df.shift(-i - self.input_horizon)[["open", "high", "low", "close"]]

#             extra_df = pd.DataFrame()
#             extra_df[f"h_o_{i}"] = shifted_df["high"] - shifted_df["open"]
#             extra_df[f"l_o_{i}"] = shifted_df["open"] - shifted_df["low"]

#             delta = np.clip((extra_df[f"h_o_{i}"] + extra_df[f"l_o_{i}"]).to_numpy(), a_min=1e-5, a_max=None)
#             extra_df[f"c_o_{i}"] = (shifted_df["high"] - shifted_df["close"]) / delta
#             df = pd.concat([df, extra_df], axis=1)

#         return self._train_eval_split(df)


# @dataclass
# class TorchModel(BaseModel, metaclass=ABCMeta):
#     x_mus: torch.Tensor
#     y_mus: torch.Tensor

#     x_std: torch.Tensor
#     y_std: torch.Tensor

#     def _init_as_candle_dataset(
#         self, candle_seq: list[Candle_OHLC]
#     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         train_x, train_y, eval_x, eval_y = ScikitLearnModel._get_candle_dataset(self, candle_seq)

#         train_x = torch.tensor(train_x, dtype=torch.float32).view(-1, self.input_horizon, 4)
#         train_y = torch.tensor(train_y, dtype=torch.float32)

#         eval_x = torch.tensor(eval_x, dtype=torch.float32).view(-1, self.input_horizon, 4)
#         eval_y = torch.tensor(eval_y, dtype=torch.float32)

#     def _get_numeric_dataset(self, candle_seq: list[Candle_OHLC]): ...
