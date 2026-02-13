from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Metric(ABC):
    y_true: torch.Tensor
    y_pred: torch.Tensor

    def __post_init__(self):
        assert len(self.y_true) == len(self.y_pred), "y_true and y_pred must have the same length"

    @abstractmethod
    def calculate(self) -> float:
        raise NotImplementedError


class MAE(Metric):
    def calculate(self) -> float:
        return float(np.abs(self.y_true[:, 3] - self.y_pred[:, 3]).mean())


class MSE(Metric):
    def calculate(self) -> float:
        return float(((self.y_true[:, 3] - self.y_pred[:, 3]) ** 2).mean())
