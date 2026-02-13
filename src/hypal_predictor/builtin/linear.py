import torch
import torch.nn as nn

from hypal_predictor.model import Model


class LinearModel(Model):
    def __init__(self, input_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size * 4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        return self.linear(x)

    def get_context_length(self) -> int:
        return self.linear.in_features // 4
