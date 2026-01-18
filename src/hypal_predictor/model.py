from abc import ABC, abstractmethod

import torch.nn as nn


class Model(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_context_length(self) -> int:
        raise NotImplementedError
