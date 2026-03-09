from abc import ABCMeta

import numpy as np

from hypal_predictor.model.base import BaseModel


class ScikitLearnCompatibleModel(BaseModel[np.ndarray], metaclass=ABCMeta):
    def _split(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert len(x) == len(y)
        _n = int(len(x) * self.train_size)
        train_x = x[:_n, ...]
        eval_x = x[_n:, ...]
        train_y = y[:_n, ...]
        eval_y = y[_n:, ...]
        return train_x, train_y, eval_x, eval_y
