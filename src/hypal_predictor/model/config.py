from dataclasses import dataclass

from hypal_predictor.normalizer import Normalizer


@dataclass
class ModelConfig:
    input_horizont_length: int
    output_horizont_length: int
    train_size: float
    anomaly_threshold: float
    candles_to_train: int
    normalizer: Normalizer
