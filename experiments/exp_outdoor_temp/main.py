from pathlib import Path

from hypal_utils.critical_zone.rule import ZoneRule_AND, ZoneRule_GREATER, ZoneRule_LESS, ZoneRule_NOT

from hypal_predictor.experiment import run_experiment
from hypal_predictor.model import BaseModel
from hypal_predictor.model.scikit_compat.candle import SLCC_CatBoostRegressorModel, SLCC_LinearRegressionModel
from hypal_predictor.model.scikit_compat.numeric import SLCN_CatBoostRegressorModel, SLCN_LinearRegressionModel
from hypal_predictor.model.scikit_compat.shrinked_candle import (
    SLCsC_CatBoostRegressorModel,  # noqa: F401
    SLCsC_LinearRegressionModel,  # noqa: F401
)
from hypal_predictor.model.torch_compat.candle import TCC_Transformer
from hypal_predictor.model.torch_compat.numeric import TCN_Transformer
from hypal_predictor.model.torch_compat.shrinked_candle import TCsC_Transformer  # noqa: F401

"""
Сценарий для оптимального роста сакуры.
Сакура — растение, предпочитающее умеренный климат.
Оптимальная температура для цветения — +18°C, при этом рост и развитие происходят в диапазоне от 5°C до 35°C

Скажем, что значения вне этого диапазона будут критичными
"""
CRITICAL_ZONE = ZoneRule_NOT(
    rule=ZoneRule_AND(
        lhs=ZoneRule_GREATER(
            value=5,
        ),
        rhs=ZoneRule_LESS(
            value=35,
        ),
    )
)

DATABAG_NAME = "temperature.dbag2"
SOURCE = "Moscow"
SENSOR = "Sheremetievo"
AXIS = "temperature"
TRAIN_SIZE = 0.8
TIMEFRAMES = [
    "30:m",
    "5:m",
    "1:m",
]
INP_SEQ_LENS = [5, 20, 30]
OUT_SEQ_LENS = [1, 5, 15]
ROLLOUT_MULTS = [1, 2, 3]
MODELS: list[type[BaseModel]] = [
    # Catboost
    # > SLCsC_CatBoostRegressorModel,
    SLCN_CatBoostRegressorModel,
    SLCC_CatBoostRegressorModel,
    # Linreg
    # > SLCsC_LinearRegressionModel,
    SLCN_LinearRegressionModel,
    SLCC_LinearRegressionModel,
    # Transformers
    TCN_Transformer,
    # > TCsC_Transformer
    TCC_Transformer,
]


def main():
    run_experiment(
        experiment_path=Path(__file__).resolve().parent,
        source=SOURCE,
        sensor=SENSOR,
        axis=AXIS,
        databag2_name=DATABAG_NAME,
        models=MODELS,
        critical_zone=CRITICAL_ZONE,
        inp_seq_lens=INP_SEQ_LENS,
        out_seq_lens=OUT_SEQ_LENS,
        timeframes=TIMEFRAMES,
        rollout_mults=ROLLOUT_MULTS,
        train_size=TRAIN_SIZE,
    )


if __name__ == "__main__":
    main()
