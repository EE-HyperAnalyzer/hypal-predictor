from pathlib import Path

import pandas as pd

from hypal_predictor.critical_zone import ZoneRule_AND, ZoneRule_GREATER, ZoneRule_LESS, ZoneRule_NOT
from hypal_predictor.dbag import Databag2
from hypal_predictor.metrics.eval import eval_regression_and_classification
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
from hypal_predictor.timeframe import Timeframe

EXPERIMENT_PATH = Path(__file__).resolve().parent
DATABAG_PATH = EXPERIMENT_PATH / "databags" / "pressure.dbag2"
GRAPHIC_PATH = EXPERIMENT_PATH / "results" / "candlegraph.html"
METRICS_PATH = EXPERIMENT_PATH / "results"


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
    # SLCsC_CatBoostRegressorModel,
    SLCN_CatBoostRegressorModel,
    SLCC_CatBoostRegressorModel,
    # Linreg
    # SLCsC_LinearRegressionModel,
    SLCN_LinearRegressionModel,
    SLCC_LinearRegressionModel,
    # Transformers
    TCN_Transformer,
    # - TCsC_Transformer
    TCC_Transformer,
]

"""
Сценарий определения состояние атмосферного давления.
Норма - 760 мм рт.ст.

Назовём всю зону, выходящую за 760 +- 10 мм рт.ст. критической
"""
CRITICAL_ZONE = ZoneRule_NOT(
    rule=ZoneRule_AND(
        lhs=ZoneRule_GREATER(
            value=750,
        ),
        rhs=ZoneRule_LESS(
            value=770,
        ),
    )
)


def main():
    databag = Databag2.read(DATABAG_PATH)
    axis_data = databag.get_data("Moscow", "Sheremetievo", "pressure")
    assert axis_data
    axis_data.write_html(GRAPHIC_PATH)

    metrics_df = pd.DataFrame()
    for timeframe in TIMEFRAMES:
        data = axis_data.get_data(timeframe=Timeframe.from_str(timeframe))

        if len(data) == 0:
            print(f"No data for timeframe={timeframe}")
            continue

        for model_t in MODELS:
            for inp_seq_len in INP_SEQ_LENS:
                for out_seq_len in OUT_SEQ_LENS:
                    model = model_t(
                        input_horizon=inp_seq_len,
                        output_horizon=out_seq_len,
                        train_size=TRAIN_SIZE,
                    )
                    print(
                        f"Training {model.__class__.__name__} on timeframe={timeframe} with inp_seq_len={inp_seq_len}, out_seq_len={out_seq_len}"
                    )

                    candles = [d.candle for d in data]
                    _n = int(len(candles) * TRAIN_SIZE)
                    train_seq = candles[:_n]
                    test_seq = candles[_n:]
                    model.fit(train_seq)

                    for rollout_mult in ROLLOUT_MULTS:
                        metrics = eval_regression_and_classification(
                            model, test_seq, CRITICAL_ZONE, rollout_multiplier=rollout_mult
                        )
                        print(
                            f"\t rm: {rollout_mult} | r2: {metrics.regression.r2:.3f} | f2: {metrics.classification.f2_050:.3f}"
                        )
                        t_s = Timeframe.from_str(timeframe).as_seconds()
                        metrics_df = pd.concat(
                            [
                                metrics_df,
                                pd.DataFrame(
                                    {
                                        "model": model.__class__.__name__,
                                        "timeframe_sec": t_s,
                                        "inp_seq_len": inp_seq_len,
                                        "out_seq_len": out_seq_len,
                                        "rollout_multiplier": rollout_mult,
                                        "forecasting_window": t_s * out_seq_len * rollout_mult,
                                        "r2": metrics.regression.r2,
                                        "mse": metrics.regression.mse,
                                        "mae": metrics.regression.mae,
                                        "critical_detection_f1": metrics.classification.critical_detection_f1,
                                        "critical_detection_precision": metrics.classification.critical_detection_precision,
                                        "critical_detection_recall": metrics.classification.critical_detection_recall,
                                        "critical_undetection_precision": metrics.classification.critical_undetection_precision,
                                        "critical_undetection_recall": metrics.classification.critical_undetection_recall,
                                        "critical_undetection_f1": metrics.classification.critical_undetection_f1,
                                        "f2_025": metrics.classification.f2_025,
                                        "f2_050": metrics.classification.f2_050,
                                        "f2_075": metrics.classification.f2_075,
                                    },
                                    index=[len(metrics_df)],
                                ),
                            ]
                        )
                        metrics_df.to_csv(METRICS_PATH / f"{Timeframe.from_str(timeframe).as_seconds()}.csv")


if __name__ == "__main__":
    main()
