from pathlib import Path

import pandas as pd

from hypal_predictor.critical_zone import ZoneRule_AND, ZoneRule_GREATER, ZoneRule_LESS, ZoneRule_NOT
from hypal_predictor.dbag import Databag
from hypal_predictor.model.builtin import CatBoostRegressorModel, LinearRegressionModel
from hypal_predictor.timeframe import Timeframe
from hypal_predictor.utils import to_train_valid_split

EXPERIMENT_PATH = Path(__file__).resolve().parent
DATABAG_PATH = EXPERIMENT_PATH / "databags" / "temperature.dbag"

TRAIN_RATIO = 0.8

TIMEFRAMES = ["30:m", "1:h"]
INP_SEQ_LENS = [1, 5, 10, 15, 20]
OUT_SEQ_LENS = [1, 5, 10, 15, 20]
MODELS = [CatBoostRegressorModel, LinearRegressionModel]

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


def main():
    databag = Databag.read(DATABAG_PATH, timeframe=Timeframe.from_str("30:m"))

    df = pd.DataFrame(
        columns=[
            "model",
            "timeframe_sec",
            "inp_seq_len",
            "out_seq_len",
            "mse",
            "mae",
            "r2",
            "label_true_precision",
            "label_true_recall",
            "label_true_f1",
            "label_false_precision",
            "label_false_recall",
            "label_false_f1",
        ]
    )
    for timeframe in TIMEFRAMES:
        data = databag.get_data("Moscow", "Sheremetievo", "T", timeframe=Timeframe.from_str(timeframe))

        for model_t in MODELS:
            for inp_seq_len in INP_SEQ_LENS:
                for out_seq_len in OUT_SEQ_LENS:
                    train_x, train_y, valid_x, valid_y = to_train_valid_split(
                        data, inp_seq_len, out_seq_len, train_ratio=TRAIN_RATIO
                    )
                    model = model_t(inp_seq_len, out_seq_len)
                    print(
                        f"Training {model.__class__.__name__} on timeframe={timeframe} with inp_seq_len={inp_seq_len}, out_seq_len={out_seq_len}"
                    )
                    model.fit(train_x, train_y, valid_x, valid_y)

                    rgr_metrics = model.eval_regression(valid_x, valid_y)
                    cls_metrics = model.eval_classification(valid_x, valid_y, CRITICAL_ZONE)

                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                {
                                    "model": model.__class__.__name__,
                                    "timeframe_sec": Timeframe.from_str(timeframe).as_seconds(),
                                    "inp_seq_len": inp_seq_len,
                                    "out_seq_len": out_seq_len,
                                    "label_true_precision": cls_metrics["True"]["precision"],  # ty:ignore[not-subscriptable]
                                    "label_true_recall": cls_metrics["True"]["recall"],  # ty:ignore[not-subscriptable]
                                    "label_true_f1": cls_metrics["True"]["f1-score"],  # ty:ignore[not-subscriptable]
                                    "label_false_precision": cls_metrics["False"]["precision"],  # ty:ignore[not-subscriptable]
                                    "label_false_recall": cls_metrics["False"]["recall"],  # ty:ignore[not-subscriptable]
                                    "label_false_f1": cls_metrics["False"]["f1-score"],  # ty:ignore[not-subscriptable]
                                    **rgr_metrics,
                                },
                                index=[len(df)],
                            ),
                        ]
                    )
    df.to_csv(EXPERIMENT_PATH / "metrics.csv")


if __name__ == "__main__":
    main()
