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

TIMEFRAMES = ["30:s", "1:h"]
INP_SEQ_LENS = [1, 5, 10, 15, 20]
OUT_SEQ_LENS = [1, 5, 10, 15, 20]
MODELS = [CatBoostRegressorModel, LinearRegressionModel]

"""
Сценарий работы с датчиком температуры в LiIoN аккумуляторе.
Оптимальная температура - 20 - 25 градусов. Все что выходит за пределы - критическая зона
"""
CRITICAL_ZONE = ZoneRule_NOT(
    rule=ZoneRule_AND(
        lhs=ZoneRule_GREATER(
            value=20,
        ),
        rhs=ZoneRule_LESS(
            value=25,
        ),
    )
)


def main():
    databag = Databag.read(DATABAG_PATH, timeframe=Timeframe.from_str("30:s"))

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
        data = databag.get_data("car", "accumulator", "temp", timeframe=Timeframe.from_str(timeframe))

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
                                    "label_true_precision": cls_metrics.get("True", {}).get("precision", 0),  # ty:ignore[unresolved-attribute]
                                    "label_true_recall": cls_metrics.get("True", {}).get("recall", 0),  # ty:ignore[unresolved-attribute]
                                    "label_true_f1": cls_metrics.get("True", {}).get("f1-score", 0),  # ty:ignore[unresolved-attribute]
                                    "label_false_precision": cls_metrics.get("False", {}).get("precision", 0),  # ty:ignore[unresolved-attribute]
                                    "label_false_recall": cls_metrics.get("False", {}).get("recall", 0),  # ty:ignore[unresolved-attribute]
                                    "label_false_f1": cls_metrics.get("False", {}).get("f1-score", 0),  # ty:ignore[unresolved-attribute]
                                    **rgr_metrics,
                                },
                                index=[len(df)],
                            ),
                        ]
                    )
    df.to_csv(EXPERIMENT_PATH / "metrics.csv")


if __name__ == "__main__":
    main()
