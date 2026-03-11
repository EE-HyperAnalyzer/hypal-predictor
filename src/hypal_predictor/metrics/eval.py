import numpy as np
from hypal_utils.candles import Candle_OHLC
from hypal_utils.critical_zone.rule import ZoneRule
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score

from hypal_predictor.metrics.metrics import ClassificationMetrics, Metrics, RegressionMetrics
from hypal_predictor.model import BaseModel
from hypal_predictor.utils import candle_to_array, create_sequences


def eval_regression_and_classification(
    model: BaseModel, val_seq: list[Candle_OHLC], critical_zone: ZoneRule, rollout_multiplier: int = 1
) -> Metrics:
    assert rollout_multiplier > 0

    val_x, val_y = create_sequences(val_seq, model.input_horizon, model.output_horizon * rollout_multiplier)
    return Metrics(
        regression=eval_regression(model, val_seq, rollout_multiplier),
        classification=eval_classification(model, val_seq, critical_zone, rollout_multiplier),
    )


def eval_regression(model: BaseModel, val_seq: list[Candle_OHLC], rollout_multiplier: int = 1) -> RegressionMetrics:
    assert rollout_multiplier > 0
    val_x, val_y = create_sequences(val_seq, model.input_horizon, model.output_horizon * rollout_multiplier)

    true_ = []
    pred_ = []
    for x, y in zip(val_x, val_y, strict=True):
        z = model.predict(x, rollout_multiplier)
        true_rgr = np.array([candle_to_array(d) for d in y])
        pred_rgr = np.array([candle_to_array(d) for d in z])
        true_.append(true_rgr)
        pred_.append(pred_rgr)

    if len(true_) == 0:
        return RegressionMetrics()
    true_ = np.concatenate(true_)
    pred_ = np.concatenate(pred_)

    r2 = r2_score(true_, pred_)
    mse = mean_squared_error(true_, pred_)
    mae = mean_absolute_error(true_, pred_)
    return RegressionMetrics(r2=r2, mse=mse, mae=mae)


def eval_classification(
    model: BaseModel, val_seq: list[Candle_OHLC], critical_zone: ZoneRule, rollout_multiplier: int = 1
) -> ClassificationMetrics:
    assert rollout_multiplier > 0
    val_x, val_y = create_sequences(val_seq, model.input_horizon, model.output_horizon * rollout_multiplier)

    true_: list[bool] = []
    pred_: list[bool] = []
    for x, y in zip(val_x, val_y, strict=True):
        z = model.predict(x, rollout_multiplier)

        true_cls = any(critical_zone.is_satisfied(candle) for candle in y)
        pred_cls = any(critical_zone.is_satisfied(candle) for candle in z)

        true_.append(true_cls)
        pred_.append(pred_cls)

    if len(true_) == 0:
        return ClassificationMetrics()

    report = classification_report(true_, pred_, output_dict=True, zero_division=0.0)
    return ClassificationMetrics(
        critical_detection_precision=report.get("True", {}).get("precision", 0),
        critical_detection_recall=report.get("True", {}).get("recall", 0),
        critical_detection_f1=report.get("True", {}).get("f1-score", 0),
        critical_undetection_precision=report.get("False", {}).get("precision", 0),
        critical_undetection_recall=report.get("False", {}).get("recall", 0),
        critical_undetection_f1=report.get("False", {}).get("f1-score", 0),
    )
