from pathlib import Path

import pandas as pd
import tqdm
from hypal_utils.critical_zone.rule import ZoneRule
from hypal_utils.logger import log_info
from hypal_utils.timeframe import Timeframe

from hypal_predictor.dbag import AxisDatabag, Databag2
from hypal_predictor.metrics.eval import eval_regression_and_classification
from hypal_predictor.model import BaseModel


def run_experiment(
    experiment_path: Path,
    source: str,
    sensor: str,
    axis: str,
    databag2_name: str,
    models: list[type[BaseModel]],
    critical_zone: ZoneRule,
    inp_seq_lens: list[int],
    out_seq_lens: list[int],
    timeframes: list[str],
    rollout_mults: list[int],
    train_size: float = 0.8,
):
    DATABAG_PATH = experiment_path / "databags" / databag2_name
    RESULTS_PATH = experiment_path / "results"

    databag = Databag2.read(DATABAG_PATH)
    axis_data = databag.get_data(source, sensor, axis)
    metrics_df = pd.DataFrame()

    assert axis_data

    for timeframe in tqdm.tqdm(timeframes, desc="Timeframes"):
        log_info("Getting aggregated data...")
        tf = Timeframe.from_str(timeframe)
        data = axis_data.get_data(timeframe=tf)

        log_info("Plotting candlestick graph...")
        timeframe_folder = RESULTS_PATH / f"{tf.as_seconds()}"
        timeframe_folder.mkdir(exist_ok=True, parents=True)
        aggregated_data = AxisDatabag(source=source, sensor=sensor, axis=axis)
        for c in tqdm.tqdm(data):
            aggregated_data.add(c)

        aggregated_data.write_html(timeframe_folder / "candlestick.html")

        log_info("Training models...")
        for model_t in tqdm.tqdm(models, desc="Models"):
            for inp_seq_len in tqdm.tqdm(inp_seq_lens, desc="Input horizon"):
                for out_seq_len in tqdm.tqdm(out_seq_lens, desc="Outut horizon"):
                    model = model_t(
                        input_horizon=inp_seq_len,
                        output_horizon=out_seq_len,
                        train_size=train_size,
                    )
                    log_info(
                        f"Training {model.__class__.__name__} on timeframe={timeframe} with inp_seq_len={inp_seq_len}, out_seq_len={out_seq_len}"
                    )

                    candles = [d.candle for d in data]
                    _n = int(len(candles) * train_size)
                    train_seq = candles[:_n]
                    test_seq = candles[_n:]
                    model.fit(train_seq)

                    for rollout_mult in rollout_mults:
                        metrics = eval_regression_and_classification(
                            model, test_seq, critical_zone, rollout_multiplier=rollout_mult
                        )
                        log_info(
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
                        metrics_df.to_csv(timeframe_folder / "metrics.csv")
