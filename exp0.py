from copy import deepcopy
from pathlib import Path

import numpy as np

from src.dbag import DbagReader
from src.model.builtin import BoostingModel
from src.model.builtin.linear import LinearModel
from src.model.builtin.transformer import TimeSeriesTransformerModel
from src.normalizer import MinMaxNormalizer
from src.val import val_model

PROJECT_ROOT = Path(__file__).resolve().parent
DATABAGS_PATH = PROJECT_ROOT / "databags" / "day_1.dbag"


TRAIN_SIZE = 0.8
INP_SEQ_LEN = 50
OUT_SEQ_LEN = 1


def main():
    dbag = DbagReader(DATABAGS_PATH)

    models = [
        LinearModel(INP_SEQ_LEN, train_steps=100, batch_size=128, device="cuda", normalizer=MinMaxNormalizer()),
        TimeSeriesTransformerModel(
            INP_SEQ_LEN, train_steps=100, batch_size=128, device="cuda", normalizer=MinMaxNormalizer()
        ),
        BoostingModel(INP_SEQ_LEN, normalizer=MinMaxNormalizer()),
    ]

    for sensor in dbag.get_detectors_names():
        data = dbag.get_detector_data(sensor)

        _d = int(len(data) * TRAIN_SIZE)
        data_train = data[:_d]
        data_test = data[_d:]

        for model in models:
            model = deepcopy(model)
            model.fit(data_train)

            metrics = val_model(data=data_test, out_horizont=5, model=model)
            print(sensor, model.__class__)
            for metric_name, metric_arr in metrics.items():
                print(f"{metric_name}: {np.mean(metric_arr):.5f}")


if __name__ == "__main__":
    main()
