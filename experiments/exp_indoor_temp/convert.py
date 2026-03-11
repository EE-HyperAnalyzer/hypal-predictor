from pathlib import Path

import pandas as pd
from hypal_utils.candles import Candle_OHLC
from hypal_utils.sensor_data import SensorData
from tqdm import tqdm

from hypal_predictor.dbag import Databag2

EXPERIMENT_PATH = Path(__file__).resolve().parent
DATASET_PATH = EXPERIMENT_PATH / "datasets" / "epever_battery_temperature.csv"
DATABAG_PATH = EXPERIMENT_PATH / "databags" / "temperature.dbag2"


def main():
    data = pd.read_csv(
        DATASET_PATH,
        on_bad_lines="skip",
        sep=",",
    )
    data["Timestamp"] = pd.to_datetime(data["time"]).apply(lambda x: x.timestamp()).astype(int)
    data.set_index("Timestamp", inplace=True)

    data = data[["value"]]
    data.columns = ["car:accumulator:temp"]
    data.dropna(inplace=True)

    dbag = Databag2()
    last_temp = data.iloc[0]["car:accumulator:temp"]
    for timestamp, row in tqdm(data.iterrows()):
        dbag.add(
            SensorData(
                axis="temp",
                sensor="accumulator",
                source="car",
                timestamp=timestamp,
                candle=Candle_OHLC(
                    open=last_temp,
                    high=row["car:accumulator:temp"],
                    low=row["car:accumulator:temp"],
                    close=row["car:accumulator:temp"],
                ),
            )
        )
        last_temp = row["car:accumulator:temp"]

    dbag.write(DATABAG_PATH)


if __name__ == "__main__":
    main()
