from pathlib import Path

import pandas as pd
from hypal_utils.candles import Candle_OHLC
from hypal_utils.sensor_data import SensorData
from tqdm import tqdm

from hypal_predictor.dbag import Databag2

EXPERIMENT_PATH = Path(__file__).resolve().parent
DATASET_PATH = EXPERIMENT_PATH / "datasets" / "UUEE.01.03.2025.31.12.2025.1.0.0.en.ansi.00000000.csv"
DATABAG_PATH = EXPERIMENT_PATH / "databags" / "pressure.dbag2"


def main():
    data = pd.read_csv(
        DATASET_PATH,
        on_bad_lines="skip",
        sep=";",
        skiprows=6,
    )
    columns = data.columns.tolist()
    columns[0] = "Date"
    data.columns = columns
    data["Date"] = pd.to_datetime(data["Date"], format="%d.%m.%Y %H:%M").apply(lambda x: x.timestamp())
    data.set_index("Date", inplace=True)
    data.sort_index(inplace=True)

    data = data[["P0"]]
    data.columns = [f"Moscow:Sheremetievo:{c}" for c in data.columns]
    data.dropna(inplace=True)

    dbag = Databag2()
    last_temp = data.iloc[0]["Moscow:Sheremetievo:P0"]
    for timestamp, row in tqdm(data.iterrows()):
        dbag.add(
            SensorData(
                axis="pressure",
                sensor="Sheremetievo",
                source="Moscow",
                timestamp=timestamp,
                candle=Candle_OHLC(
                    open=last_temp,
                    high=row["Moscow:Sheremetievo:P0"],
                    low=row["Moscow:Sheremetievo:P0"],
                    close=row["Moscow:Sheremetievo:P0"],
                ),
            )
        )
        last_temp = row["Moscow:Sheremetievo:P0"]

    dbag.write(DATABAG_PATH)


if __name__ == "__main__":
    main()
