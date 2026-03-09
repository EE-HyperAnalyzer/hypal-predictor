from pathlib import Path

import pandas as pd

from hypal_predictor.dbag import Databag

EXPERIMENT_PATH = Path(__file__).resolve().parent
DATASET_PATH = EXPERIMENT_PATH / "datasets" / "epever_battery_temperature.csv"
DATABAG_PATH = EXPERIMENT_PATH / "databags" / "temperature.dbag"


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
    dbag = Databag.from_dataframe(data)
    dbag.write(DATABAG_PATH)


if __name__ == "__main__":
    main()
