from pathlib import Path

import pandas as pd

from hypal_predictor.dbag import Databag

EXPERIMENT_PATH = Path(__file__).resolve().parent
DATASET_PATH = EXPERIMENT_PATH / "datasets" / "Occupancy_Estimation.csv"
DATABAG_PATH = EXPERIMENT_PATH / "databags" / "co2.dbag"


def main():
    data = pd.read_csv(
        DATASET_PATH,
        on_bad_lines="skip",
        sep=",",
    )
    data["Timeframe"] = pd.to_datetime(data["Date"] + " " + data["Time"]).apply(lambda x: x.timestamp()).astype(int)
    data.set_index("Timeframe", inplace=True)

    data = data[["S5_CO2"]]
    data.columns = ["SmartHouse:Room:co2"]
    data.dropna(inplace=True)

    dbag = Databag.from_dataframe(data)
    dbag.write(DATABAG_PATH)


if __name__ == "__main__":
    main()
