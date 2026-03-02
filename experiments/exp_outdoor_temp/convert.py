from pathlib import Path

import pandas as pd

from hypal_predictor.dbag import Databag

EXPERIMENT_PATH = Path(__file__).resolve().parent
DATASET_PATH = EXPERIMENT_PATH / "datasets" / "UUEE.01.03.2025.31.12.2025.1.0.0.en.ansi.00000000.csv"
DATABAG_PATH = EXPERIMENT_PATH / "databags" / "temperature.dbag"


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

    data = data[["T"]]
    data.columns = [f"Moscow:Sheremetievo:{c}" for c in data.columns]
    data.dropna(inplace=True)

    dbag = Databag.from_dataframe(data)
    dbag.write(DATABAG_PATH)


if __name__ == "__main__":
    main()
