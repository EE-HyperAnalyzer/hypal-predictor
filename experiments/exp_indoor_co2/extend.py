from copy import deepcopy
from pathlib import Path

from tqdm import tqdm

from hypal_predictor.dbag import Databag, Databag2

EXPERIMENT_PATH = Path(__file__).resolve().parent
DATABAG_PATH = EXPERIMENT_PATH / "databags" / "day_2.dbag"


def main():
    databag = Databag.read(DATABAG_PATH)

    co2 = databag.get_data(
        source="NightTokyo",
        sensor="sgp30",
        axis="eco2",
    )

    data = []
    src = deepcopy(co2)
    for _ in tqdm(range(5)):
        data.extend(deepcopy(src))

    for i, d in enumerate(data):
        d.timestamp = i

    dbag = Databag2().from_sensor_data(data)
    dbag.write(EXPERIMENT_PATH / "databags" / "day_2_extended.dbag2")


if __name__ == "__main__":
    main()
