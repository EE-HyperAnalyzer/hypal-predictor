from pathlib import Path

from hypal_predictor.dbag import Databag, Databag2

EXPERIMENT_PATH = Path(__file__).resolve().parent
DATABAGv1_PATH = EXPERIMENT_PATH / "databags" / "day_1.dbag"
DATABAGv2_PATH = EXPERIMENT_PATH / "databags" / "day_1.dbag2"


def main():
    Databag2.from_dbag(Databag.read(DATABAGv1_PATH)).write(DATABAGv2_PATH)


if __name__ == "__main__":
    main()
