import json
from pathlib import Path

from hypal_utils.sensor_data import SensorData
from tqdm import tqdm


class DbagReader:
    def __init__(self, dbag_path: Path):
        self.data: dict[str, list[SensorData]] = {}

        with dbag_path.open("r") as f:
            for line in tqdm(f.readlines(), desc="Reading"):
                data = SensorData(**json.loads(line))
                name = f"{data.source}:{data.sensor}:{data.axis}"
                self.data[name] = self.data.get(name, []) + [data]

    def get_detectors_names(self) -> list[str]:
        return list(self.data.keys())

    def get_detector_data(self, detector_name: str) -> list[SensorData]:
        return self.data.get(detector_name, [])
