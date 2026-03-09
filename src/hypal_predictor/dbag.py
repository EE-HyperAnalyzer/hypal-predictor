import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from hypal_utils.candles import Candle_OHLC
from hypal_utils.sensor_data import SensorData
from tqdm import tqdm

from hypal_predictor.timeframe import Timeframe


@dataclass
class Databag:
    _source_data: dict[str, dict[str, dict[str, list[SensorData]]]] = field(default_factory=dict)
    _min_timeframe: Timeframe = field(default_factory=Timeframe.default)

    def get_data(
        self, source: str, sensor: str, axis: str, timeframe: Timeframe = Timeframe.default()
    ) -> list[SensorData]:
        if self._min_timeframe.as_seconds() > timeframe.as_seconds():
            raise ValueError("Timeframe is too small")
        return self._aggregate_data_with_timeframe(
            self._source_data.get(source, {}).get(sensor, {}).get(axis, []), timeframe
        )

    def write(self, path: Path):
        with path.open("w") as f:
            for _, sensor_data in self._source_data.items():
                for _, sensor_data in sensor_data.items():
                    for _, axis_data in sensor_data.items():
                        for data in axis_data:
                            f.write(f"{data.model_dump_json()}\n")

    @classmethod
    def read(cls, path: Path, limit: Optional[int] = None, timeframe: Timeframe = Timeframe.default()) -> "Databag":
        self = cls(_min_timeframe=timeframe)
        with path.open("r") as f:
            for i, line in enumerate(tqdm(f.readlines(), desc="Reading", total=limit)):
                if limit and i >= limit:
                    break
                data = SensorData(**json.loads(line))
                self._source_data[data.source] = self._source_data.get(data.source, {})
                self._source_data[data.source][data.sensor] = self._source_data[data.source].get(data.sensor, {})
                self._source_data[data.source][data.sensor][data.axis] = self._source_data[data.source][
                    data.sensor
                ].get(data.axis, []) + [data]
        return self

    def get_source_names(self) -> list[str]:
        return list(self._source_data.keys())

    def get_sensor_names(self, source: str) -> list[str]:
        return list(self._source_data.get(source, {}).keys())

    def get_axis_names(self, source: str, sensor: str) -> list[str]:
        return list(self._source_data.get(source, {}).get(sensor, {}).keys())

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, sep: str = ":") -> "Databag":
        source_data: dict[str, dict[str, dict[str, list[SensorData]]]] = {}

        last_close = {}
        for ts, row in tqdm(df.iterrows(), total=len(df)):
            for col, value in row.items():
                source, sensor, axis = col.split(sep)
                if col not in last_close:
                    last_close[col] = value
                candle = Candle_OHLC(open=last_close[col], high=value, low=value, close=value)
                source_data[source] = source_data.get(source, {})
                source_data[source][sensor] = source_data[source].get(sensor, {})
                source_data[source][sensor][axis] = source_data[source][sensor].get(axis, []) + [
                    SensorData(source=source, sensor=sensor, axis=axis, candle=candle, timestamp=ts)
                ]
                last_close[col] = value

        return cls(source_data)

    @classmethod
    def from_sensor_data(cls, data: list[SensorData]) -> "Databag":
        source_data = {}
        for d in data:
            source_data[d.source] = source_data.get(d.source, {})
            source_data[d.source][d.sensor] = source_data[d.source].get(d.sensor, {})
            source_data[d.source][d.sensor][d.axis] = source_data[d.source][d.sensor].get(d.axis, []) + [d]
        return cls(source_data)

    @staticmethod
    def _aggregate_data_with_timeframe(data: list[SensorData], timeframe: Timeframe) -> list[SensorData]:
        if not data:
            return []

        sorted_data: list[SensorData] = sorted(data, key=lambda x: x.timestamp)
        tf_sec: int = timeframe.as_seconds()

        groups = []
        curr_group = []
        last_group_start_timestamp = sorted_data[0].timestamp

        for item in sorted_data:
            if item.timestamp - last_group_start_timestamp < tf_sec:
                curr_group.append(item)
            else:
                groups.append(curr_group)
                curr_group = [item]
                last_group_start_timestamp = item.timestamp

        result: list[SensorData] = []
        for group in groups:
            first = group[0]
            last = group[-1]

            if len(group) == 1:
                agg_candle = first.candle
            else:
                agg_candle = Candle_OHLC(
                    open=first.candle.open,
                    high=max(d.candle.high for d in group),
                    low=min(d.candle.low for d in group),
                    close=last.candle.close,
                )

            result.append(
                SensorData(
                    source=first.source,
                    sensor=first.sensor,
                    axis=first.axis,
                    candle=agg_candle,
                    timestamp=first.timestamp,
                )
            )

        return result


@dataclass
class AxisDatabag:
    source: str
    sensor: str
    axis: str

    _data: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
    )
    _min_timeframe: Timeframe = field(default_factory=Timeframe.default)

    def __post_init__(self):
        self._data["timestamp"] = self._data["timestamp"].astype(int)
        for col in ["open", "high", "low", "close"]:
            self._data[col] = self._data[col].astype(float)
        self._data.set_index("timestamp", inplace=True)

    def get_data(self, timeframe: Timeframe = Timeframe.default()) -> list[SensorData]:
        if self._min_timeframe.as_seconds() > timeframe.as_seconds():
            raise ValueError("Timeframe is too small")
        return self._aggregate_data_with_timeframe(timeframe)

    def add(self, sensor_data: SensorData):
        self._data = pd.concat(
            [
                self._data,
                pd.DataFrame(
                    {
                        "timestamp": sensor_data.timestamp,
                        "open": sensor_data.candle.open,
                        "high": sensor_data.candle.high,
                        "low": sensor_data.candle.low,
                        "close": sensor_data.candle.close,
                    },
                    index=[sensor_data.timestamp],
                ),
            ],
            ignore_index=True,
        )

    def write_html(self, path: Path):

        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=self._data.timestamp,
                open=self._data.open,
                high=self._data.high,
                low=self._data.low,
                close=self._data.close,
            )
        )
        fig.write_html(path)

    def _aggregate_data_with_timeframe(self, timeframe: Timeframe) -> list[SensorData]:
        data = self._data.sort_values(by="timestamp")
        tf_sec: int = timeframe.as_seconds()
        groups = []
        curr_group = []
        last_group_start_timestamp = self._data.iloc[0].timestamp

        for _, item in data.iterrows():
            item = SensorData(
                source=self.source,
                sensor=self.sensor,
                axis=self.axis,
                candle=Candle_OHLC(open=item.open, high=item.high, low=item.low, close=item.close),
                timestamp=item.timestamp,
            )

            if item.timestamp - last_group_start_timestamp < tf_sec:
                curr_group.append(item)
            else:
                groups.append(curr_group)
                curr_group = [item]
                last_group_start_timestamp = item.timestamp

        result: list[SensorData] = []
        for group in groups:
            first = group[0]
            last = group[-1]

            if len(group) == 1:
                agg_candle = first.candle
            else:
                agg_candle = Candle_OHLC(
                    open=first.candle.open,
                    high=max(d.candle.high for d in group),
                    low=min(d.candle.low for d in group),
                    close=last.candle.close,
                )

            result.append(
                SensorData(
                    source=first.source,
                    sensor=first.sensor,
                    axis=first.axis,
                    candle=agg_candle,
                    timestamp=first.timestamp,
                )
            )

        return result

    def __repr__(self) -> str:
        return f"<AxisDatabag source={self.source} sensor={self.sensor} axis={self.axis}>\n{self._data}"


@dataclass
class Databag2:
    _data: dict[str, dict[str, dict[str, AxisDatabag]]] = field(default_factory=dict)
    _min_timeframe: Timeframe = field(default_factory=Timeframe.default)

    def get_data(self, source: str, sensor: str, axis: str) -> AxisDatabag | None:
        return self._data.get(source, {}).get(sensor, {}).get(axis)

    def add(self, data: SensorData):
        source = data.source
        sensor = data.sensor
        axis = data.axis

        self._data[source] = self._data.get(source, {})
        self._data[source][sensor] = self._data[source].get(sensor, {})
        self._data[source][sensor][axis] = self._data[source][sensor].get(axis, AxisDatabag(source, sensor, axis))

        self._data[source][sensor][axis].add(data)

    def write(self, path: Path):
        with path.open("wb") as f:
            pickle.dump(self._data, f)

    @classmethod
    def read(cls, path: Path) -> "Databag2":
        with path.open("rb") as f:
            return cls(pickle.load(f))

    @classmethod
    def from_sensor_data(cls, data: list[SensorData]) -> "Databag2":
        dbag = cls()
        for d in tqdm(data):
            dbag.add(d)
        return dbag

    @staticmethod
    def _aggregate_data_with_timeframe(data: list[SensorData], timeframe: Timeframe) -> list[SensorData]:
        if not data:
            return []

        sorted_data: list[SensorData] = sorted(data, key=lambda x: x.timestamp)
        print(len(sorted_data))
        tf_sec: int = timeframe.as_seconds()

        groups = []
        curr_group = []
        last_group_start_timestamp = sorted_data[0].timestamp

        for item in sorted_data:
            if item.timestamp - last_group_start_timestamp < tf_sec:
                curr_group.append(item)
            else:
                groups.append(curr_group)
                curr_group = [item]
                last_group_start_timestamp = item.timestamp

        result: list[SensorData] = []
        for group in groups:
            first = group[0]
            last = group[-1]

            if len(group) == 1:
                agg_candle = first.candle
            else:
                agg_candle = Candle_OHLC(
                    open=first.candle.open,
                    high=max(d.candle.high for d in group),
                    low=min(d.candle.low for d in group),
                    close=last.candle.close,
                )

            result.append(
                SensorData(
                    source=first.source,
                    sensor=first.sensor,
                    axis=first.axis,
                    candle=agg_candle,
                    timestamp=first.timestamp,
                )
            )

        return result
