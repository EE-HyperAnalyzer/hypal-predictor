from dataclasses import dataclass
from typing import Literal


@dataclass
class Timeframe:
    num: int
    det: Literal["s", "m", "h", "d"]

    def as_seconds(self) -> int:
        match self.det:
            case "s":
                return self.num
            case "m":
                return self.num * 60
            case "h":
                return self.num * 3600
            case "d":
                return self.num * 86400

    def __str__(self) -> str:
        return f"{self.num}:{self.det}"

    @staticmethod
    def from_str(timeframe: str) -> "Timeframe":
        assert timeframe.count(":") == 1
        timeframe_n, timeframe_det = timeframe.split(":")
        timeframe_n = int(timeframe_n)
        assert timeframe_det in ["s", "m", "h", "d"]
        return Timeframe(num=timeframe_n, det=timeframe_det)  # ty:ignore[invalid-argument-type]
