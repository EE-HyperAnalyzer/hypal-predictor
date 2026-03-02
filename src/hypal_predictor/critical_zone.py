from abc import ABC, abstractmethod

from hypal_utils.candles import Candle_OHLC


class ZoneRule(ABC):
    @abstractmethod
    def is_satisfied(self, candle: Candle_OHLC) -> bool:
        raise NotImplementedError


class ZoneRule_LESS(ZoneRule):
    def __init__(self, value: float):
        self.value = value

    def is_satisfied(self, candle: Candle_OHLC) -> bool:
        arr = [candle.open, candle.high, candle.low, candle.close]
        return any(x < self.value for x in arr)


class ZoneRule_GREATER(ZoneRule):
    def __init__(self, value: float):
        self.value = value

    def is_satisfied(self, candle: Candle_OHLC) -> bool:
        arr = [candle.open, candle.high, candle.low, candle.close]
        return any(x > self.value for x in arr)


class ZoneRule_AND(ZoneRule):
    def __init__(self, lhs: ZoneRule, rhs: ZoneRule):
        self.lhs = lhs
        self.rhs = rhs

    def is_satisfied(self, candle: Candle_OHLC) -> bool:
        return self.lhs.is_satisfied(candle) and self.rhs.is_satisfied(candle)


class ZoneRule_OR(ZoneRule):
    def __init__(self, lhs: ZoneRule, rhs: ZoneRule):
        self.lhs = lhs
        self.rhs = rhs

    def is_satisfied(self, candle: Candle_OHLC) -> bool:
        return self.lhs.is_satisfied(candle) or self.rhs.is_satisfied(candle)


class ZoneRule_NOT(ZoneRule):
    def __init__(self, rule: ZoneRule):
        self.rule = rule

    def is_satisfied(self, candle: Candle_OHLC) -> bool:
        return not self.rule.is_satisfied(candle)
