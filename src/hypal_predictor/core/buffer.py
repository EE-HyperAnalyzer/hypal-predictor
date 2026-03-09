from __future__ import annotations

import json
from enum import Enum

from hypal_utils.candles import Candle_OHLC
from hypal_utils.sensor_data import SensorData
from redis.asyncio import Redis

from hypal_predictor.core.aggregator import CandleAggregator
from hypal_predictor.timeframe import Timeframe


class ModelState(str, Enum):
    WAITING = "WAITING"
    GATHERING = "GATHERING"
    TRAINING = "TRAINING"
    READY = "READY"


class TimeframeBuffer:
    """
    Управляет состоянием модели для конкретной тройки (source, sensor, axis)
    на конкретном таймфрейме. Хранит состояние и буфер свечей в Redis.

    Redis-ключи:
      state:{source}:{sensor}:{axis}:{tf}  → Hash
      buf:tf:{source}:{sensor}:{axis}:{tf} → List (JSON-свечи)
    """

    def __init__(
        self,
        redis: Redis,
        source: str,
        sensor: str,
        axis: str,
        timeframe: Timeframe,
        num_train_samples: int,
    ):
        self.redis = redis
        self.source = source
        self.sensor = sensor
        self.axis = axis
        self.timeframe = timeframe
        self.num_train_samples = num_train_samples
        self._aggregator = CandleAggregator(timeframe=timeframe)

        tf_str = str(timeframe)
        key_prefix = f"{source}:{sensor}:{axis}:{tf_str}"
        self._state_key = f"state:{key_prefix}"
        self._buf_key = f"buf:tf:{key_prefix}"

    async def get_state(self) -> ModelState:
        val = await self.redis.hget(self._state_key, "state")  # type: ignore[not-awaitable]
        if val is None:
            return ModelState.WAITING
        return ModelState(val.decode() if isinstance(val, bytes) else val)

    async def set_state(self, state: ModelState, task_id: str | None = None) -> None:
        mapping: dict[str, str] = {"state": state.value}
        if task_id is not None:
            mapping["task_id"] = task_id
        await self.redis.hset(self._state_key, mapping=mapping)  # type: ignore[not-awaitable]

    async def get_task_id(self) -> str | None:
        val = await self.redis.hget(self._state_key, "task_id")  # type: ignore[not-awaitable]
        if val is None:
            return None
        return val.decode() if isinstance(val, bytes) else val

    async def get_gathered_count(self) -> int:
        val = await self.redis.hget(self._state_key, "gathered_count")  # type: ignore[not-awaitable]
        if val is None:
            return 0
        return int(val.decode() if isinstance(val, bytes) else val)

    async def get_all_candles(self) -> list[Candle_OHLC]:
        raw = await self.redis.lrange(self._buf_key, 0, -1)  # type: ignore[not-awaitable]
        return [Candle_OHLC(**json.loads(r)) for r in raw]

    async def push_raw(self, data: SensorData) -> SensorData | None:
        """
        Принимает сырую 1s-свечу, агрегирует, при завершении окна:
        - добавляет агрегированную свечу в Redis-буфер
        - возвращает агрегированную SensorData (для дальнейшей обработки)
        - если буфер достиг num_train_samples и состояние GATHERING → возвращает сигнал
        """
        state = await self.get_state()
        if state == ModelState.WAITING:
            # Начинаем сбор
            await self.set_state(ModelState.GATHERING)
            state = ModelState.GATHERING

        # Если уже идёт обучение или модель готова — пропускаем агрегацию
        if state not in (ModelState.GATHERING,):
            return None

        aggregated = self._aggregator.push(data)
        if aggregated is None:
            return None

        # Сохраняем агрегированную свечу в Redis
        candle_json = aggregated.candle.model_dump_json()
        await self.redis.rpush(self._buf_key, candle_json)  # type: ignore[not-awaitable]
        count = await self.redis.llen(self._buf_key)  # type: ignore[not-awaitable]
        await self.redis.hset(self._state_key, "gathered_count", count)  # type: ignore[not-awaitable]

        return aggregated

    async def is_ready_to_train(self) -> bool:
        state = await self.get_state()
        if state not in (ModelState.GATHERING,):
            return False
        count = await self.get_gathered_count()
        return count >= self.num_train_samples

    async def mark_training(self, task_id: str) -> None:
        await self.set_state(ModelState.TRAINING, task_id=task_id)

    async def mark_ready(self) -> None:
        await self.set_state(ModelState.READY)

    async def mark_gathering(self) -> None:
        await self.set_state(ModelState.GATHERING)

    async def reset(self) -> None:
        """Полный сброс: очищает буфер и возвращает в WAITING."""
        await self.redis.delete(self._state_key, self._buf_key)
        # Сбрасываем in-memory агрегатор
        self._aggregator = CandleAggregator(timeframe=self.timeframe)
