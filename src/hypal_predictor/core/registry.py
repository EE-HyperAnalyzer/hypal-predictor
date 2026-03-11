from __future__ import annotations

import logging

from hypal_utils.sensor_data import SensorData
from hypal_utils.timeframe import Timeframe
from redis.asyncio import Redis

from hypal_predictor.core.buffer import ModelState, TimeframeBuffer
from hypal_predictor.schemas.config import TimeframeSettings

logger = logging.getLogger(__name__)


class SensorRegistry:
    """
    Хранит TimeframeBuffer для каждой комбинации (source, sensor, axis, timeframe).
    При получении нового SensorData направляет его в нужные буферы.
    Когда буфер готов к обучению — запускает Celery task.
    """

    def __init__(self, redis: Redis):
        self.redis = redis
        # Ключ: (source, sensor, axis, tf_str) → TimeframeBuffer
        self._buffers: dict[tuple[str, str, str, str], TimeframeBuffer] = {}

    def register(
        self,
        source: str,
        sensor: str,
        axis: str,
        timeframe_settings: dict[str, TimeframeSettings],
    ) -> list[TimeframeBuffer]:
        """
        Регистрирует или обновляет буферы для сенсора.
        Возвращает список созданных/существующих буферов.
        """
        buffers: list[TimeframeBuffer] = []
        for tf_str, settings in timeframe_settings.items():
            key = (source, sensor, axis, tf_str)
            if key not in self._buffers:
                buf = TimeframeBuffer(
                    redis=self.redis,
                    source=source,
                    sensor=sensor,
                    axis=axis,
                    timeframe=Timeframe.from_str(tf_str),
                    settings=settings,
                )
                self._buffers[key] = buf
                logger.info("Registered buffer: %s", key)
            buffers.append(self._buffers[key])
        return buffers

    def get_buffer(
        self,
        source: str,
        sensor: str,
        axis: str,
        timeframe: str,
    ) -> TimeframeBuffer | None:
        return self._buffers.get((source, sensor, axis, timeframe))

    def get_all_buffers(
        self,
        source: str,
        sensor: str,
        axis: str,
    ) -> list[tuple[str, TimeframeBuffer]]:
        """Возвращает список (tf_str, buffer) для заданного сенсора."""
        result: list[tuple[str, TimeframeBuffer]] = []
        for (s, sn, ax, tf), buf in self._buffers.items():
            if s == source and sn == sensor and ax == axis:
                result.append((tf, buf))
        return result

    def remove(self, source: str, sensor: str, axis: str) -> None:
        """Удаляет все буферы сенсора из реестра (Redis-ключи остаются)."""
        keys_to_remove = [k for k in self._buffers if k[:3] == (source, sensor, axis)]
        for k in keys_to_remove:
            del self._buffers[k]
        logger.info("Removed sensor from registry: %s:%s:%s", source, sensor, axis)

    async def consume(self, data: SensorData) -> None:
        """
        Основной entry point. Принимает SensorData, раздаёт в буферы,
        при готовности к обучению — запускает задачу.
        """
        # Ленивый импорт для избежания циклических зависимостей
        from hypal_predictor.tasks.training import train_model  # noqa: PLC0415

        matching_buffers = self.get_all_buffers(data.source, data.sensor, data.axis)
        if not matching_buffers:
            logger.debug(
                "No buffers for %s:%s:%s — skipping",
                data.source,
                data.sensor,
                data.axis,
            )
            return

        for tf_str, buf in matching_buffers:
            state = await buf.get_state()
            if state == ModelState.GATHERING and await buf.is_ready_to_train():
                await self._launch_training(buf, tf_str, train_model)
                continue

            aggregated = await buf.push_raw(data)
            if aggregated is None:
                continue

            # Проверяем, нужно ли запустить обучение
            if await buf.is_ready_to_train():
                await self._launch_training(buf, tf_str, train_model)

    async def _launch_training(
        self,
        buf: TimeframeBuffer,
        tf_str: str,
        train_model_task,  # передаём явно, чтобы не импортировать повторно
    ) -> None:
        candles = await buf.get_all_candles()
        candles_json = [c.model_dump() for c in candles]

        result = train_model_task.delay(
            source=buf.source,
            sensor=buf.sensor,
            axis=buf.axis,
            timeframe=tf_str,
            input_horizon=buf.input_horizon,
            output_horizon=buf.output_horizon,
            rollout_multiplier=buf.rollout_multiplier,
            model_type=buf.model_type,
            candles_json=candles_json,
        )
        await buf.mark_training(task_id=result.id)
        logger.info(
            "Training launched for %s:%s:%s tf=%s task_id=%s",
            buf.source,
            buf.sensor,
            buf.axis,
            tf_str,
            result.id,
        )

    def remove_timeframe(self, source: str, sensor: str, axis: str, timeframe: str) -> None:
        """Удаляет буфер для конкретного таймфрейма сенсора из реестра (Redis-ключи остаются)."""
        key = (source, sensor, axis, timeframe)
        self._buffers.pop(key, None)
        logger.info("Removed timeframe %s for sensor %s:%s:%s from registry", timeframe, source, sensor, axis)


# ---------------------------------------------------------------------------
# Module-level singleton — инициализируется в lifespan приложения.
# ---------------------------------------------------------------------------
registry: SensorRegistry | None = None


def get_registry() -> SensorRegistry:
    """
    Возвращает глобальный синглтон SensorRegistry.

    Raises:
        RuntimeError: Если реестр не был инициализирован через init_registry().
    """
    if registry is None:
        raise RuntimeError("SensorRegistry not initialized. Call init_registry() in lifespan.")
    return registry


def init_registry(redis: Redis) -> SensorRegistry:
    """
    Инициализирует глобальный синглтон SensorRegistry.
    Должен вызываться один раз при старте приложения (в lifespan).
    """
    global registry
    registry = SensorRegistry(redis=redis)
    logger.info("SensorRegistry initialized.")
    return registry
