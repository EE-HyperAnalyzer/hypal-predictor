import json
import logging

from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class SignalAggregator:
    """
    Для каждого сенсора агрегирует булевы сигналы is_critical
    по всем READY-таймфреймам (OR-логика).

    Хранит предыдущее состояние в Redis:
      signal:{source}:{sensor}:{axis} → Hash { is_critical, triggered_tfs (JSON), timestamp }

    При изменении состояния возвращает событие.
    """

    def __init__(self, redis: Redis):
        self.redis = redis

    async def update(
        self,
        source: str,
        sensor: str,
        axis: str,
        per_tf_signals: dict[str, bool],  # tf_str → is_critical
        timestamp: int,
    ) -> tuple[bool, str | None, list[str]]:
        """
        Принимает сигналы со всех таймфреймов.
        Возвращает (current_is_critical, event | None, triggered_timeframes).

        event: "CRITICAL_ENTERED" | "CRITICAL_EXITED" | None
        """
        triggered = [tf for tf, val in per_tf_signals.items() if val]
        current_is_critical = bool(triggered)

        sig_key = f"signal:{source}:{sensor}:{axis}"
        prev_raw = await self.redis.hget(sig_key, "is_critical")  # type: ignore[not-awaitable]
        previous_is_critical = (
            prev_raw is not None and (prev_raw.decode() if isinstance(prev_raw, bytes) else prev_raw) == "1"
        )

        # Определяем событие
        event: str | None = None
        if not previous_is_critical and current_is_critical:
            event = "CRITICAL_ENTERED"
        elif previous_is_critical and not current_is_critical:
            event = "CRITICAL_EXITED"

        # Сохраняем новое состояние
        await self.redis.hset(  # type: ignore[not-awaitable]
            sig_key,
            mapping={
                "is_critical": "1" if current_is_critical else "0",
                "triggered_tfs": json.dumps(triggered),
                "timestamp": timestamp,
            },
        )

        if event:
            logger.info(
                "Signal change [%s]: %s:%s:%s triggered=%s",
                event,
                source,
                sensor,
                axis,
                triggered,
            )

        return current_is_critical, event, triggered

    async def get_current_signal(self, source: str, sensor: str, axis: str) -> dict:
        """Возвращает текущий агрегированный сигнал."""
        sig_key = f"signal:{source}:{sensor}:{axis}"
        data = await self.redis.hgetall(sig_key)  # type: ignore[not-awaitable]
        if not data:
            return {"is_critical": False, "triggered_timeframes": [], "timestamp": None}

        def _decode(v: bytes | str) -> str:
            return v.decode() if isinstance(v, bytes) else v

        return {
            "is_critical": _decode(data.get(b"is_critical", data.get("is_critical", "0"))) == "1",
            "triggered_timeframes": json.loads(_decode(data.get(b"triggered_tfs", data.get("triggered_tfs", "[]")))),
            "timestamp": int(_decode(data.get(b"timestamp", data.get("timestamp", "0")))) or None,
        }
