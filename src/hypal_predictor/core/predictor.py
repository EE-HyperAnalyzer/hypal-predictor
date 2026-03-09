import json
import logging

from hypal_utils.candles import Candle_OHLC
from redis.asyncio import Redis

from hypal_predictor.critical_zone import ZoneRule
from hypal_predictor.model.base import BaseModel as PredictorBaseModel

logger = logging.getLogger(__name__)


class PredictorEngine:
    """
    Для одной тройки (source, sensor, axis) + таймфрейма:
    - Берёт последние input_horizon свечей из буфера
    - Вызывает model.predict(x, rollout_multiplier)
    - Проверяет каждую предсказанную свечу на критическую зону
    - Сохраняет результат в Redis: pred:{source}:{sensor}:{axis}:{tf}
    """

    def __init__(self, redis: Redis):
        self.redis = redis

    async def run(
        self,
        source: str,
        sensor: str,
        axis: str,
        timeframe: str,
        model: PredictorBaseModel,
        rollout_multiplier: int,
        critical_zone: ZoneRule | None,
    ) -> tuple[list[Candle_OHLC], bool]:
        """
        Возвращает (predicted_candles, is_critical).
        is_critical = True если хотя бы одна предсказанная свеча попала в critical_zone.
        Если critical_zone is None — is_critical = False.
        """
        buf_key = f"buf:tf:{source}:{sensor}:{axis}:{timeframe}"
        raw = await self.redis.lrange(buf_key, -model.input_horizon, -1)  # type: ignore[not-awaitable]

        if len(raw) < model.input_horizon:
            logger.debug(
                "Not enough candles for prediction: got %d, need %d",
                len(raw),
                model.input_horizon,
            )
            return [], False

        x = [Candle_OHLC(**json.loads(r)) for r in raw]

        try:
            predicted = model.predict(x, rollout_multiplier)
        except Exception:
            logger.exception(
                "model.predict failed for %s:%s:%s tf=%s",
                source,
                sensor,
                axis,
                timeframe,
            )
            return [], False

        is_critical = False
        if critical_zone is not None:
            is_critical = any(critical_zone.is_satisfied(c) for c in predicted)

        pred_key = f"pred:{source}:{sensor}:{axis}:{timeframe}"
        pred_data = [c.model_dump() for c in predicted]
        await self.redis.set(pred_key, json.dumps(pred_data))

        logger.debug(
            "Prediction saved [%s:%s:%s tf=%s]: %d candles, is_critical=%s",
            source,
            sensor,
            axis,
            timeframe,
            len(predicted),
            is_critical,
        )

        return predicted, is_critical

    async def get_last_prediction(
        self,
        source: str,
        sensor: str,
        axis: str,
        timeframe: str,
    ) -> list[Candle_OHLC]:
        """Возвращает последнее сохранённое предсказание (или пустой список)."""
        pred_key = f"pred:{source}:{sensor}:{axis}:{timeframe}"
        raw = await self.redis.get(pred_key)
        if raw is None:
            return []
        try:
            return [Candle_OHLC(**c) for c in json.loads(raw)]
        except Exception:
            logger.exception(
                "Failed to deserialize prediction for %s:%s:%s tf=%s",
                source,
                sensor,
                axis,
                timeframe,
            )
            return []
