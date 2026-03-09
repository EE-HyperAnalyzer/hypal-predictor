import logging

import httpx
from hypal_utils.candles import Candle_OHLC

from hypal_predictor.config import settings

logger = logging.getLogger(__name__)


class CoreAPINotifier:
    """Отправляет POST-уведомления в CoreAPI при изменении критической зоны."""

    def __init__(self, timeout_s: int | None = None):
        self._timeout = timeout_s or settings.core_api_timeout_s

    async def notify(
        self,
        core_api_url: str,
        source: str,
        sensor: str,
        axis: str,
        event: str,  # "CRITICAL_ENTERED" | "CRITICAL_EXITED"
        timestamp: int,
        triggered_timeframes: list[str],
        predictions: dict[str, list[Candle_OHLC]],
    ) -> bool:
        """
        Отправляет уведомление в CoreAPI.

        Возвращает True при успешной доставке.
        Ошибки логируются и не пробрасываются — вызывающий код не должен падать.
        """
        payload = {
            "source": source,
            "sensor": sensor,
            "axis": axis,
            "event": event,
            "timestamp": timestamp,
            "triggered_timeframes": triggered_timeframes,
            "predictions": {tf: [c.model_dump() for c in candles] for tf, candles in predictions.items()},
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(core_api_url, json=payload)
                resp.raise_for_status()
                logger.info(
                    "Notified CoreAPI [%s]: %s:%s:%s triggered=%s",
                    event,
                    source,
                    sensor,
                    axis,
                    triggered_timeframes,
                )
                return True
        except httpx.HTTPStatusError as e:
            logger.error(
                "CoreAPI returned error %d for %s:%s:%s: %s",
                e.response.status_code,
                source,
                sensor,
                axis,
                e,
            )
        except httpx.TimeoutException:
            logger.error(
                "CoreAPI request timed out (%.1fs) for %s:%s:%s",
                self._timeout,
                source,
                sensor,
                axis,
            )
        except Exception as e:
            logger.error(
                "Failed to notify CoreAPI for %s:%s:%s: %s",
                source,
                sensor,
                axis,
                e,
            )
        return False


# ---------------------------------------------------------------------------
# Module-level singleton — используется во всём приложении.
# ---------------------------------------------------------------------------
notifier = CoreAPINotifier()
