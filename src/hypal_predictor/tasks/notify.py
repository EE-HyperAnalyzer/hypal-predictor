import logging

import httpx

from hypal_predictor.tasks.app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    name="hypal_predictor.tasks.notify.send_signal",
    max_retries=5,
    default_retry_delay=30,
)
def send_signal(
    core_api_url: str,
    source: str,
    sensor: str,
    axis: str,
    event: str,
    timestamp: int,
    triggered_timeframes: list[str],
    predictions: dict,  # tf → list[dict]
) -> bool:
    """
    Celery task: отправляет POST в CoreAPI.
    Используется как резервный путь при недоступности CoreAPI
    (retry с экспоненциальным back-off).

    Args:
        core_api_url: URL эндпоинта CoreAPI для уведомлений.
        source: Идентификатор источника данных.
        sensor: Идентификатор сенсора.
        axis: Ось сенсора.
        event: Тип события ("CRITICAL_ENTERED" | "CRITICAL_EXITED").
        timestamp: Unix-время события.
        triggered_timeframes: Таймфреймы, триггернувшие событие.
        predictions: Предсказания в виде {tf: [candle_dict, ...]}.

    Returns:
        True при успешной отправке.

    Raises:
        Celery Retry при любой ошибке (до max_retries раз).
    """
    payload = {
        "source": source,
        "sensor": sensor,
        "axis": axis,
        "event": event,
        "timestamp": timestamp,
        "triggered_timeframes": triggered_timeframes,
        "predictions": predictions,
    }

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(core_api_url, json=payload)
            resp.raise_for_status()

        logger.info(
            "send_signal task OK: %s %s:%s:%s",
            event,
            source,
            sensor,
            axis,
        )
        return True

    except Exception as exc:
        retry_number = send_signal.request.retries
        countdown = 30 * (2**retry_number)

        logger.warning(
            "send_signal failed (attempt %d/%d), retrying in %ds: %s",
            retry_number + 1,
            send_signal.max_retries + 1,
            countdown,
            exc,
        )

        raise send_signal.retry(exc=exc, countdown=countdown)
