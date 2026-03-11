from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from celery import Task
from hypal_utils.candles import Candle_OHLC

from hypal_predictor.config import settings
from hypal_predictor.tasks.app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="hypal_predictor.tasks.training.train_model",
    max_retries=3,
    default_retry_delay=60,
)
def train_model(
    self: Task,
    source: str,
    sensor: str,
    axis: str,
    timeframe: str,
    input_horizon: int,
    output_horizon: int,
    rollout_multiplier: int,
    model_type: str,
    candles_json: list[dict],
) -> dict:
    """
    Celery task: обучает модель для (source, sensor, axis, timeframe).

    Порядок действий:
    1. Читает SensorConfig из БД через async-сессию.
    2. Создаёт экземпляр модели через create_model().
    3. Обучает модель: model.fit(candles).
    4. Сохраняет модель на диск через ModelStore.
    5. Вычисляет метрики (MSE, MAE, R²) на валидационной выборке.
    6. Финализирует запись TrainingHistory в БД.
    7. Обновляет Redis-ключ: TRAINING → READY.
    """
    logger.info(
        "Training started: %s:%s:%s tf=%s n_candles=%d",
        source,
        sensor,
        axis,
        timeframe,
        len(candles_json),
    )
    started_at = datetime.now(tz=timezone.utc)

    async def _run() -> dict:
        from hypal_predictor.core.model_factory import create_model
        from hypal_predictor.core.model_store import model_store
        from hypal_predictor.db.engine import AsyncSessionLocal
        from hypal_predictor.db.repos.training_history import (
            add as add_history,
        )
        from hypal_predictor.db.repos.training_history import (
            finish as finish_history,
        )

        async with AsyncSessionLocal() as session:
            history = await add_history(
                session=session,
                source=source,
                sensor=sensor,
                axis=axis,
                timeframe=timeframe,
                started_at=started_at,
            )
            await session.commit()

        candles = [Candle_OHLC(**c) for c in candles_json]
        model = create_model(
            model_type=model_type,
            input_horizon=input_horizon,
            output_horizon=output_horizon,
        )

        logger.info(
            "Fitting %s model for %s:%s:%s tf=%s",
            model_type,
            source,
            sensor,
            axis,
            timeframe,
        )
        try:
            model.fit(candles)
        except Exception as exc:
            logger.exception("model.fit() failed for %s:%s:%s tf=%s", source, sensor, axis, timeframe)
            raise exc

        model_path = model_store.save(source, sensor, axis, timeframe, model)
        logger.info("Model saved to %s", model_path)

        mse, mae, r2 = _eval_metrics(model, candles, rollout_multiplier)
        logger.info(
            "Metrics for %s:%s:%s tf=%s — MSE=%.6f MAE=%.6f R2=%.4f",
            source,
            sensor,
            axis,
            timeframe,
            mse,
            mae,
            r2,
        )

        async with AsyncSessionLocal() as session:
            await finish_history(
                session=session,
                record_id=history.id,
                finished_at=datetime.now(tz=timezone.utc),
                mse=mse,
                mae=mae,
                r2=r2,
                model_path=str(model_path),
            )
            await session.commit()

        # Обновляем состояние в Redis: TRAINING → READY
        _set_redis_state(source, sensor, axis, timeframe, "READY")

        return {
            "status": "ok",
            "model_path": str(model_path),
            "mse": mse,
            "mae": mae,
            "r2": r2,
        }

    try:
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(_run())
        finally:
            loop.close()

        logger.info("Training finished: %s:%s:%s tf=%s", source, sensor, axis, timeframe)
        return result

    except Exception as exc:
        logger.exception("Training task failed for %s:%s:%s tf=%s", source, sensor, axis, timeframe)
        # Откатываем состояние: TRAINING → GATHERING, чтобы можно было повторить
        _set_redis_state(source, sensor, axis, timeframe, "GATHERING")
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_redis_state(
    source: str,
    sensor: str,
    axis: str,
    timeframe: str,
    state: str,
) -> None:
    """Синхронно обновляет поле 'state' в Redis-хэше состояния буфера."""
    import redis as redis_sync

    key = f"state:{source}:{sensor}:{axis}:{timeframe}"
    try:
        r = redis_sync.from_url(settings.redis_url, decode_responses=True)
        r.hset(key, mapping={"state": state})
        r.close()
    except Exception:
        logger.exception("Failed to set Redis state=%s for key=%s", state, key)


def _eval_metrics(
    model,
    candles: list[Candle_OHLC],
    rollout_multiplier: int,
) -> tuple[float, float, float]:
    """
    Быстрая оценка качества модели на валидационной части свечей (последние 20%).

    Returns:
        Кортеж (mse, mae, r2). При нехватке данных возвращает (0.0, 0.0, 0.0).
    """
    import numpy as np

    from hypal_predictor.utils import candle_to_array, create_sequences

    split = int(len(candles) * 0.8)
    val_candles = candles[split:]

    min_required = model.input_horizon + model.output_horizon * rollout_multiplier
    if len(val_candles) < min_required:
        logger.warning(
            "Not enough validation candles (%d < %d), skipping metric evaluation.",
            len(val_candles),
            min_required,
        )
        return 0.0, 0.0, 0.0

    val_x, val_y = create_sequences(
        val_candles,
        model.input_horizon,
        model.output_horizon * rollout_multiplier,
    )

    trues: list[list] = []
    preds: list[list] = []

    for x, y in zip(val_x, val_y):
        try:
            pred = model.predict(x, rollout_multiplier)
        except Exception:
            logger.exception("model.predict() failed during metric evaluation, skipping sample.")
            continue

        trues.append([candle_to_array(c) for c in y])
        preds.append([candle_to_array(c) for c in pred])

    if not trues:
        return 0.0, 0.0, 0.0

    t = np.concatenate(trues)  # (N, 4)
    p = np.concatenate(preds)  # (N, 4)

    mse = float(np.mean((t - p) ** 2))
    mae = float(np.mean(np.abs(t - p)))

    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean(axis=0)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return mse, mae, r2
