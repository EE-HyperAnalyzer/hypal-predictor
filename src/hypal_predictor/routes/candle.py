import asyncio
import logging
import time

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from hypal_utils.sensor_data import SensorData

from hypal_predictor.core.buffer import ModelState
from hypal_predictor.core.model_store import model_store
from hypal_predictor.core.registry import get_registry
from hypal_predictor.schemas.candle import CandleIngestRequest, CandleIngestResponse

router = APIRouter(prefix="/candle", tags=["Candle"])
logger = logging.getLogger(__name__)


@router.post("/ingest", response_model=CandleIngestResponse)
async def ingest_candles(request: Request, body: CandleIngestRequest):
    """
    Принимает батч SensorData и раздаёт по буферам.
    После каждой агрегированной свечи запускает предсказание если модель READY.
    """
    registry = get_registry()

    async def process_one(data: SensorData):
        await registry.consume(data)
        await _maybe_predict(request, data.source, data.sensor, data.axis)

    await asyncio.gather(*[process_one(d) for d in body.candles])
    return CandleIngestResponse(received=len(body.candles))


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Стриминг SensorData через WebSocket."""
    await websocket.accept()
    registry = get_registry()
    logger.info("WebSocket connected")
    try:
        while True:
            raw = await websocket.receive_json()
            data = SensorData(**raw)
            await registry.consume(data)
            await websocket.send_json({"status": "ok", "sensor": f"{data.source}:{data.sensor}:{data.axis}"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        await websocket.close()


async def _maybe_predict(request: Request, source: str, sensor: str, axis: str):
    """Запускает предсказание для всех READY-таймфреймов сенсора."""
    from hypal_predictor.db.engine import AsyncSessionLocal
    from hypal_predictor.db.repos.critical_zone import get as get_cz
    from hypal_predictor.db.repos.sensor_config import get as get_config
    from hypal_predictor.schemas.critical_zone import rule_from_json

    predictor = request.app.state.predictor
    signal_agg = request.app.state.signal_aggregator
    registry = get_registry()

    per_tf_signals: dict[str, bool] = {}
    predictions: dict = {}
    timestamp = 0

    async with AsyncSessionLocal() as session:
        config = await get_config(session, source, sensor, axis)
        if config is None:
            return

        cz_record = await get_cz(session, source, sensor, axis)
        critical_zone = None
        if cz_record:
            try:
                critical_zone = rule_from_json(cz_record.rule_json)
            except Exception:
                pass

    for tf_str, buf in registry.get_all_buffers(source, sensor, axis):
        state = await buf.get_state()
        if state != ModelState.READY:
            continue

        model = model_store.load(source, sensor, axis, tf_str)
        if model is None:
            continue

        candles, is_critical = await predictor.run(
            source=source,
            sensor=sensor,
            axis=axis,
            timeframe=tf_str,
            model=model,
            rollout_multiplier=config.rollout_multiplier,
            critical_zone=critical_zone,
        )
        per_tf_signals[tf_str] = is_critical
        predictions[tf_str] = candles
        timestamp = int(time.time())

    if not per_tf_signals:
        return

    _, event, triggered = await signal_agg.update(
        source=source,
        sensor=sensor,
        axis=axis,
        per_tf_signals=per_tf_signals,
        timestamp=timestamp,
    )

    if event and config.core_api_url:
        from hypal_predictor.core.notifier import notifier

        await notifier.notify(
            core_api_url=config.core_api_url,
            source=source,
            sensor=sensor,
            axis=axis,
            event=event,
            timestamp=timestamp,
            triggered_timeframes=triggered,
            predictions=predictions,
        )
