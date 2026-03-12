import json
import logging
import time

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from hypal_utils.sensor_data import SensorData
from pydantic import ValidationError

from hypal_predictor.core.buffer import ModelState
from hypal_predictor.core.model_store import model_store
from hypal_predictor.core.registry import get_registry
from hypal_predictor.db.engine import AsyncSessionLocal
from hypal_predictor.db.repos.sensor_config import get as get_sensor_config
from hypal_predictor.db.repos.sensor_config import upsert as upsert_sensor_config

router = APIRouter(prefix="/candle", tags=["Candle"])
logger = logging.getLogger(__name__)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Стриминг SensorData через WebSocket."""
    await websocket.accept()
    registry = get_registry()
    client = getattr(websocket, "client", None)
    client_repr = f"{client.host}:{client.port}" if client else "unknown"
    logger.info("WebSocket connected: %s", client_repr)
    try:
        while True:
            raw = await websocket.receive_json()
            try:
                data = SensorData(**raw)
            except ValidationError as e:
                logger.warning("Invalid websocket payload from %s: %s; payload=%s", client_repr, e, raw)
                await websocket.send_json(
                    {
                        "source": raw.get("source"),
                        "sensor": raw.get("sensor"),
                        "axis": raw.get("axis"),
                        "timeframes": {},
                        "error": "Invalid SensorData payload",
                    }
                )
                continue

            logger.info("data: %s", data)
            should_process = await _ensure_sensor_config(data)
            if should_process:
                await registry.consume(data)
                try:
                    await _maybe_predict(websocket, data.source, data.sensor, data.axis)
                except Exception:
                    logger.exception(
                        "Prediction pipeline failed after websocket ingestion for %s:%s:%s",
                        data.source,
                        data.sensor,
                        data.axis,
                    )

            response = await _build_ws_status_response(websocket, data.source, data.sensor, data.axis)
            await websocket.send_json(response)
    except WebSocketDisconnect as e:
        logger.info("WebSocket disconnected: %s code=%s", client_repr, e.code)
    except Exception:
        logger.exception("WebSocket error from %s", client_repr)
        try:
            await websocket.close()
        except Exception:
            logger.debug("WebSocket already closed: %s", client_repr)


async def _ensure_sensor_config(data: SensorData) -> bool:
    registry = get_registry()
    existing = registry.get_all_buffers(data.source, data.sensor, data.axis)
    if existing:
        return True

    async with AsyncSessionLocal() as session:
        record = await get_sensor_config(session, data.source, data.sensor, data.axis)
        if record is None:
            await upsert_sensor_config(
                session=session,
                source=data.source,
                sensor=data.sensor,
                axis=data.axis,
                timeframes=json.dumps({}),
            )
            logger.info(
                "Auto-created WAITING placeholder config for %s:%s:%s",
                data.source,
                data.sensor,
                data.axis,
            )
            return False

        timeframes = json.loads(record.timeframes)
        if not timeframes:
            logger.debug(
                "Sensor %s:%s:%s is in WAITING state: no timeframes configured",
                data.source,
                data.sensor,
                data.axis,
            )
            return False

        registry.register(source=data.source, sensor=data.sensor, axis=data.axis, timeframe_settings=timeframes)
        return True


async def _maybe_predict(connection: Request | WebSocket, source: str, sensor: str, axis: str):
    """Запускает предсказание для всех READY-таймфреймов сенсора."""
    from hypal_predictor.db.engine import AsyncSessionLocal
    from hypal_predictor.db.repos.critical_zone import get as get_cz
    from hypal_predictor.db.repos.sensor_config import get as get_config

    predictor = connection.app.state.predictor
    signal_agg = connection.app.state.signal_aggregator
    redis = connection.app.state.redis
    registry = get_registry()

    per_tf_signals: dict[str, bool] = {}
    timestamp = 0

    async with AsyncSessionLocal() as session:
        config = await get_config(session, source, sensor, axis)
        if config is None:
            return

        timeframes = json.loads(config.timeframes)

        cz_record = await get_cz(session, source, sensor, axis)
        critical_zone = None
        if cz_record:
            try:
                critical_zone = cz_record.rule_json
            except Exception:
                pass

    for tf_str, buf in registry.get_all_buffers(source, sensor, axis):
        state = await buf.get_state()
        if state != ModelState.READY:
            continue

        model = model_store.load(source, sensor, axis, tf_str)
        if model is None:
            continue

        _, is_critical = await predictor.run(
            source=source,
            sensor=sensor,
            axis=axis,
            timeframe=tf_str,
            model=model,
            rollout_multiplier=timeframes[tf_str]["rollout_multiplier"],
            critical_zone=critical_zone,
        )
        per_tf_signals[tf_str] = is_critical
        timestamp = int(time.time())
        await redis.hset(
            f"state:{source}:{sensor}:{axis}:{tf_str}",
            mapping={"is_critical": "1" if is_critical else "0"},
        )

    if not per_tf_signals:
        return

    await signal_agg.update(
        source=source,
        sensor=sensor,
        axis=axis,
        per_tf_signals=per_tf_signals,
        timestamp=timestamp,
    )


async def _build_ws_status_response(
    connection: Request | WebSocket,
    source: str,
    sensor: str,
    axis: str,
) -> dict:
    from hypal_predictor.db.engine import AsyncSessionLocal
    from hypal_predictor.db.repos.sensor_config import get as get_config

    registry = get_registry()
    redis = connection.app.state.redis

    async with AsyncSessionLocal() as session:
        config = await get_config(session, source, sensor, axis)

    if config is None:
        return {
            "source": source,
            "sensor": sensor,
            "axis": axis,
            "timeframes": {},
        }

    timeframes = json.loads(config.timeframes)
    if not timeframes:
        return {
            "source": source,
            "sensor": sensor,
            "axis": axis,
            "timeframes": {},
        }

    result: dict[str, dict] = {}
    for tf_str in timeframes:
        buf = registry.get_buffer(source, sensor, axis, tf_str)
        if buf is None:
            result[tf_str] = {"status": ModelState.WAITING.value}
            continue

        state = await buf.get_state()
        if state == ModelState.READY:
            raw = await redis.hget(f"state:{source}:{sensor}:{axis}:{tf_str}", "is_critical")
            raw_value = raw.decode() if isinstance(raw, bytes) else raw
            result[tf_str] = {
                "status": ModelState.READY.value,
                "is_critical": raw_value == "1",
            }
        elif state == ModelState.GATHERING:
            gathered_count = await buf.get_gathered_count()
            progress_percent = 0.0
            if timeframes[tf_str]["num_train_samples"] > 0:
                progress_percent = min(100.0, round(gathered_count / timeframes[tf_str]["num_train_samples"] * 100, 2))
            result[tf_str] = {
                "status": state.value,
                "gathered_count": gathered_count,
                "num_train_samples": timeframes[tf_str]["num_train_samples"],
                "progress_percent": progress_percent,
            }
        else:
            result[tf_str] = {"status": state.value}

    return {
        "source": source,
        "sensor": sensor,
        "axis": axis,
        "timeframes": result,
    }
