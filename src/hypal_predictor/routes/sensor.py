import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from hypal_utils.timeframe import Timeframe
from sqlalchemy.ext.asyncio import AsyncSession

from hypal_predictor.core.model_store import model_store
from hypal_predictor.core.registry import get_registry
from hypal_predictor.db.engine import get_session
from hypal_predictor.db.repos import sensor_config as sc_repo
from hypal_predictor.schemas.config import SensorConfigRequest, SensorConfigResponse, TimeframeSettings
from hypal_predictor.schemas.status import TimeframeStatus

router = APIRouter(prefix="/sensor", tags=["Sensor"])
logger = logging.getLogger(__name__)


@router.post("/{source}/{sensor}/{axis}/config", response_model=SensorConfigResponse)
async def upsert_config(
    source: str,
    sensor: str,
    axis: str,
    body: SensorConfigRequest,
    session: AsyncSession = Depends(get_session),
):
    """Создаёт или обновляет конфигурацию сенсора. Регистрирует буферы."""
    for tf in body.timeframes.keys():
        try:
            Timeframe.from_str(tf)
        except Exception:
            raise HTTPException(status_code=422, detail=f"Invalid timeframe: {tf!r}")

    record = await sc_repo.upsert(
        session=session,
        source=source,
        sensor=sensor,
        axis=axis,
        timeframes=json.dumps(body.timeframes),
    )

    registry = get_registry()
    registry.register(
        source=source,
        sensor=sensor,
        axis=axis,
        timeframe_settings=body.timeframes,
    )

    return SensorConfigResponse(
        source=source,
        sensor=sensor,
        axis=axis,
        timeframes=json.loads(record.timeframes),
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.get("/{source}/{sensor}/{axis}/config", response_model=SensorConfigResponse)
async def get_config(
    source: str,
    sensor: str,
    axis: str,
    session: AsyncSession = Depends(get_session),
):
    """Возвращает текущую конфигурацию сенсора."""
    record = await sc_repo.get(session, source, sensor, axis)
    if record is None:
        raise HTTPException(404, f"Sensor {source}:{sensor}:{axis} not found")

    timeframes = json.loads(record.timeframes)
    if isinstance(timeframes, list):
        timeframes = {}
    return SensorConfigResponse(
        source=source,
        sensor=sensor,
        axis=axis,
        timeframes=timeframes,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.delete("/{source}/{sensor}/{axis}", status_code=204)
async def delete_sensor(
    source: str,
    sensor: str,
    axis: str,
    session: AsyncSession = Depends(get_session),
):
    """Удаляет сенсор, все модели и буферы."""
    deleted = await sc_repo.delete(session, source, sensor, axis)
    if not deleted:
        raise HTTPException(404, f"Sensor {source}:{sensor}:{axis} not found")

    registry = get_registry()
    for tf_str, _ in registry.get_all_buffers(source, sensor, axis):
        model_store.delete(source, sensor, axis, tf_str)
    registry.remove(source, sensor, axis)


@router.post("/{source}/{sensor}/{axis}/reset", status_code=200)
async def reset_sensor(
    source: str,
    sensor: str,
    axis: str,
    session: AsyncSession = Depends(get_session),
):
    """Сбрасывает все таймфреймы сенсора в WAITING."""
    record = await sc_repo.get(session, source, sensor, axis)
    if record is None:
        raise HTTPException(404, f"Sensor {source}:{sensor}:{axis} not found")

    registry = get_registry()
    for tf_str, buf in registry.get_all_buffers(source, sensor, axis):
        await buf.reset()

    return {"status": "reset", "source": source, "sensor": sensor, "axis": axis}


@router.get("/", response_model=list[SensorConfigResponse])
async def list_sensors(session: AsyncSession = Depends(get_session)):
    """Возвращает список всех зарегистрированных сенсоров."""
    records = await sc_repo.get_all(session)
    result = []
    for r in records:
        tfs = json.loads(r.timeframes)
        if isinstance(tfs, list):
            tfs = {}
        result.append(
            SensorConfigResponse(
                source=r.source,
                sensor=r.sensor,
                axis=r.axis,
                timeframes=tfs,
                created_at=r.created_at,
                updated_at=r.updated_at,
            )
        )
    return result


@router.post("/{source}/{sensor}/{axis}/timeframe/{timeframe}", response_model=SensorConfigResponse)
async def add_timeframe(
    source: str,
    sensor: str,
    axis: str,
    timeframe: str,
    body: TimeframeSettings,
    session: AsyncSession = Depends(get_session),
):
    try:
        tf = Timeframe.from_str(timeframe)
    except Exception:
        raise HTTPException(status_code=422, detail=f"Invalid timeframe: {tf!r}")

    record = await sc_repo.get(session, source, sensor, axis)
    if record is None:
        raise HTTPException(404, detail=f"Sensor {source}:{sensor}:{axis} not found")

    current_tfs = json.loads(record.timeframes)
    if not isinstance(current_tfs, dict):
        current_tfs = {}

    if timeframe in current_tfs:
        raise HTTPException(409, detail=f"Timeframe {tf} already exists for this sensor")

    current_tfs[timeframe] = body.model_dump()

    updated_record = await sc_repo.upsert(
        session=session,
        source=source,
        sensor=sensor,
        axis=axis,
        timeframes=json.dumps(current_tfs),
    )

    registry = get_registry()
    registry.register(
        source=source,
        sensor=sensor,
        axis=axis,
        timeframe_settings={str(tf): body},
    )

    return SensorConfigResponse(
        source=source,
        sensor=sensor,
        axis=axis,
        timeframes=current_tfs,
        created_at=updated_record.created_at,
        updated_at=updated_record.updated_at,
    )


@router.delete("/{source}/{sensor}/{axis}/timeframe/{timeframe}", status_code=204)
async def remove_timeframe(
    source: str,
    sensor: str,
    axis: str,
    timeframe: str,
    session: AsyncSession = Depends(get_session),
):
    """Удаляет таймфрейм из сенсора."""
    record = await sc_repo.get(session, source, sensor, axis)
    if record is None:
        raise HTTPException(404, detail=f"Sensor {source}:{sensor}:{axis} not found")

    current_tfs = json.loads(record.timeframes)
    if not isinstance(current_tfs, dict):
        current_tfs = {}

    if timeframe not in current_tfs:
        raise HTTPException(404, detail=f"Timeframe {timeframe} not found for this sensor")

    del current_tfs[timeframe]

    model_store.delete(source, sensor, axis, timeframe)
    registry = get_registry()
    registry.remove_timeframe(source, sensor, axis, timeframe)

    if current_tfs:
        await sc_repo.upsert(
            session=session,
            source=source,
            sensor=sensor,
            axis=axis,
            timeframes=json.dumps(current_tfs),
        )
    else:
        await sc_repo.delete(session, source, sensor, axis)
        registry.remove(source, sensor, axis)


@router.get("/{source}/{sensor}/{axis}/timeframe/{timeframe}", response_model=TimeframeStatus)
async def get_timeframe_status(
    source: str,
    sensor: str,
    axis: str,
    timeframe: str,
    session: AsyncSession = Depends(get_session),
):
    record = await sc_repo.get(session, source, sensor, axis)
    if record is None:
        raise HTTPException(404, detail=f"Sensor {source}:{sensor}:{axis} not found")

    registry = get_registry()
    buffer = registry.get_buffer(source, sensor, axis, timeframe)
    if buffer is None:
        raise HTTPException(404, detail=f"Timeframe {timeframe} for {source}:{sensor}:{axis} not found")

    state = await buffer.get_state()
    gathered_count = await buffer.get_gathered_count()

    record = json.loads(record.timeframes)[timeframe]
    return TimeframeStatus(
        model_type=record["model_type"],
        input_horizon=record["input_horizon"],
        output_horizon=record["output_horizon"],
        rollout_multiplier=record["rollout_multiplier"],
        critical_zone=record["critical_zone"],
        state=state,
        candles_gathered=gathered_count,
        num_train_samples=record["num_train_samples"],
        is_in_critical_zone=False,
    )
