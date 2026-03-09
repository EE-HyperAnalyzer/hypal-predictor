import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from hypal_predictor.core.model_store import model_store
from hypal_predictor.core.registry import get_registry
from hypal_predictor.db.engine import get_session
from hypal_predictor.db.repos import sensor_config as sc_repo
from hypal_predictor.schemas.config import SensorConfigRequest, SensorConfigResponse
from hypal_predictor.timeframe import Timeframe

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
    for tf in body.timeframes:
        try:
            Timeframe.from_str(tf)
        except Exception:
            raise HTTPException(status_code=422, detail=f"Invalid timeframe: {tf!r}")

    now = datetime.now(tz=timezone.utc)
    record = await sc_repo.upsert(
        session=session,
        source=source,
        sensor=sensor,
        axis=axis,
        timeframes=json.dumps(body.timeframes),
        input_horizon=body.input_horizon,
        output_horizon=body.output_horizon,
        rollout_multiplier=body.rollout_multiplier,
        num_train_samples=body.num_train_samples,
        model_type=body.model_type,
        core_api_url=body.core_api_url,
        updated_at=now,
    )

    registry = get_registry()
    registry.register(
        source=source,
        sensor=sensor,
        axis=axis,
        timeframes=body.timeframes,
        num_train_samples=body.num_train_samples,
    )

    return SensorConfigResponse(
        source=source,
        sensor=sensor,
        axis=axis,
        timeframes=body.timeframes,
        input_horizon=record.input_horizon,
        output_horizon=record.output_horizon,
        rollout_multiplier=record.rollout_multiplier,
        num_train_samples=record.num_train_samples,
        model_type=record.model_type,
        core_api_url=record.core_api_url,
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
    return SensorConfigResponse(
        source=source,
        sensor=sensor,
        axis=axis,
        timeframes=timeframes,
        input_horizon=record.input_horizon,
        output_horizon=record.output_horizon,
        rollout_multiplier=record.rollout_multiplier,
        num_train_samples=record.num_train_samples,
        model_type=record.model_type,
        core_api_url=record.core_api_url,
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
        result.append(
            SensorConfigResponse(
                source=r.source,
                sensor=r.sensor,
                axis=r.axis,
                timeframes=tfs,
                input_horizon=r.input_horizon,
                output_horizon=r.output_horizon,
                rollout_multiplier=r.rollout_multiplier,
                num_train_samples=r.num_train_samples,
                model_type=r.model_type,
                core_api_url=r.core_api_url,
                created_at=r.created_at,
                updated_at=r.updated_at,
            )
        )
    return result
