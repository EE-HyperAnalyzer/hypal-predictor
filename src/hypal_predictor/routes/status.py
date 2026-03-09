import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from hypal_predictor.core.registry import get_registry
from hypal_predictor.db.engine import get_session
from hypal_predictor.db.repos import sensor_config as sc_repo
from hypal_predictor.db.repos import training_history as th_repo
from hypal_predictor.schemas.prediction import PredictionResponse, SignalResponse
from hypal_predictor.schemas.status import SensorStatusResponse, TimeframeMetrics, TimeframeStatus

router = APIRouter(prefix="/sensor", tags=["Status"])
logger = logging.getLogger(__name__)


@router.get("/{source}/{sensor}/{axis}/status", response_model=SensorStatusResponse)
async def get_status(
    source: str,
    sensor: str,
    axis: str,
    session: AsyncSession = Depends(get_session),
):
    config = await sc_repo.get(session, source, sensor, axis)
    if config is None:
        raise HTTPException(404, f"Sensor {source}:{sensor}:{axis} not found")

    registry = get_registry()
    timeframe_statuses: dict[str, TimeframeStatus] = {}

    for tf_str, buf in registry.get_all_buffers(source, sensor, axis):
        state = await buf.get_state()
        count = await buf.get_gathered_count()

        latest = await th_repo.get_latest(session, source, sensor, axis, tf_str)
        metrics = None
        if latest and latest.mse is not None:
            metrics = TimeframeMetrics(mse=latest.mse, mae=latest.mae, r2=latest.r2)

        timeframe_statuses[tf_str] = TimeframeStatus(
            state=state,
            candles_gathered=count,
            num_train_samples=config.num_train_samples,
            last_trained_at=latest.finished_at if latest else None,
            is_in_critical_zone=False,
            metrics=metrics,
        )

    return SensorStatusResponse(
        source=source,
        sensor=sensor,
        axis=axis,
        timeframes=timeframe_statuses,
    )


@router.get("/{source}/{sensor}/{axis}/prediction", response_model=PredictionResponse)
async def get_prediction(
    source: str,
    sensor: str,
    axis: str,
    request: Request,
):
    registry = get_registry()
    predictor = request.app.state.predictor
    predictions: dict = {}

    for tf_str, buf in registry.get_all_buffers(source, sensor, axis):
        candles = await predictor.get_last_prediction(source, sensor, axis, tf_str)
        if candles:
            predictions[tf_str] = candles

    if not predictions:
        raise HTTPException(404, "No predictions available yet")

    return PredictionResponse(
        source=source,
        sensor=sensor,
        axis=axis,
        predictions=predictions,
    )


@router.get("/{source}/{sensor}/{axis}/signal", response_model=SignalResponse)
async def get_signal(
    source: str,
    sensor: str,
    axis: str,
    request: Request,
):
    signal_agg = request.app.state.signal_aggregator
    predictor = request.app.state.predictor
    registry = get_registry()

    current = await signal_agg.get_current_signal(source, sensor, axis)

    predictions: dict = {}
    for tf_str, buf in registry.get_all_buffers(source, sensor, axis):
        candles = await predictor.get_last_prediction(source, sensor, axis, tf_str)
        if candles:
            predictions[tf_str] = candles

    return SignalResponse(
        source=source,
        sensor=sensor,
        axis=axis,
        is_critical=current["is_critical"],
        triggered_timeframes=current["triggered_timeframes"],
        timestamp=current["timestamp"],
        predictions=predictions,
    )
