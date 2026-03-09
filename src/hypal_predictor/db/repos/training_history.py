from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from hypal_predictor.db.models import TrainingHistory


async def add(
    session: AsyncSession,
    source: str,
    sensor: str,
    axis: str,
    timeframe: str,
    started_at: datetime,
    **optional_fields: object,
) -> TrainingHistory:
    record = TrainingHistory(
        source=source,
        sensor=sensor,
        axis=axis,
        timeframe=timeframe,
        started_at=started_at,
        **optional_fields,
    )
    session.add(record)
    await session.commit()
    await session.refresh(record)
    return record


async def finish(
    session: AsyncSession,
    record_id: int,
    finished_at: datetime,
    mse: Optional[float],
    mae: Optional[float],
    r2: Optional[float],
    model_path: Optional[str],
) -> TrainingHistory | None:
    result = await session.execute(select(TrainingHistory).where(TrainingHistory.id == record_id))
    record = result.scalar_one_or_none()
    if record is None:
        return None

    record.finished_at = finished_at
    record.mse = mse
    record.mae = mae
    record.r2 = r2
    record.model_path = model_path

    await session.commit()
    await session.refresh(record)
    return record


async def get_latest(
    session: AsyncSession,
    source: str,
    sensor: str,
    axis: str,
    timeframe: str,
) -> TrainingHistory | None:
    result = await session.execute(
        select(TrainingHistory)
        .where(
            TrainingHistory.source == source,
            TrainingHistory.sensor == sensor,
            TrainingHistory.axis == axis,
            TrainingHistory.timeframe == timeframe,
        )
        .order_by(TrainingHistory.started_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()
