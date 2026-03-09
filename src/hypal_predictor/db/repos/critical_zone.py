from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from hypal_predictor.db.models import CriticalZoneRule


async def get(
    session: AsyncSession,
    source: str,
    sensor: str,
    axis: str,
) -> CriticalZoneRule | None:
    result = await session.execute(
        select(CriticalZoneRule).where(
            CriticalZoneRule.source == source,
            CriticalZoneRule.sensor == sensor,
            CriticalZoneRule.axis == axis,
        )
    )
    return result.scalar_one_or_none()


async def upsert(
    session: AsyncSession,
    source: str,
    sensor: str,
    axis: str,
    rule_json: str,
) -> CriticalZoneRule:
    existing = await get(session, source, sensor, axis)
    if existing is not None:
        await session.delete(existing)
        await session.flush()

    zone = CriticalZoneRule(
        source=source,
        sensor=sensor,
        axis=axis,
        rule_json=rule_json,
        created_at=datetime.now(tz=timezone.utc),
    )
    session.add(zone)
    await session.commit()
    await session.refresh(zone)
    return zone
