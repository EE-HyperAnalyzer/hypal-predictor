from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from hypal_predictor.db.models import SensorConfig


async def get(
    session: AsyncSession,
    source: str,
    sensor: str,
    axis: str,
) -> SensorConfig | None:
    result = await session.execute(
        select(SensorConfig).where(
            SensorConfig.source == source,
            SensorConfig.sensor == sensor,
            SensorConfig.axis == axis,
        )
    )
    return result.scalar_one_or_none()


async def get_all(session: AsyncSession) -> list[SensorConfig]:
    result = await session.execute(select(SensorConfig))
    return list(result.scalars().all())


async def upsert(
    session: AsyncSession,
    source: str,
    sensor: str,
    axis: str,
    **fields: object,
) -> SensorConfig:
    now = datetime.now(tz=timezone.utc).replace(tzinfo=None)

    fields.pop("created_at", None)
    fields.pop("updated_at", None)

    config = await get(session, source, sensor, axis)

    if config is None:
        config = SensorConfig(
            source=source,
            sensor=sensor,
            axis=axis,
            created_at=now,
            updated_at=now,
            **fields,
        )
        session.add(config)
    else:
        for key, value in fields.items():
            setattr(config, key, value)
        config.updated_at = now

    await session.commit()
    await session.refresh(config)
    return config


async def delete(
    session: AsyncSession,
    source: str,
    sensor: str,
    axis: str,
) -> bool:
    config = await get(session, source, sensor, axis)
    if config is None:
        return False
    await session.delete(config)
    await session.commit()
    return True
