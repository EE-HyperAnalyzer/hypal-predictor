import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from hypal_predictor.db.engine import get_session
from hypal_predictor.db.repos import critical_zone as cz_repo
from hypal_predictor.db.repos import sensor_config as sc_repo
from hypal_predictor.schemas.critical_zone import (
    CriticalZoneRuleDTO,
    parse_rule,
    rule_from_json,
    rule_to_json,
)

router = APIRouter(prefix="/sensor", tags=["CriticalZone"])
logger = logging.getLogger(__name__)


@router.put("/{source}/{sensor}/{axis}/critical-zone")
async def set_critical_zone(
    source: str,
    sensor: str,
    axis: str,
    body: CriticalZoneRuleDTO,
    session: AsyncSession = Depends(get_session),
):
    config = await sc_repo.get(session, source, sensor, axis)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail=f"Sensor {source}:{sensor}:{axis} not found",
        )

    rule_json = rule_to_json(parse_rule(body))
    await cz_repo.upsert(session, source, sensor, axis, rule_json)
    return {"status": "ok", "rule": body}


@router.get("/{source}/{sensor}/{axis}/critical-zone")
async def get_critical_zone(
    source: str,
    sensor: str,
    axis: str,
    session: AsyncSession = Depends(get_session),
):
    """Возвращает текущее правило критической зоны для сенсора."""
    record = await cz_repo.get(session, source, sensor, axis)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"No critical zone rule for {source}:{sensor}:{axis}",
        )
    dto = rule_from_json(record.rule_json)
    return {"rule": dto}
