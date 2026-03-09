from __future__ import annotations

import json
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from hypal_predictor.critical_zone import (
    ZoneRule,
    ZoneRule_AND,
    ZoneRule_GREATER,
    ZoneRule_LESS,
    ZoneRule_NOT,
    ZoneRule_OR,
)


class LessRule(BaseModel):
    type: Literal["LESS"]
    value: float


class GreaterRule(BaseModel):
    type: Literal["GREATER"]
    value: float


class AndRule(BaseModel):
    type: Literal["AND"]
    lhs: CriticalZoneRuleDTO
    rhs: CriticalZoneRuleDTO


class OrRule(BaseModel):
    type: Literal["OR"]
    lhs: CriticalZoneRuleDTO
    rhs: CriticalZoneRuleDTO


class NotRule(BaseModel):
    type: Literal["NOT"]
    rule: CriticalZoneRuleDTO


CriticalZoneRuleDTO = Annotated[
    Union[LessRule, GreaterRule, AndRule, OrRule, NotRule],
    Field(discriminator="type"),
]

# Rebuild forward references for recursive models
AndRule.model_rebuild()
OrRule.model_rebuild()
NotRule.model_rebuild()


def parse_rule(dto: CriticalZoneRuleDTO) -> ZoneRule:
    """Convert a CriticalZoneRuleDTO into a ZoneRule domain object."""
    match dto:
        case LessRule(value=v):
            return ZoneRule_LESS(v)
        case GreaterRule(value=v):
            return ZoneRule_GREATER(v)
        case AndRule(lhs=lhs, rhs=rhs):
            return ZoneRule_AND(parse_rule(lhs), parse_rule(rhs))
        case OrRule(lhs=lhs, rhs=rhs):
            return ZoneRule_OR(parse_rule(lhs), parse_rule(rhs))
        case NotRule(rule=r):
            return ZoneRule_NOT(parse_rule(r))
        case _:
            raise ValueError(f"Unknown rule DTO type: {type(dto)!r}")


def serialize_rule(rule: ZoneRule) -> CriticalZoneRuleDTO:
    """Convert a ZoneRule domain object into a CriticalZoneRuleDTO."""
    match rule:
        case ZoneRule_LESS(value=v):
            return LessRule(type="LESS", value=v)
        case ZoneRule_GREATER(value=v):
            return GreaterRule(type="GREATER", value=v)
        case ZoneRule_AND(lhs=lhs, rhs=rhs):
            return AndRule(type="AND", lhs=serialize_rule(lhs), rhs=serialize_rule(rhs))
        case ZoneRule_OR(lhs=lhs, rhs=rhs):
            return OrRule(type="OR", lhs=serialize_rule(lhs), rhs=serialize_rule(rhs))
        case ZoneRule_NOT(rule=r):
            return NotRule(type="NOT", rule=serialize_rule(r))
        case _:
            raise ValueError(f"Unknown ZoneRule type: {type(rule)!r}")


def rule_to_json(rule: ZoneRule) -> str:
    """Serialize a ZoneRule domain object to a JSON string for DB storage."""
    dto = serialize_rule(rule)
    return dto.model_dump_json()


def rule_from_json(raw: str) -> ZoneRule:
    """Deserialize a ZoneRule domain object from a JSON string (from DB)."""
    data = json.loads(raw)
    # Use LessRule as a temporary anchor; TypeAdapter handles discriminated union
    from pydantic import TypeAdapter

    adapter: TypeAdapter[CriticalZoneRuleDTO] = TypeAdapter(CriticalZoneRuleDTO)
    dto = adapter.validate_python(data)
    return parse_rule(dto)
