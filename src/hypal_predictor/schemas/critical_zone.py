from typing import Annotated, Union

from hypal_utils.critical_zone.rule import (
    ZoneRule,
    ZoneRule_AND,
    ZoneRule_GREATER,
    ZoneRule_LESS,
    ZoneRule_NOT,
    ZoneRule_OR,
)
from pydantic import Field

CriticalZoneRuleDTO = Annotated[
    Union[ZoneRule, ZoneRule_AND, ZoneRule_GREATER, ZoneRule_LESS, ZoneRule_NOT, ZoneRule_OR],
    Field(discriminator="type"),
]

# Rebuild forward references for recursive models
ZoneRule_AND.model_rebuild()
ZoneRule_OR.model_rebuild()
ZoneRule_NOT.model_rebuild()
