from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKeyConstraint, Integer, PrimaryKeyConstraint, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from hypal_predictor.db.engine import Base


class SensorConfig(Base):
    __tablename__ = "sensor_configs"

    source: Mapped[str] = mapped_column(String, nullable=False)
    sensor: Mapped[str] = mapped_column(String, nullable=False)
    axis: Mapped[str] = mapped_column(String, nullable=False)

    timeframes: Mapped[str] = mapped_column(Text, nullable=False)
    input_horizon: Mapped[int] = mapped_column(Integer, nullable=False)
    output_horizon: Mapped[int] = mapped_column(Integer, nullable=False)
    rollout_multiplier: Mapped[int] = mapped_column(Integer, nullable=False)
    num_train_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    model_type: Mapped[str] = mapped_column(String, nullable=False)
    core_api_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    __table_args__ = (PrimaryKeyConstraint("source", "sensor", "axis", name="pk_sensor_configs"),)

    def __repr__(self) -> str:
        return f"<SensorConfig source={self.source!r} sensor={self.sensor!r} axis={self.axis!r}>"


class CriticalZoneRule(Base):
    __tablename__ = "critical_zone_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    source: Mapped[str] = mapped_column(String, nullable=False)
    sensor: Mapped[str] = mapped_column(String, nullable=False)
    axis: Mapped[str] = mapped_column(String, nullable=False)

    rule_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            ["source", "sensor", "axis"],
            [
                "sensor_configs.source",
                "sensor_configs.sensor",
                "sensor_configs.axis",
            ],
            name="fk_critical_zone_rules_sensor_config",
            ondelete="CASCADE",
        ),
    )

    def __repr__(self) -> str:
        return f"<CriticalZoneRule id={self.id} source={self.source!r} sensor={self.sensor!r} axis={self.axis!r}>"


class TrainingHistory(Base):
    __tablename__ = "training_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    source: Mapped[str] = mapped_column(String, nullable=False)
    sensor: Mapped[str] = mapped_column(String, nullable=False)
    axis: Mapped[str] = mapped_column(String, nullable=False)
    timeframe: Mapped[str] = mapped_column(String, nullable=False)

    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    mse: Mapped[Optional[float]] = mapped_column(nullable=True)
    mae: Mapped[Optional[float]] = mapped_column(nullable=True)
    r2: Mapped[Optional[float]] = mapped_column(nullable=True)

    model_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<TrainingHistory id={self.id} source={self.source!r} "
            f"sensor={self.sensor!r} axis={self.axis!r} timeframe={self.timeframe!r}>"
        )
