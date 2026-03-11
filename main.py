import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from redis.asyncio import Redis

from hypal_predictor import HOST_ADDRESS, HOST_PORT
from hypal_predictor.config import settings
from hypal_predictor.core.predictor import PredictorEngine
from hypal_predictor.core.registry import init_registry
from hypal_predictor.core.signal_aggregator import SignalAggregator
from hypal_predictor.db.engine import AsyncSessionLocal, init_db
from hypal_predictor.db.repos.sensor_config import get_all as get_all_sensors
from hypal_predictor.routes.candle import router as candle_router
from hypal_predictor.routes.critical_zone import router as critical_zone_router
from hypal_predictor.routes.health import router as health_router
from hypal_predictor.routes.sensor import router as sensor_router
from hypal_predictor.routes.status import router as status_router
from hypal_predictor.schemas.config import TimeframeSettings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── 1. Database ────────────────────────────────────────────────────────
    logger.info("Initializing database…")
    await init_db()

    # ── 2. Redis ───────────────────────────────────────────────────────────
    logger.info("Connecting to Redis at %s…", settings.redis_url)
    redis: Redis = Redis.from_url(settings.redis_url, decode_responses=False)
    await redis.ping()
    app.state.redis = redis

    # ── 3. Core services ───────────────────────────────────────────────────
    registry = init_registry(redis)
    app.state.predictor = PredictorEngine(redis=redis)
    app.state.signal_aggregator = SignalAggregator(redis=redis)

    # ── 4. Restore sensor configs from DB ──────────────────────────────────
    logger.info("Restoring sensor configurations from DB…")
    async with AsyncSessionLocal() as session:
        configs = await get_all_sensors(session)

    for cfg in configs:
        try:
            timeframe_settings_json = json.loads(cfg.timeframes)
            timeframe_settings = {tf: TimeframeSettings(**settings) for tf, settings in timeframe_settings_json.items()}
            registry.register(
                source=cfg.source,
                sensor=cfg.sensor,
                axis=cfg.axis,
                timeframe_settings=timeframe_settings,
            )
            logger.info(
                "Restored sensor %s:%s:%s with timeframes %s",
                cfg.source,
                cfg.sensor,
                cfg.axis,
                list(timeframe_settings.keys()),
            )
        except Exception:
            logger.exception(
                "Failed to restore sensor %s:%s:%s",
                cfg.source,
                cfg.sensor,
                cfg.axis,
            )

    logger.info("Startup complete. %d sensor(s) restored.", len(configs))

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────
    logger.info("Shutting down…")
    await redis.aclose()
    logger.info("Redis connection closed.")


app = FastAPI(
    lifespan=lifespan,
    title="Hypal Predictor",
    description=(
        "API-сервис для предсказания значений сенсоров в формате OHLC-свечей. "
        "Поддерживает несколько таймфреймов, асинхронное обучение через Celery "
        "и детекцию критических зон."
    ),
    version="0.2.0",
)

app.include_router(health_router)
app.include_router(candle_router)
app.include_router(sensor_router)
app.include_router(critical_zone_router)
app.include_router(status_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=HOST_ADDRESS,
        port=HOST_PORT,
        reload=False,
        log_level=settings.log_level.lower(),
    )
