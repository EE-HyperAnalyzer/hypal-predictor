from fastapi import APIRouter, Request

router = APIRouter(tags=["Health"])


@router.get("/ping")
async def ping():
    return {"status": "ok"}


@router.get("/health")
async def health(request: Request):
    redis_ok = False
    try:
        redis_ok = await request.app.state.redis.ping()
    except Exception:
        pass
    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": redis_ok,
        "version": "0.2.0",
    }
