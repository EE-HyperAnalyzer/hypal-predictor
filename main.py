from contextlib import asynccontextmanager

from fastapi import FastAPI

from hypal_predictor.routes.candle import router as candle_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan, title="Hypal Predictor")

app.include_router(candle_router)


@app.post("/ping")
async def ping():
    return "pong"


if __name__ == "__main__":
    import uvicorn

    from hypal_predictor import HOST_ADDRESS, HOST_PORT

    uvicorn.run(app, host=HOST_ADDRESS, port=HOST_PORT)
