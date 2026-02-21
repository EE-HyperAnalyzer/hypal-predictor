from fastapi import WebSocket
from fastapi.routing import APIRouter
from hypal_utils.logger import log_info
from hypal_utils.sensor_data import SensorData

from src.model_m import get_model_manager

router = APIRouter(prefix="/candle", tags=["Candle"])


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    model_manager = get_model_manager()

    while True:
        try:
            raw_json_data = await websocket.receive_json()
            sensor_data = SensorData(**raw_json_data)

            # log_info(f"recv: {sensor_data}")

            id_ = f"{sensor_data.source}:{sensor_data.sensor}:{sensor_data.axis}"
            log_info(id_)
            if id_ not in model_manager.models:
                log_info(f"Added new timeframe for: {id_}")
                model_manager.add_timeframe(sensor_id=id_, timeframe="1:s")

            log_info(f"Consume!: {id_}")
            model_manager.consume(sensor_data)

        except ConnectionError as e:
            log_info(f"Connection closed: {e}")
            await websocket.close()
