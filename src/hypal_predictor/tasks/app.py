from celery import Celery

from hypal_predictor.config import settings

celery_app = Celery(
    "hypal_predictor",
    broker=settings.redis_url,
    backend=settings.redis_result_url,
    include=[
        "hypal_predictor.tasks.training",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_concurrency=settings.max_parallel_training,
    task_routes={
        "hypal_predictor.tasks.training.*": {"queue": "training"},
    },
)
