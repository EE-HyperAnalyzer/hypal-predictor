from celery import Celery

app = Celery("training_tasks", broker="redis://localhost:6379/0")


@app.task
def train_model_task(data):
    # Код обучения модели
    print(f"Обучение на {len(data)} примерах...")
    # ... ваш код обучения ...
    return "Модель обучена"
