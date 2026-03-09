# Hypal Predictor — Implementation Plan

## 1. Анализ текущего состояния

### Что уже есть

| Компонент | Файл | Статус |
|---|---|---|
| FastAPI app | `main.py` | Заглушка, только `/ping` |
| BaseModel | `model/base.py` | Готов — `fit()`, `predict()`, rollout |
| ScikitLearn-модели | `model/scikit_compat/` | Готовы — LinearRegression, CatBoost |
| Torch-модели | `model/torch_compat/` | Готов Transformer |
| CriticalZone правила | `critical_zone.py` | Готово — LESS, GREATER, AND, OR, NOT |
| Timeframe | `timeframe.py` | Готов — парсинг, конвертация в секунды |
| Databag / AxisDatabag | `dbag.py` | Готова агрегация свечей по таймфрейму |
| Метрики | `metrics/` | Готовы — Regression + Classification |
| Celery worker | `worker.py` | Заглушка |
| WebSocket `/candle/ws` | `routes/candle.py` | Заглушка |
| Router `/timeframe` | `routes/timeframe.py` | Пустой |

### Что отсутствует

- Персистентность конфигурации (БД)
- State machine для каждой тройки `(source, sensor, axis)` × `timeframe`
- Реальный pipeline: накопление → агрегация → обучение → предсказание
- JSON DSL для критической зоны (настройка через API)
- Агрегация сигналов по всем таймфреймам для одного сенсора
- Уведомление CoreAPI при смене состояния критической зоны
- LRU-кеш моделей в памяти + выгрузка на диск

---

## 2. Целевая архитектура

```
CoreAPI ──── POST /candle/ingest ────► CandleIngestor
             WS   /candle/ws                │
                                            │  raw 1-second OHLC candles
                                            ▼
                                    SensorRegistry
                                    (in-memory + Redis)
                                            │
                          ┌─────────────────┼──────────────────┐
                          ▼                 ▼                  ▼
                    TimeframeBuffer   TimeframeBuffer   TimeframeBuffer
                       (tf=1:m)         (tf=5:m)          (tf=1:h)
                       [GATHERING]     [GATHERING]        [TRAINING]
                          │                 │                  │
                     num_train_samples достигнут?              │
                          │                 │                  │
                          ▼                 ▼                  ▼
                       Celery Training Queue (Redis broker)
                                            │
                                     train_model task
                                     (неблокирующее)
                                            │
                                     ModelStore (disk)
                                            │
                                     TRAINING → READY
                                            │
                                     PredictorEngine
                                            │
                                    CriticalZoneChecker
                                            │
                          ┌─────────────────┼──────────────────┐
                          ▼                 ▼                  ▼
                    signal(tf=1:m)   signal(tf=5:m)   signal(tf=1:h)
                          └─────────────────┴──────────────────┘
                                            │
                                    SignalAggregator
                                    (OR по всем таймфреймам)
                                            │
                                  изменение состояния?
                                    (ENTERED / EXITED)
                                            │
                                  POST CoreAPI /signal
```

---

## 3. State Machine

Для каждой комбинации `(source, sensor, axis, timeframe)` — независимая машина состояний.

```
              ┌──────────┐
  конфиг      │          │
  получен     │ WAITING  │
  ──────────► │          │
              └────┬─────┘
                   │ первая агрегированная свеча получена
                   ▼
              ┌──────────┐
              │          │  накапливаем агрегированные свечи
              │GATHERING │◄───────────────────────────────┐
              │          │  gathered_count < num_train_samples
              └────┬─────┘
                   │ gathered_count == num_train_samples
                   ▼
              ┌──────────┐
              │          │  Celery task запущен (неблокирующее)
              │ TRAINING │  новые свечи продолжают копиться
              │          │
              └────┬─────┘
                   │ Celery task завершён успешно
                   ▼
              ┌──────────┐
              │          │  предсказание + детекция критической зоны
              │  READY   │  на каждой новой агрегированной свече
              │          │
              └──────────┘
                   │
                   │ сброс / реконфигурация
                   ▼
               WAITING
```

**Ключевые правила:**
- В состоянии `TRAINING` новые агрегированные свечи продолжают
  накапливаться, чтобы после перехода в `READY` модель сразу
  имела актуальные данные.
- В состоянии `READY` модель переобучается периодически:
  каждые `num_train_samples` новых свечей запускается новый
  Celery task (retrain). Старая модель при этом продолжает
  работать до завершения обучения новой.
- При изменении конфигурации (`POST /sensor/{id}/config`) все
  таймфреймы сбрасываются в `WAITING`, накопленные буферы очищаются.

---

## 4. API Endpoints

### 4.1 Ingestion (прием свечей)

| Метод | Путь | Описание |
|---|---|---|
| `POST` | `/candle/ingest` | Принять батч 1-секундных OHLC-свечей |
| `WS` | `/candle/ws` | Стриминг 1-секундных свечей через WebSocket |

### 4.2 Конфигурация сенсора

| Метод | Путь | Описание |
|---|---|---|
| `POST` | `/sensor/{source}/{sensor}/{axis}/config` | Создать / обновить конфигурацию сенсора |
| `GET` | `/sensor/{source}/{sensor}/{axis}/config` | Получить текущую конфигурацию |
| `DELETE` | `/sensor/{source}/{sensor}/{axis}` | Удалить сенсор и все его модели |
| `POST` | `/sensor/{source}/{sensor}/{axis}/reset` | Сбросить все модели в состояние WAITING |

### 4.3 Критическая зона

| Метод | Путь | Описание |
|---|---|---|
| `PUT` | `/sensor/{source}/{sensor}/{axis}/critical-zone` | Задать правила критической зоны (JSON DSL) |
| `GET` | `/sensor/{source}/{sensor}/{axis}/critical-zone` | Получить текущие правила в JSON DSL |

### 4.4 Статус и предсказания

| Метод | Путь | Описание |
|---|---|---|
| `GET` | `/sensor/{source}/{sensor}/{axis}/status` | Состояние всех моделей (per timeframe) |
| `GET` | `/sensor/{source}/{sensor}/{axis}/prediction` | Последние предсказания (per timeframe) |
| `GET` | `/sensor/{source}/{sensor}/{axis}/signal` | Агрегированный сигнал критической зоны |
| `GET` | `/sensors` | Список всех зарегистрированных сенсоров |

### 4.5 Служебные

| Метод | Путь | Описание |
|---|---|---|
| `GET` | `/health` | Статус сервиса, Celery, Redis |
| `GET` | `/ping` | Простая проверка живости |

---

## 5. Pydantic-схемы

### 5.1 SensorConfig

`POST /sensor/{source}/{sensor}/{axis}/config`

```json
{
  "source": "car",
  "sensor": "accumulator",
  "axis": "temp",
  "timeframes": ["1:m", "5:m", "1:h"],
  "input_horizon": 10,
  "output_horizon": 5,
  "rollout_multiplier": 1,
  "num_train_samples": 500,
  "model_type": "linear",
  "core_api_url": "http://core-api:8000/signal"
}
```

**Параметры:**
- `source` — источник данных (например `"car"`)
- `sensor` — имя сенсора (например `"accumulator"`)
- `axis` — ось/канал сенсора (например `"temp"`)
- `timeframes` — список таймфреймов для обучения (формат `N:s/m/h/d`)
- `input_horizon` — длина входного окна (в агрегированных свечах)
- `output_horizon` — горизонт предсказания (в агрегированных свечах)
- `rollout_multiplier` — количество шагов авторегрессии при предсказании
- `num_train_samples` — минимальное кол-во агрегированных свечей для запуска обучения
- `model_type` — `"linear"` | `"catboost"` | `"transformer"`
- `core_api_url` — URL для POST-уведомлений (опционально)

### 5.2 CriticalZoneRule (JSON DSL)

`PUT /sensor/{source}/{sensor}/{axis}/critical-zone`

Рекурсивная структура, отражающая иерархию классов `ZoneRule`:

```json
{
  "type": "NOT",
  "rule": {
    "type": "AND",
    "lhs": { "type": "GREATER", "value": 20.0 },
    "rhs": { "type": "LESS",    "value": 25.0 }
  }
}
```

Поддерживаемые типы:
- `"LESS"`    — `{ "type": "LESS", "value": float }`
- `"GREATER"` — `{ "type": "GREATER", "value": float }`
- `"AND"`     — `{ "type": "AND", "lhs": Rule, "rhs": Rule }`
- `"OR"`      — `{ "type": "OR",  "lhs": Rule, "rhs": Rule }`
- `"NOT"`     — `{ "type": "NOT", "rule": Rule }`

### 5.3 CandleIngestRequest

`POST /candle/ingest`

```json
{
  "candles": [
    {
      "source": "car",
      "sensor": "accumulator",
      "axis": "temp",
      "timestamp": 1700000000,
      "candle": {
        "open": 22.1,
        "high": 22.4,
        "low": 21.9,
        "close": 22.3
      }
    }
  ]
}
```

Каждый элемент является объектом `SensorData` (`source`, `sensor`, `axis`, `timestamp`, `candle`).
Сервис сам выполняет агрегацию по таймфреймам на основе поля `timestamp`.

### 5.4 SensorStatus

`GET /sensor/{source}/{sensor}/{axis}/status`

```json
{
  "source": "car",
  "sensor": "accumulator",
  "axis": "temp",
  "timeframes": {
    "1:m": {
      "state": "READY",
      "candles_gathered": 1200,
      "num_train_samples": 500,
      "last_trained_at": "2025-01-01T12:00:00Z",
      "last_predicted_at": "2025-01-01T12:05:00Z",
      "is_in_critical_zone": false,
      "metrics": { "mse": 0.012, "mae": 0.087, "r2": 0.94 }
    },
    "5:m": {
      "state": "GATHERING",
      "candles_gathered": 134,
      "num_train_samples": 500,
      "last_trained_at": null,
      "last_predicted_at": null,
      "is_in_critical_zone": false,
      "metrics": null
    }
  }
}
```

### 5.5 SignalResponse

`GET /sensor/{source}/{sensor}/{axis}/signal`

```json
{
  "source": "car",
  "sensor": "accumulator",
  "axis": "temp",
  "is_critical": true,
  "triggered_timeframes": ["1:m"],
  "timestamp": 1700000300,
  "predictions": {
    "1:m": [
      {
        "open": 26.1,
        "high": 28.3,
        "low": 25.8,
        "close": 27.5
      }
    ]
  }
}
```

### 5.6 CoreAPI Notification (исходящий POST)

```json
{
  "source": "car",
  "sensor": "accumulator",
  "axis": "temp",
  "event": "CRITICAL_ENTERED",
  "timestamp": 1700000300,
  "triggered_timeframes": ["1:m"],
  "predictions": {
    "1:m": [
      {
        "open": 26.1,
        "high": 28.3,
        "low": 25.8,
        "close": 27.5
      }
    ]
  }
}
```

Возможные значения `event`: `"CRITICAL_ENTERED"`, `"CRITICAL_EXITED"`.

---

## 6. Логика агрегации сигналов

```python
# Для сенсора S с таймфреймами [tf1, tf2, tf3]:

per_tf_is_critical: dict[str, bool] = {
    "1:m": ready_model_1m.is_critical,   # True если хотя бы одна предсказанная свеча в зоне
    "5:m": ready_model_5m.is_critical,
    "1:h": ready_model_1h.is_critical,
}

# Только READY-модели участвуют в агрегации
ready_signals = {
    tf: val
    for tf, val in per_tf_is_critical.items()
    if model_state[tf] == "READY"
}

# Если ни одной READY-модели нет — сигнал не отправляем
if not ready_signals:
    return

# Агрегация: OR по всем READY-таймфреймам
current_signal: bool = any(ready_signals.values())
triggered = [tf for tf, v in ready_signals.items() if v]

# Детекция изменения состояния → уведомление CoreAPI
if not previous_signal and current_signal:
    notify_core_api(source, sensor, axis, event="CRITICAL_ENTERED", triggered_timeframes=triggered)
elif previous_signal and not current_signal:
    notify_core_api(source, sensor, axis, event="CRITICAL_EXITED", triggered_timeframes=[])

previous_signal = current_signal
```

---

## 7. Хранилище

### 7.1 Redis

| Ключ | Тип | Содержимое |
|---|---|---|
| `buf:tf:{source}:{sensor}:{axis}:{tf}` | List | Агрегированные свечи таймфрейма (JSON) |
| `state:{source}:{sensor}:{axis}:{tf}` | Hash | state, gathered_count, task_id, last_trained_at, metrics |
| `pred:{source}:{sensor}:{axis}:{tf}` | String | JSON последних предсказаний (список `Candle_OHLC`) |
| `signal:{source}:{sensor}:{axis}` | Hash | is_critical, triggered_tfs, timestamp |

Сырые 1-секундные свечи **не хранятся в Redis** — они обрабатываются
in-process в `CandleIngestor` и сразу агрегируются в буферы таймфреймов.

### 7.2 SQLite / PostgreSQL (через SQLAlchemy Async)

**`sensor_configs`** — конфигурация сенсора

```sql
source           TEXT NOT NULL
sensor           TEXT NOT NULL
axis             TEXT NOT NULL
timeframes       TEXT       -- JSON: ["1:m","5:m"]
input_horizon    INTEGER
output_horizon   INTEGER
rollout_mult     INTEGER
num_train_samp   INTEGER
model_type       TEXT       -- "linear" | "catboost" | "transformer"
core_api_url     TEXT
created_at       DATETIME
updated_at       DATETIME
PRIMARY KEY (source, sensor, axis)
```

**`critical_zone_rules`** — правила критической зоны

```sql
id           INTEGER PRIMARY KEY AUTOINCREMENT
source       TEXT NOT NULL
sensor       TEXT NOT NULL
axis         TEXT NOT NULL
rule_json    TEXT    -- JSON DSL
created_at   DATETIME
FOREIGN KEY (source, sensor, axis) REFERENCES sensor_configs(source, sensor, axis)
```

**`training_history`** — история обучений

```sql
id           INTEGER PRIMARY KEY AUTOINCREMENT
source       TEXT NOT NULL
sensor       TEXT NOT NULL
axis         TEXT NOT NULL
timeframe    TEXT
started_at   DATETIME
finished_at  DATETIME
mse          REAL
mae          REAL
r2           REAL
model_path   TEXT
```

### 7.3 Файловая система — сохранённые модели

```
models/
└── {source}/
    └── {sensor}/
        └── {axis}/
            ├── 1m.pkl
            ├── 5m.pkl
            └── 1h.pkl
```

Сериализация через `joblib` для sklearn/catboost и `torch.save` для Transformer.

---

## 8. Celery

### 8.1 Конфигурация

```python
# tasks/app.py
app = Celery(
    "hypal_predictor",
    broker=REDIS_URL,           # redis://localhost:6379/0
    backend=REDIS_RESULT_URL,   # redis://localhost:6379/1
)
app.conf.worker_concurrency = MAX_PARALLEL_TRAINING  # env, default=2
app.conf.task_routes = {
    "tasks.training.*": {"queue": "training"},
    "tasks.notify.*":   {"queue": "notify"},
}
```

### 8.2 Задача обучения

```python
@app.task(bind=True, name="tasks.training.train_model", max_retries=3)
def train_model(
    self,
    source: str,
    sensor: str,
    axis: str,
    timeframe: str,
    candles_json: list[dict],
):
    try:
        # 1. Обновить состояние: TRAINING (в Redis)
        # 2. Загрузить конфигурацию из БД
        # 3. Десериализовать свечи из candles_json
        # 4. Создать модель через ModelFactory
        # 5. model.fit(candles)
        # 6. Сохранить модель через ModelStore (на диск)
        # 7. Сохранить метрики в training_history (БД)
        # 8. Обновить состояние: READY (в Redis)
        # 9. Запустить первое предсказание
    except Exception as exc:
        # Обновить состояние: GATHERING (откат)
        raise self.retry(exc=exc, countdown=60)
```

### 8.3 Задача уведомления CoreAPI

```python
@app.task(name="tasks.notify.send_signal", max_retries=5)
def send_signal(core_api_url: str, payload: dict):
    # POST payload → core_api_url
    # Retry с экспоненциальным back-off при ошибках сети
```

---

## 9. Управление моделями в памяти (ModelStore)

```python
class ModelStore:
    """
    LRU-кеш обученных моделей в памяти.
    При превышении MAX_LOADED_MODELS вытесняет LRU-элемент
    (модель остается на диске, при следующем обращении
    подгружается обратно).
    """
    _cache: OrderedDict[tuple[str, str, str, str], BaseModel]  # (source, sensor, axis, tf)
    MAX_LOADED: int    # env MAX_LOADED_MODELS, default=10
    MODELS_DIR: Path   # env MODELS_DIR, default="models/"

    def get(self, source: str, sensor: str, axis: str, timeframe: str) -> BaseModel | None:
        ...  # из кеша или загружает с диска

    def put(self, source: str, sensor: str, axis: str, timeframe: str, model: BaseModel) -> None:
        ...  # сохраняет на диск и в кеш, вытесняет при overflow

    def evict(self, source: str, sensor: str, axis: str, timeframe: str) -> None:
        ...  # удаляет из кеша

    def delete(self, source: str, sensor: str, axis: str, timeframe: str) -> None:
        ...  # удаляет из кеша И с диска
```

---

## 10. Целевая структура файлов

```
src/hypal_predictor/
├── __init__.py
├── config.py                      # Settings через pydantic-settings (env-vars)
├── critical_zone.py               # (существует) ZoneRule-иерархия
├── timeframe.py                   # (существует) Timeframe dataclass
├── utils.py                       # (существует) helpers
│
├── schemas/                       # Pydantic-схемы запросов и ответов
│   ├── __init__.py
│   ├── candle.py                  # CandleItem, CandleIngestRequest
│   ├── config.py                  # SensorConfigRequest/Response
│   ├── critical_zone.py           # CriticalZoneRuleDTO (JSON DSL + парсер)
│   ├── prediction.py              # PredictionResponse, SignalResponse
│   └── status.py                  # SensorStatus, TimeframeStatus
│
├── db/                            # Персистентность
│   ├── __init__.py
│   ├── engine.py                  # SQLAlchemy async engine + session
│   ├── models.py                  # ORM-модели (SensorConfig, CriticalZoneRule, TrainingHistory)
│   └── repos/
│       ├── sensor_config.py       # CRUD для sensor_configs
│       ├── critical_zone.py       # CRUD для critical_zone_rules
│       └── training_history.py    # CRUD для training_history
│
├── core/
│   ├── __init__.py
│   ├── aggregator.py              # CandleAggregator: 1s → timeframe свечи
│   ├── buffer.py                  # TimeframeBuffer: накопление агрег. свечей
│   ├── model_factory.py           # ModelFactory: создание модели по model_type
│   ├── model_store.py             # ModelStore: LRU-кеш + диск
│   ├── predictor.py               # PredictorEngine: predict + critical zone check
│   ├── registry.py                # SensorRegistry: главный in-memory реестр
│   ├── signal_aggregator.py       # SignalAggregator: OR по таймфреймам
│   └── notifier.py                # CoreAPINotifier: POST уведомлений
│
├── tasks/
│   ├── __init__.py
│   ├── app.py                     # Celery app instance
│   ├── training.py                # train_model task
│   └── notify.py                  # send_signal task
│
├── routes/
│   ├── __init__.py
│   ├── candle.py                  # POST /candle/ingest, WS /candle/ws
│   ├── sensor.py                  # CRUD /sensor/{id}/config, reset, delete
│   ├── critical_zone.py           # PUT/GET /sensor/{id}/critical-zone
│   ├── status.py                  # GET /sensor/{id}/status, /prediction, /signal
│   └── health.py                  # GET /health, /ping
│
├── model/                         # (существует, минимальные изменения)
│   ├── base.py
│   ├── scikit_compat/
│   └── torch_compat/
│
├── metrics/                       # (существует, без изменений)
│   ├── eval.py
│   └── metrics.py
│
└── dataset/                       # (существует, без изменений)
    ├── base.py
    └── time_series_dataset.py
```

---

## 11. Переменные окружения

| Переменная | Default | Описание |
|---|---|---|
| `HOST_ADDRESS` | `0.0.0.0` | Адрес FastAPI сервера |
| `HOST_PORT` | `10000` | Порт FastAPI сервера |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis broker для Celery + буферы |
| `REDIS_RESULT_URL` | `redis://localhost:6379/1` | Redis backend для результатов Celery |
| `DATABASE_URL` | `sqlite+aiosqlite:///./hypal.db` | URL базы данных |
| `MODELS_DIR` | `models/` | Директория для сохранения моделей |
| `MAX_PARALLEL_TRAINING` | `2` | Макс. параллельных Celery-задач обучения |
| `MAX_LOADED_MODELS` | `10` | Макс. моделей одновременно в памяти |
| `MODEL_IDLE_TIMEOUT_S` | `3600` | Время простоя (сек) до выгрузки из памяти |
| `DEFAULT_MODEL_TYPE` | `linear` | Модель по умолчанию (linear/catboost/transformer) |
| `LOG_LEVEL` | `INFO` | Уровень логирования |

---

## 12. Поэтапный план реализации

### Фаза 1 — Инфраструктура (основа)

1. **`config.py`** — `Settings` класс через `pydantic-settings` (все env-vars)
2. **`db/engine.py`** — async SQLAlchemy + alembic миграции
3. **`db/models.py`** — ORM-модели: `SensorConfig`, `CriticalZoneRule`, `TrainingHistory`
4. **`db/repos/`** — репозитории (CRUD)
5. **`schemas/critical_zone.py`** — JSON DSL: `CriticalZoneRuleDTO` + `parse_rule()` / `serialize_rule()`
6. **`tasks/app.py`** — инициализация Celery

### Фаза 2 — Ядро pipeline

7. **`core/aggregator.py`** — `CandleAggregator`: принимает сырые свечи, агрегирует по таймфреймам
8. **`core/buffer.py`** — `TimeframeBuffer`: хранит агрегированные свечи, знает своё состояние (WAITING/GATHERING/TRAINING/READY)
9. **`core/model_factory.py`** — `ModelFactory.create(model_type, input_horizon, output_horizon)`
10. **`core/model_store.py`** — `ModelStore`: LRU-кеш + сериализация на диск
11. **`core/registry.py`** — `SensorRegistry`: главный реестр сенсоров, оркестрирует буферы и состояния
12. **`tasks/training.py`** — реальный `train_model` Celery-task

### Фаза 3 — Предсказание и сигналы

13. **`core/predictor.py`** — `PredictorEngine`: `predict()` + проверка критической зоны
14. **`core/signal_aggregator.py`** — `SignalAggregator`: OR по таймфреймам, детекция изменения
15. **`core/notifier.py`** — `CoreAPINotifier`: async POST в CoreAPI + retry
16. **`tasks/notify.py`** — Celery-task для уведомлений (резервный путь)

### Фаза 4 — API

17. **`schemas/`** — все оставшиеся Pydantic-схемы (candle, config, prediction, status)
18. **`routes/candle.py`** — реальный `POST /candle/ingest` и WS
19. **`routes/sensor.py`** — конфигурация сенсора
20. **`routes/critical_zone.py`** — управление критической зоной
21. **`routes/status.py`** — статус, предсказания, сигнал
22. **`routes/health.py`** — health check

### Фаза 5 — Интеграция и полировка

23. **`main.py`** — обновить `lifespan`: инициализация БД, Redis, ModelStore, SensorRegistry
24. **`Dockerfile`** — добавить запуск Celery worker как отдельный сервис
25. **`docker-compose.yml`** — FastAPI + Celery Worker + Redis
26. **Тесты** — unit-тесты для aggregator, buffer, signal_aggregator; integration-тест для полного pipeline

---

## 13. Решения по спорным моментам

| Вопрос | Решение |
|---|---|
| Где хранить агрегированные буферы? | В Redis (List) — переживает рестарт сервиса |
| Как сериализовать модели? | `joblib` для sklearn/catboost, `torch.save` для Transformer |
| Как передавать данные в Celery task? | JSON-сериализация свечей (не pickle) |
| Агрегация сигналов: AND или OR? | OR — любой таймфрейм сигнализирует → тревога |
| Что если CoreAPI недоступен? | Retry через Celery task с экспоненциальным back-off |
| Переобучение READY-модели? | Каждые `num_train_samples` новых свечей (скользящее окно) |
| Отрицательные значения (температура)? | Все модели работают с сырыми float — нет ограничений, нормализация Z-score |