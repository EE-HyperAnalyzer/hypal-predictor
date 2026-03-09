from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import joblib
import torch

from hypal_predictor.config import settings
from hypal_predictor.model.base import BaseModel as PredictorBaseModel
from hypal_predictor.model.torch_compat.base import TorchCompatibleModel

logger = logging.getLogger(__name__)


class ModelStore:
    """
    LRU-кеш обученных моделей в памяти с персистентностью на диск.

    Ключ кеша: (source, sensor, axis, timeframe_str).
    При переполнении кеша (> max_loaded) вытесняет наименее недавно
    использованную модель — она остаётся на диске и может быть перезагружена.

    Стратегия сериализации:
    - TorchCompatibleModel → torch.save / torch.load
    - Всё остальное       → joblib.dump / joblib.load
    """

    def __init__(
        self,
        models_dir: str | None = None,
        max_loaded: int | None = None,
    ):
        self._models_dir = Path(models_dir or settings.models_dir)
        self._max_loaded = max_loaded or settings.max_loaded_models
        self._cache: OrderedDict[tuple[str, str, str, str], PredictorBaseModel] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        source: str,
        sensor: str,
        axis: str,
        timeframe: str,
        model: PredictorBaseModel,
    ) -> Path:
        """
        Сохраняет модель на диск и добавляет её в LRU-кеш.

        Returns:
            Путь к сохранённому файлу.
        """
        path = self._model_path(source, sensor, axis, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(model, TorchCompatibleModel):
            torch.save(model, path)
        else:
            joblib.dump(model, path)

        logger.info("Model saved: %s", path)
        self._put_cache(source, sensor, axis, timeframe, model)
        return path

    def load(
        self,
        source: str,
        sensor: str,
        axis: str,
        timeframe: str,
    ) -> PredictorBaseModel | None:
        """
        Возвращает модель из кеша (если есть) или загружает с диска.
        Возвращает None, если файл на диске не найден.
        """
        key = self._cache_key(source, sensor, axis, timeframe)

        if key in self._cache:
            self._cache.move_to_end(key)
            logger.debug("Model cache hit: %s", key)
            return self._cache[key]

        path = self._model_path(source, sensor, axis, timeframe)
        if not path.exists():
            logger.debug("Model not found on disk: %s", path)
            return None

        model = self._load_from_disk(path)
        if model is None:
            return None

        logger.info("Model loaded from disk: %s", path)
        self._put_cache(source, sensor, axis, timeframe, model)
        return model

    def evict(self, source: str, sensor: str, axis: str, timeframe: str) -> None:
        """
        Удаляет модель из in-memory кеша.
        Файл на диске остаётся нетронутым.
        """
        key = self._cache_key(source, sensor, axis, timeframe)
        removed = self._cache.pop(key, None)
        if removed is not None:
            logger.debug("Evicted from cache: %s", key)

    def delete(self, source: str, sensor: str, axis: str, timeframe: str) -> None:
        """
        Удаляет модель из кеша и с диска.
        """
        self.evict(source, sensor, axis, timeframe)
        path = self._model_path(source, sensor, axis, timeframe)
        if path.exists():
            path.unlink()
            logger.info("Model deleted from disk: %s", path)

    def is_cached(self, source: str, sensor: str, axis: str, timeframe: str) -> bool:
        """Проверяет наличие модели в in-memory кеше."""
        return self._cache_key(source, sensor, axis, timeframe) in self._cache

    def exists_on_disk(self, source: str, sensor: str, axis: str, timeframe: str) -> bool:
        """Проверяет наличие файла модели на диске."""
        return self._model_path(source, sensor, axis, timeframe).exists()

    def model_path(self, source: str, sensor: str, axis: str, timeframe: str) -> Path:
        """Возвращает ожидаемый путь к файлу модели (файл может не существовать)."""
        return self._model_path(source, sensor, axis, timeframe)

    @property
    def cached_keys(self) -> list[tuple[str, str, str, str]]:
        """Список ключей моделей, находящихся в памяти (от новейшей к старейшей)."""
        return list(reversed(self._cache.keys()))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _model_path(self, source: str, sensor: str, axis: str, timeframe: str) -> Path:
        """
        Структура каталогов: <models_dir>/<source>/<sensor>/<axis>/<tf_safe>.pkl
        Двоеточие из строки таймфрейма убирается, чтобы не ломать пути на Windows.
        """
        tf_safe = timeframe.replace(":", "")
        return self._models_dir / source / sensor / axis / f"{tf_safe}.pkl"

    @staticmethod
    def _cache_key(source: str, sensor: str, axis: str, timeframe: str) -> tuple[str, str, str, str]:
        return (source, sensor, axis, timeframe)

    def _put_cache(
        self,
        source: str,
        sensor: str,
        axis: str,
        timeframe: str,
        model: PredictorBaseModel,
    ) -> None:
        """
        Помещает модель в кеш и применяет LRU-вытеснение при переполнении.
        """
        key = self._cache_key(source, sensor, axis, timeframe)
        self._cache[key] = model
        self._cache.move_to_end(key)

        while len(self._cache) > self._max_loaded:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.debug("LRU eviction from cache: %s", evicted_key)

    def _load_from_disk(self, path: Path) -> PredictorBaseModel | None:
        """
        Пытается загрузить модель с диска.
        Сначала пробует joblib (scikit-learn-совместимые модели),
        затем torch.load (PyTorch-модели).
        """
        try:
            model = joblib.load(path)
            return model
        except Exception:
            logger.debug("joblib.load failed for %s, trying torch.load", path)

        try:
            model = torch.load(path, weights_only=False)
            return model
        except Exception:
            logger.exception("Failed to load model from %s", path)

        return None


# ---------------------------------------------------------------------------
# Module-level singleton — инициализируется при первом импорте.
# Весь код в приложении должен использовать именно этот объект.
# ---------------------------------------------------------------------------
model_store = ModelStore()
