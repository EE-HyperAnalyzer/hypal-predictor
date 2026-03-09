from hypal_predictor.model.base import BaseModel as PredictorBaseModel
from hypal_predictor.model.scikit_compat.candle.catbreg import SLCC_CatBoostRegressorModel
from hypal_predictor.model.scikit_compat.candle.linreg import SLCC_LinearRegressionModel
from hypal_predictor.model.torch_compat.candle.transformer import TCC_Transformer

_MODEL_REGISTRY: dict[str, type] = {
    "linear": SLCC_LinearRegressionModel,
    "catboost": SLCC_CatBoostRegressorModel,
    "transformer": TCC_Transformer,
}


def create_model(
    model_type: str,
    input_horizon: int,
    output_horizon: int,
) -> PredictorBaseModel:
    """
    Создаёт и возвращает экземпляр модели по строковому идентификатору.

    Args:
        model_type: Идентификатор модели. Допустимые значения: "linear", "catboost", "transformer".
        input_horizon: Длина входного окна (количество свечей на вход).
        output_horizon: Длина выходного горизонта (количество свечей на выход).

    Returns:
        Новый (необученный) экземпляр соответствующей модели.

    Raises:
        ValueError: Если model_type не зарегистрирован.
    """
    cls = _MODEL_REGISTRY.get(model_type)
    if cls is None:
        raise ValueError(f"Unknown model_type={model_type!r}. Available: {list(_MODEL_REGISTRY)}")
    return cls(input_horizon=input_horizon, output_horizon=output_horizon)


def available_model_types() -> list[str]:
    """Возвращает список всех зарегистрированных типов моделей."""
    return list(_MODEL_REGISTRY.keys())
