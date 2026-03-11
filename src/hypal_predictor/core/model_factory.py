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
    train_size: float = 0.8,
) -> PredictorBaseModel:
    cls = _MODEL_REGISTRY.get(model_type)
    if cls is None:
        raise ValueError(f"Unknown model_type={model_type!r}. Available: {list(_MODEL_REGISTRY)}")
    return cls(
        input_horizon=input_horizon,
        output_horizon=output_horizon,
        train_size=train_size,
    )


def available_model_types() -> list[str]:
    return list(_MODEL_REGISTRY.keys())
