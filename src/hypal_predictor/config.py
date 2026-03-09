from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    host_address: str = "0.0.0.0"
    host_port: int = 10000

    redis_url: str = "redis://localhost:6379/0"
    redis_result_url: str = "redis://localhost:6379/1"

    database_url: str = "sqlite+aiosqlite:///./hypal.db"

    models_dir: str = "models"

    max_parallel_training: int = 2
    max_loaded_models: int = 10
    model_idle_timeout_s: int = 3600

    default_model_type: str = "linear"

    log_level: str = "INFO"

    core_api_timeout_s: int = 10


settings = Settings()
