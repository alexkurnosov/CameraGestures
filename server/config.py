from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Data storage (on-disk files: trained models, update trigger)
    data_dir: Path = Path("data")

    # PostgreSQL
    postgres_host: str = "db"
    postgres_port: int = 5432
    postgres_db: str = "gestures"
    postgres_user: str = "gestures"
    postgres_password: str

    # Training
    auto_train_threshold: int = 10
    trainer: str = "rf_mlp"  # "rf_mlp" | "lstm"

    # Model versioning
    max_model_versions: int = 5

    # Authentication — both fields are required (no default); server refuses to start if absent
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expire_days: int = 365
    registration_token: str

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    def ensure_dirs(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
