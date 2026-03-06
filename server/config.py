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

    # Data storage
    data_dir: Path = Path("data")

    # Training
    auto_train_threshold: int = 10
    trainer: str = "rf_mlp"  # "rf_mlp" | "lstm"

    # Model versioning
    max_model_versions: int = 5

    @property
    def examples_dir(self) -> Path:
        return self.data_dir / "examples"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "gestures.db"

    def ensure_dirs(self) -> None:
        self.examples_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
