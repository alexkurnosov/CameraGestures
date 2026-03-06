"""SQLAlchemy Core database setup and table definitions."""

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from config import settings

_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        db_url = f"sqlite+aiosqlite:///{settings.db_path}"
        _engine = create_async_engine(db_url, echo=False)
    return _engine


metadata = sa.MetaData()

# Stores uploaded training examples.
# hand_film_json: full HandFilm serialised as JSON text
examples_table = sa.Table(
    "examples",
    metadata,
    sa.Column("id", sa.String, primary_key=True),
    sa.Column("gesture_id", sa.String, nullable=False, index=True),
    sa.Column("session_id", sa.String, nullable=False),
    sa.Column("user_id", sa.String, nullable=True),
    sa.Column("hand_film_json", sa.Text, nullable=False),
    sa.Column("created_at", sa.Float, nullable=False),
)

# Stores metadata for each trained model version.
models_table = sa.Table(
    "models",
    metadata,
    sa.Column("id", sa.String, primary_key=True),
    sa.Column("tflite_path", sa.String, nullable=False),
    sa.Column("gesture_ids_json", sa.Text, nullable=False),   # JSON list[str]
    sa.Column("metrics_json", sa.Text, nullable=True),        # JSON ModelMetrics
    sa.Column("trainer", sa.String, nullable=False),
    sa.Column("trained_on", sa.Integer, nullable=False),
    sa.Column("trained_at", sa.Float, nullable=False),
)


async def init_db() -> None:
    """Create tables if they do not exist."""
    settings.ensure_dirs()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
