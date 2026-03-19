"""Manage trained model versions in SQLite + .tflite files on disk."""

import json
import time
import uuid
from pathlib import Path

import sqlalchemy as sa

from config import settings
from database import get_engine, models_table


async def save_model(
    tflite_path: Path,
    gesture_ids: list[str],
    trainer: str,
    trained_on: int,
    metrics: dict | None = None,
    min_in_view_duration: float | None = None,
) -> str:
    """
    Register a new model version. Prunes old versions to keep at most
    settings.max_model_versions entries (deleting both DB rows and .tflite files).
    Returns the new model UUID.
    """
    model_id = str(uuid.uuid4())
    engine = get_engine()

    async with engine.begin() as conn:
        await conn.execute(
            models_table.insert().values(
                id=model_id,
                tflite_path=str(tflite_path),
                gesture_ids_json=json.dumps(gesture_ids),
                metrics_json=json.dumps(metrics) if metrics else None,
                trainer=trainer,
                trained_on=trained_on,
                trained_at=time.time(),
                min_in_view_duration=min_in_view_duration,
            )
        )
        await _prune(conn)

    return model_id


async def get_latest_model() -> dict | None:
    """Return the most recently trained model row as a dict, or None."""
    engine = get_engine()
    async with engine.connect() as conn:
        result = await conn.execute(
            sa.select(models_table)
            .order_by(models_table.c.trained_at.desc())
            .limit(1)
        )
        row = result.mappings().first()
    if row is None:
        return None
    return _row_to_dict(row)


async def get_model_by_id(model_id: str) -> dict | None:
    engine = get_engine()
    async with engine.connect() as conn:
        result = await conn.execute(
            sa.select(models_table).where(models_table.c.id == model_id)
        )
        row = result.mappings().first()
    if row is None:
        return None
    return _row_to_dict(row)


async def list_models() -> list[dict]:
    engine = get_engine()
    async with engine.connect() as conn:
        result = await conn.execute(
            sa.select(models_table).order_by(models_table.c.trained_at.desc())
        )
        return [_row_to_dict(row) for row in result.mappings()]


async def delete_all_models() -> int:
    """Delete every model version — both DB rows and .tflite files on disk.
    Returns the number of models deleted."""
    engine = get_engine()
    async with engine.begin() as conn:
        result = await conn.execute(
            sa.select(models_table.c.id, models_table.c.tflite_path)
        )
        rows = result.all()
        for row in rows:
            Path(row.tflite_path).unlink(missing_ok=True)
        await conn.execute(models_table.delete())
    return len(rows)


# --- Helpers ---

def _row_to_dict(row) -> dict:
    return {
        "id": row["id"],
        "tflite_path": row["tflite_path"],
        "gesture_ids": json.loads(row["gesture_ids_json"]),
        "metrics": json.loads(row["metrics_json"]) if row["metrics_json"] else None,
        "trainer": row["trainer"],
        "trained_on": row["trained_on"],
        "trained_at": row["trained_at"],
        "min_in_view_duration": row["min_in_view_duration"],
    }


async def _prune(conn) -> None:
    """Delete oldest model rows (and their .tflite files) beyond the version cap."""
    result = await conn.execute(
        sa.select(models_table.c.id, models_table.c.tflite_path)
        .order_by(models_table.c.trained_at.desc())
    )
    rows = result.all()

    to_delete = rows[settings.max_model_versions :]
    for row in to_delete:
        path = Path(row.tflite_path)
        if path.exists():
            path.unlink(missing_ok=True)
        await conn.execute(
            models_table.delete().where(models_table.c.id == row.id)
        )
