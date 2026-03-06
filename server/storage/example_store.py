"""Persist and query TrainingExample records in SQLite."""

import json
import time
import uuid

import sqlalchemy as sa

from database import examples_table, get_engine
from models import TrainingExamplePayload


async def save_example(payload: TrainingExamplePayload) -> str:
    """Insert one training example. Returns its generated UUID."""
    example_id = str(uuid.uuid4())
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            examples_table.insert().values(
                id=example_id,
                gesture_id=payload.gesture_id,
                session_id=payload.session_id,
                user_id=payload.user_id,
                hand_film_json=payload.hand_film.model_dump_json(),
                created_at=time.time(),
            )
        )
    return example_id


async def count_by_gesture(gesture_id: str) -> int:
    """Return the number of stored examples for a given gesture slug."""
    engine = get_engine()
    async with engine.connect() as conn:
        result = await conn.execute(
            sa.select(sa.func.count())
            .select_from(examples_table)
            .where(examples_table.c.gesture_id == gesture_id)
        )
        return result.scalar_one()


async def count_all_by_gesture() -> dict[str, int]:
    """Return {gesture_id: count} for every gesture that has at least one example."""
    engine = get_engine()
    async with engine.connect() as conn:
        result = await conn.execute(
            sa.select(
                examples_table.c.gesture_id,
                sa.func.count().label("cnt"),
            ).group_by(examples_table.c.gesture_id)
        )
        return {row.gesture_id: row.cnt for row in result}


async def delete_all_examples(gesture_id: str | None = None) -> int:
    """Delete examples. Pass gesture_id to wipe only one gesture, or None for all.
    Returns the number of rows deleted."""
    engine = get_engine()
    async with engine.begin() as conn:
        stmt = examples_table.delete()
        if gesture_id is not None:
            stmt = stmt.where(examples_table.c.gesture_id == gesture_id)
        result = await conn.execute(stmt)
    return result.rowcount


async def load_all_examples() -> list[dict]:
    """
    Load all examples as dicts with keys:
      id, gesture_id, session_id, user_id, hand_film (parsed dict), created_at
    """
    engine = get_engine()
    async with engine.connect() as conn:
        result = await conn.execute(
            sa.select(examples_table).order_by(examples_table.c.created_at)
        )
        rows = result.mappings().all()

    examples = []
    for row in rows:
        examples.append(
            {
                "id": row["id"],
                "gesture_id": row["gesture_id"],
                "session_id": row["session_id"],
                "user_id": row["user_id"],
                "hand_film": json.loads(row["hand_film_json"]),
                "created_at": row["created_at"],
            }
        )
    return examples
