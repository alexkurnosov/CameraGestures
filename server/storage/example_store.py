"""Persist and query TrainingExample records in SQLite."""

import json
import time
import uuid

import sqlalchemy as sa

from database import examples_table, get_engine
from models import TrainingExamplePayload


async def save_example(payload: TrainingExamplePayload) -> str:
    """Insert or update one training example. Returns its UUID.

    When ``payload.id`` is supplied and already exists the row is updated
    (upsert); otherwise a new row is created.
    """
    example_id = payload.id or str(uuid.uuid4())
    engine = get_engine()
    async with engine.begin() as conn:
        # Check if this id already exists
        if payload.id:
            existing = await conn.execute(
                sa.select(examples_table.c.id).where(
                    examples_table.c.id == example_id
                )
            )
            if existing.first() is not None:
                await conn.execute(
                    examples_table.update()
                    .where(examples_table.c.id == example_id)
                    .values(
                        gesture_id=payload.gesture_id,
                        session_id=payload.session_id,
                        user_id=payload.user_id,
                        hand_film_json=payload.hand_film.model_dump_json(),
                    )
                )
                return example_id

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


async def get_examples_by_gesture(gesture_id: str) -> list[dict]:
    """Return all examples for a given gesture slug, ordered by created_at."""
    engine = get_engine()
    async with engine.connect() as conn:
        result = await conn.execute(
            sa.select(examples_table)
            .where(examples_table.c.gesture_id == gesture_id)
            .order_by(examples_table.c.created_at)
        )
        rows = result.mappings().all()

    return [
        {
            "id": row["id"],
            "gesture_id": row["gesture_id"],
            "session_id": row["session_id"],
            "user_id": row["user_id"],
            "hand_film": json.loads(row["hand_film_json"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


async def delete_example_by_id(example_id: str) -> bool:
    """Delete a single example by UUID. Returns True if a row was deleted."""
    engine = get_engine()
    async with engine.begin() as conn:
        result = await conn.execute(
            examples_table.delete().where(examples_table.c.id == example_id)
        )
    return result.rowcount > 0


async def update_example_gesture(example_id: str, new_gesture_id: str) -> bool:
    """Update the gesture_id of a single example. Returns True if a row was updated."""
    engine = get_engine()
    async with engine.begin() as conn:
        result = await conn.execute(
            examples_table.update()
            .where(examples_table.c.id == example_id)
            .values(gesture_id=new_gesture_id)
        )
    return result.rowcount > 0


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
