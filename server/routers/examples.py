"""POST   /examples           — upload one training example.
   GET    /examples/stats     — per-gesture counts.
   DELETE /examples           — wipe all examples (or one gesture's worth)."""

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from auth import get_current_device
from models import (
    ExampleStatsResponse,
    GestureStats,
    TrainingExamplePayload,
    UploadExampleResponse,
)
from storage.example_store import (
    count_by_gesture,
    count_all_by_gesture,
    delete_all_examples,
    save_example,
)

router = APIRouter(prefix="/examples", tags=["examples"])


@router.post("", response_model=UploadExampleResponse, status_code=201)
async def upload_example(
    payload: TrainingExamplePayload,
    background_tasks: BackgroundTasks,
    _: str = Depends(get_current_device),
) -> UploadExampleResponse:
    """
    Store one labelled HandFilm.
    After saving, checks whether the auto-train threshold is met and, if so,
    schedules a background training job (imported lazily to avoid circular deps).
    """
    if not payload.hand_film.frames:
        raise HTTPException(status_code=422, detail="hand_film must contain at least one frame.")

    example_id = await save_example(payload)
    total = await count_by_gesture(payload.gesture_id)

    # Trigger auto-train check without blocking the response
    background_tasks.add_task(_maybe_auto_train, payload.gesture_id)

    return UploadExampleResponse(id=example_id, total_for_gesture=total)


@router.get("/stats", response_model=ExampleStatsResponse)
async def example_stats(_: str = Depends(get_current_device)) -> ExampleStatsResponse:
    counts = await count_all_by_gesture()
    gestures = [GestureStats(gesture_id=gid, count=cnt) for gid, cnt in counts.items()]
    return ExampleStatsResponse(gestures=gestures, total=sum(counts.values()))


@router.delete("", status_code=200)
async def wipe_examples(
    gesture_id: Optional[str] = Query(
        default=None,
        description="Delete only examples for this gesture slug. Omit to delete everything.",
    ),
    _: str = Depends(get_current_device),
) -> dict:
    """
    Delete training examples.
    - `DELETE /examples`                    → wipe all examples
    - `DELETE /examples?gesture_id=wave`    → wipe only the 'wave' gesture
    """
    deleted = await delete_all_examples(gesture_id=gesture_id)
    return {"deleted_examples": deleted, "gesture_id": gesture_id}


# --- Helpers ---

async def _maybe_auto_train(gesture_id: str) -> None:
    """
    Import training state lazily to break the circular dependency between
    routers/examples ↔ routers/training.
    """
    from routers.training import maybe_auto_train
    await maybe_auto_train()
