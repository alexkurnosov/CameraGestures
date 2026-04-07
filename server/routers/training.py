"""POST /train        — trigger training job.
   GET  /model/status — poll training state."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from fastapi import APIRouter, Depends

from auth import get_current_device

from config import settings
from models import ModelStatusResponse, TrainingJobResponse, TriggerTrainingRequest
from storage.example_store import count_all_by_gesture, load_all_examples
from storage.model_store import save_model
from training_state import training_state

router = APIRouter(tags=["training"])


@router.post("/train", response_model=TrainingJobResponse)
async def trigger_training(
    body: TriggerTrainingRequest = TriggerTrainingRequest(),
    _: str = Depends(get_current_device),
) -> TrainingJobResponse:
    """Kick off a training job in the background (regardless of threshold)."""
    async with training_state._lock:
        if training_state.is_running():
            return TrainingJobResponse(
                job_id=training_state.job_id or "", status="already_running"
            )
        job_id = str(uuid.uuid4())
        training_state.job_id = job_id
        training_state.status = "training"
        training_state.min_in_view_duration = body.min_in_view_duration
        training_state.error = None

    asyncio.create_task(_run_training(job_id))
    return TrainingJobResponse(job_id=job_id, status="started")


@router.get("/model/status", response_model=ModelStatusResponse)
async def model_status(_: str = Depends(get_current_device)) -> ModelStatusResponse:
    return ModelStatusResponse(
        status=training_state.status,
        accuracy=training_state.accuracy,
        trained_on=training_state.trained_on,
        gesture_ids=training_state.gesture_ids,
        trained_at=training_state.trained_at,
        error=training_state.error,
    )


# --- Auto-train helper (called from examples router) ---

async def maybe_auto_train() -> None:
    """
    Trigger training if every known gesture has at least AUTO_TRAIN_THRESHOLD
    examples and no job is currently running.
    """
    async with training_state._lock:
        if training_state.is_running():
            return

        counts = await count_all_by_gesture()
        if not counts:
            return

        threshold = settings.auto_train_threshold
        if all(v >= threshold for v in counts.values()):
            job_id = str(uuid.uuid4())
            training_state.job_id = job_id
            training_state.status = "training"
            training_state.error = None
            asyncio.create_task(_run_training(job_id))


# --- Background training task ---

async def _run_training(job_id: str) -> None:
    """
    Runs the selected trainer in a thread pool executor so it doesn't block
    the event loop during heavy Keras/TF computation.
    """
    loop = asyncio.get_running_loop()
    # Snapshot the threshold before entering the thread so we don't need the lock there.
    async with training_state._lock:
        min_in_view = training_state.min_in_view_duration
    try:
        result: dict[str, Any] = await loop.run_in_executor(None, _train_sync)
        async with training_state._lock:
            if training_state.job_id == job_id:
                training_state.status = "ready"
                training_state.accuracy = (result["metrics"] or {}).get("accuracy")
                training_state.trained_on = result["trained_on"]
                training_state.gesture_ids = result["gesture_ids"]
                training_state.trained_at = result.get("trained_at")
                training_state.error = None
        # Persist min_in_view_duration alongside the model record.
        result["min_in_view_duration"] = min_in_view
    except Exception as exc:
        async with training_state._lock:
            if training_state.job_id == job_id:
                training_state.status = "failed"
                training_state.error = str(exc)


def _train_sync() -> dict[str, Any]:
    """Synchronous training call — runs in a thread pool."""
    import time
    from storage.example_store import load_all_examples as _load  # sync wrapper
    import asyncio as _asyncio

    # We're in a thread, so we need a fresh event loop to run the async load
    examples = _asyncio.run(_load())

    if settings.trainer == "lstm":
        from ml.trainer_lstm import train
    else:
        from ml.trainer_rf_mlp import train

    result = train(examples)

    # Persist model metadata to DB (also needs its own loop)
    import asyncio as _asyncio2
    model_id = _asyncio2.run(
        save_model(
            tflite_path=result["tflite_path"],
            gesture_ids=result["gesture_ids"],
            trainer=settings.trainer,
            trained_on=result["trained_on"],
            metrics=result["metrics"],
            min_in_view_duration=result.get("min_in_view_duration"),
        )
    )
    result["trained_at"] = __import__("time").time()
    return result
