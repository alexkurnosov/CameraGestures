"""GET    /model/download — serve latest .tflite binary.
   GET    /model/info     — model metadata.
   DELETE /model          — wipe all model versions."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from models import ModelInfoResponse
from storage.model_store import delete_all_models, get_latest_model
from training_state import training_state

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/download")
async def download_model() -> FileResponse:
    """
    Return the latest trained .tflite file.
    Sets Content-Disposition so the client saves it as 'gesture_model.tflite'.
    Returns 404 if no model has been trained yet.
    """
    model = await get_latest_model()
    if model is None:
        raise HTTPException(status_code=404, detail="No model has been trained yet.")

    tflite_path = Path(model["tflite_path"])
    if not tflite_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Model file not found on disk. It may have been pruned.",
        )

    return FileResponse(
        path=str(tflite_path),
        media_type="application/octet-stream",
        filename="gesture_model.tflite",
        headers={"Content-Disposition": 'attachment; filename="gesture_model.tflite"'},
    )


@router.get("/info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """Return metadata about the most recently trained model."""
    model = await get_latest_model()
    if model is None:
        raise HTTPException(status_code=404, detail="No model has been trained yet.")

    metrics = model.get("metrics") or {}
    return ModelInfoResponse(
        model_id=model["id"],
        trainer=model["trainer"],
        trained_on=model["trained_on"],
        trained_at=model["trained_at"],
        gesture_ids=model["gesture_ids"],
        accuracy=metrics.get("accuracy"),
        f1=metrics.get("f1_weighted"),
        confusion_matrix=metrics.get("confusion_matrix"),
        min_in_view_duration=model.get("min_in_view_duration"),
    )


@router.delete("", status_code=200)
async def wipe_models() -> dict:
    """Delete all trained model versions (DB rows + .tflite files) and reset training state."""
    if training_state.is_running():
        raise HTTPException(
            status_code=409, detail="A training job is currently running. Wait for it to finish first."
        )
    deleted = await delete_all_models()
    async with training_state._lock:
        training_state.status = "idle"
        training_state.job_id = None
        training_state.accuracy = None
        training_state.trained_on = 0
        training_state.gesture_ids = []
        training_state.trained_at = None
        training_state.error = None
        training_state.min_in_view_duration = 1.2
    return {"deleted_models": deleted}
