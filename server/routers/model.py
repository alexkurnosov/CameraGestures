"""GET    /model/download      — serve latest .tflite binary.
   GET    /model/preprocessor — serve preprocessor.js.
   GET    /model/info         — model metadata.
   GET    /model/metrics      — detailed metrics for the latest model (public).
   GET    /model/metrics/list — list trained models with summary metrics (public).
   GET    /model/metrics/{id} — detailed metrics for a specific model (public).
   DELETE /model              — wipe all model versions."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from auth import get_current_device

from models import (
    ModelInfoResponse,
    ModelMetricsListResponse,
    ModelMetricsResponse,
    ModelMetricsSummary,
)
from storage.model_store import (
    delete_all_models,
    get_latest_model,
    get_model_by_id,
    list_models,
)
from training_state import training_state

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/download")
async def download_model(_: str = Depends(get_current_device)) -> FileResponse:
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


@router.get("/preprocessor")
async def download_preprocessor(_: str = Depends(get_current_device)) -> FileResponse:
    """Return preprocessor.js — the shared feature-extraction source of truth."""
    js_path = Path(__file__).parent.parent / "ml" / "preprocessor.js"
    if not js_path.exists():
        raise HTTPException(status_code=404, detail="preprocessor.js not found on server.")
    return FileResponse(
        path=str(js_path),
        media_type="application/javascript",
        filename="preprocessor.js",
    )


@router.get("/info", response_model=ModelInfoResponse)
async def model_info(_: str = Depends(get_current_device)) -> ModelInfoResponse:
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


def _model_row_to_metrics_response(model: dict) -> ModelMetricsResponse:
    """Map a storage row to the public metrics payload.

    Models trained before the extended-metrics change only have accuracy /
    f1_weighted / confusion_matrix — the richer fields default to empty.
    """
    metrics = model.get("metrics") or {}
    return ModelMetricsResponse(
        model_id=model["id"],
        trainer=model["trainer"],
        trained_at=model["trained_at"],
        trained_on=model["trained_on"],
        gesture_ids=model["gesture_ids"],
        balance_strategy=metrics.get("balance_strategy"),
        accuracy=metrics.get("accuracy"),
        f1_weighted=metrics.get("f1_weighted"),
        confusion_matrix=metrics.get("confusion_matrix"),
        val_size=metrics.get("val_size"),
        train_size=metrics.get("train_size"),
        per_class=metrics.get("per_class") or [],
        none_aware=metrics.get("none_aware") or {},
        confidence_by_class=metrics.get("confidence_by_class") or [],
        threshold_curves=metrics.get("threshold_curves") or [],
        auc=metrics.get("auc") or {},
    )


# --- Public (unauthenticated) metrics endpoints ---
# These are intentionally not gated by get_current_device so the iOS
# interpretation page can fetch them without registering a device token.

@router.get("/metrics", response_model=ModelMetricsResponse)
async def latest_model_metrics() -> ModelMetricsResponse:
    """Detailed metrics for the most recently trained model."""
    model = await get_latest_model()
    if model is None:
        raise HTTPException(status_code=404, detail="No model has been trained yet.")
    return _model_row_to_metrics_response(model)


@router.get("/metrics/list", response_model=ModelMetricsListResponse)
async def list_model_metrics() -> ModelMetricsListResponse:
    """Summary row per trained model version, newest first."""
    rows = await list_models()
    summaries = [
        ModelMetricsSummary(
            model_id=row["id"],
            trained_at=row["trained_at"],
            trained_on=row["trained_on"],
            accuracy=(row.get("metrics") or {}).get("accuracy"),
            f1_weighted=(row.get("metrics") or {}).get("f1_weighted"),
        )
        for row in rows
    ]
    return ModelMetricsListResponse(models=summaries)


@router.get("/metrics/{model_id}", response_model=ModelMetricsResponse)
async def model_metrics_by_id(model_id: str) -> ModelMetricsResponse:
    """Detailed metrics for a specific trained model."""
    model = await get_model_by_id(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found.")
    return _model_row_to_metrics_response(model)


@router.delete("", status_code=200)
async def wipe_models(_: str = Depends(get_current_device)) -> dict:
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
