"""Pydantic models — mirrors HandGestureTypes Swift structs exactly."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Point3D(BaseModel):
    x: float
    y: float
    z: float


class HandShot(BaseModel):
    landmarks: list[Point3D]  # always 21
    timestamp: float
    left_or_right: str  # "left" | "right" | "unknown"


class HandFilm(BaseModel):
    frames: list[HandShot]
    start_time: float


class TrainingExamplePayload(BaseModel):
    hand_film: HandFilm
    gesture_id: str  # slug, e.g. "thumbs_up"
    session_id: str
    user_id: str | None = None


# --- Response schemas ---

class UploadExampleResponse(BaseModel):
    id: str
    total_for_gesture: int


class GestureStats(BaseModel):
    gesture_id: str
    count: int


class ExampleStatsResponse(BaseModel):
    gestures: list[GestureStats]
    total: int


class TrainingJobResponse(BaseModel):
    job_id: str
    status: str  # "started" | "already_running"


class ModelStatusResponse(BaseModel):
    status: str  # "idle" | "training" | "ready" | "failed"
    accuracy: float | None = None
    trained_on: int = 0
    gesture_ids: list[str] = Field(default_factory=list)
    trained_at: float | None = None
    error: str | None = None


class ModelInfoResponse(BaseModel):
    model_id: str
    trainer: str
    trained_on: int
    trained_at: float
    gesture_ids: list[str]
    accuracy: float | None = None
    f1: float | None = None
    confusion_matrix: list[list[int]] | None = None
