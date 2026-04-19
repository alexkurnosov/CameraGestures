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
    is_absent: bool = False  # True when no hand was detected (zero landmarks placeholder)


class HandFilm(BaseModel):
    frames: list[HandShot]
    start_time: float


class DeviceRegisterRequest(BaseModel):
    device_id: str
    registration_token: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TrainingExamplePayload(BaseModel):
    id: str | None = None  # client-provided UUID; server generates one when absent
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


class UpdateExampleRequest(BaseModel):
    gesture_id: str


class ExampleResponse(BaseModel):
    id: str
    gesture_id: str
    session_id: str
    user_id: str | None = None
    hand_film: HandFilm
    created_at: float


class ExampleListResponse(BaseModel):
    examples: list[ExampleResponse]
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


class TriggerTrainingRequest(BaseModel):
    min_in_view_duration: float = 1.2
    # Class-imbalance strategy — see ml.trainer_rf_mlp.BALANCE_STRATEGIES.
    balance_strategy: str = "class_weight"


class ModelInfoResponse(BaseModel):
    model_id: str
    trainer: str
    trained_on: int
    trained_at: float
    gesture_ids: list[str]
    accuracy: float | None = None
    f1: float | None = None
    confusion_matrix: list[list[int]] | None = None
    min_in_view_duration: float | None = None
