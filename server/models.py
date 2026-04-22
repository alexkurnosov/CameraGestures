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


# --- Detailed metrics (public /model/metrics endpoint) ---

class PerClassMetric(BaseModel):
    gesture_id: str
    precision: float
    recall: float
    f1: float
    support_val: int
    support_train: int


class ConfidenceByClass(BaseModel):
    gesture_id: str
    count: int
    mean: float | None = None
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None


class ThresholdPoint(BaseModel):
    threshold: float
    coverage: float
    precision: float | None = None
    fires: int


class NoneAwareMetrics(BaseModel):
    none_false_positive_rate: float | None = None
    none_support_val: int | None = None
    real_accuracy: float | None = None
    real_support_val: int | None = None


class AucMetrics(BaseModel):
    roc_auc_macro: float | None = None
    pr_auc_macro: float | None = None


class ModelMetricsResponse(BaseModel):
    model_id: str
    trainer: str
    trained_at: float
    trained_on: int
    gesture_ids: list[str]
    balance_strategy: str | None = None

    # Aggregate (same as /model/info, repeated so /metrics is self-contained).
    accuracy: float | None = None
    f1_weighted: float | None = None
    confusion_matrix: list[list[int]] | None = None

    # Split sizes after train/val split.
    val_size: int | None = None
    train_size: int | None = None

    # Detailed bundles.
    per_class: list[PerClassMetric] = Field(default_factory=list)
    none_aware: NoneAwareMetrics = Field(default_factory=NoneAwareMetrics)
    confidence_by_class: list[ConfidenceByClass] = Field(default_factory=list)
    threshold_curves: list[ThresholdPoint] = Field(default_factory=list)
    auc: AucMetrics = Field(default_factory=AucMetrics)


class ModelMetricsSummary(BaseModel):
    model_id: str
    trained_at: float
    trained_on: int
    accuracy: float | None = None
    f1_weighted: float | None = None


class ModelMetricsListResponse(BaseModel):
    models: list[ModelMetricsSummary]
