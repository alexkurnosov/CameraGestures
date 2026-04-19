"""
Shared in-process training state.

Kept in its own module so both routers/examples.py and routers/training.py
can import it without creating circular dependencies.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingState:
    status: str = "idle"          # "idle" | "training" | "ready" | "failed"
    job_id: Optional[str] = None
    accuracy: Optional[float] = None
    trained_on: int = 0
    gesture_ids: list[str] = field(default_factory=list)
    trained_at: Optional[float] = None
    error: Optional[str] = None
    min_in_view_duration: float = 1.2
    balance_strategy: str = "class_weight"
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)

    def is_running(self) -> bool:
        return self.status == "training"


# Module-level singleton — shared across all requests within the process
training_state = TrainingState()
