"""
Thin wrapper around preprocessor.js executed via py_mini_racer (V8).

All preprocessing logic lives in preprocessor.js — this file only handles
loading the JS and converting results to numpy arrays.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import py_mini_racer

TARGET_FRAMES = 60   # kept for callers that import this constant
_JS_PATH = Path(__file__).with_name("preprocessor.js")

# One V8 context per thread — avoids locking while keeping thread safety.
_local = threading.local()


def _ctx() -> py_mini_racer.MiniRacer:
    if not hasattr(_local, "ctx"):
        _local.ctx = py_mini_racer.MiniRacer()
        _local.ctx.eval(_JS_PATH.read_text(encoding="utf-8"))
    return _local.ctx


def feature_matrix(hand_film: dict) -> np.ndarray:
    """Return float32 array of shape (TARGET_FRAMES, 126)."""
    flat = _ctx().call("featureMatrix", hand_film)
    return np.array(flat, dtype=np.float32).reshape(TARGET_FRAMES, 126)


def summary_features(hand_film: dict) -> np.ndarray:
    """Return float32 array of shape (256,)."""
    result = _ctx().call("summaryFeatures", hand_film)
    return np.array(result, dtype=np.float32)
