"""
Phase 2 trainer — Keras LSTM on full 60×126 sequence (stub).

Architecture (planned):
    Input(60, 126) → LSTM(64) → Dense(32, relu) → Dense(n_classes, softmax)

Export:
    tf.lite.TFLiteConverter.from_keras_model() with float16 quantisation → gesture_model.tflite

Training time:
    ~30 s on CPU for 200 examples, 10 classes

This module is a stub in phase 1. Activate by setting TRAINER=lstm in .env.
"""

from __future__ import annotations


def train(examples: list[dict]) -> dict:
    raise NotImplementedError(
        "LSTM trainer is not yet implemented (phase 2). "
        "Set TRAINER=rf_mlp in your .env to use the MLP trainer."
    )
