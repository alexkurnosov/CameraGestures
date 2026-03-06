"""
Phase 1 trainer — shallow Keras MLP on statistical summary features.

Input per example : 256-feature vector (see preprocessor.summary_features)
Architecture      : Dense(128, relu) → Dropout(0.3) → Dense(64, relu) → Dense(n_classes, softmax)
Export            : tf.lite.TFLiteConverter → gesture_model.tflite
Training time     : < 5 s on CPU for ≤ 500 examples, 10 classes
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import numpy as np

from config import settings
from ml.preprocessor import summary_features


def train(examples: list[dict]) -> dict:
    """
    Train an MLP on the supplied examples and export a .tflite file.

    Parameters
    ----------
    examples : list of dicts with keys {gesture_id, hand_film}

    Returns
    -------
    dict with keys:
        tflite_path : Path
        gesture_ids : list[str]
        trained_on  : int
        metrics     : dict  (accuracy, f1 per class, confusion matrix)
    """
    # Deferred import so startup is fast when TF is not used
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    if len(examples) < 2:
        raise ValueError("Need at least 2 training examples.")

    gesture_ids_ordered = sorted({e["gesture_id"] for e in examples})
    label_enc = LabelEncoder()
    label_enc.fit(gesture_ids_ordered)
    n_classes = len(gesture_ids_ordered)

    X = np.array([summary_features(e["hand_film"]) for e in examples], dtype=np.float32)
    y = label_enc.transform([e["gesture_id"] for e in examples])

    if len(np.unique(y)) < 2:
        raise ValueError("Need examples for at least 2 different gestures to train.")

    # Split for evaluation (80/20); fall back to no split if dataset is tiny
    if len(X) >= 10:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_val, y_train, y_val = X, X, y, y

    # --- Build Keras MLP ---
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(256,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    epochs = min(100, max(20, len(X_train) * 2))
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=min(32, len(X_train)),
        validation_data=(X_val, y_val),
        verbose=0,
    )

    # --- Evaluate ---
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    acc = float(accuracy_score(y_val, y_pred))
    f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_val, y_pred, labels=list(range(n_classes))).tolist()

    metrics = {
        "accuracy": acc,
        "f1_weighted": float(f1),
        "confusion_matrix": cm,
        "gesture_ids": gesture_ids_ordered,
    }

    # --- Export to TFLite ---
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    model_id = str(uuid.uuid4())
    tflite_path = settings.models_dir / f"{model_id}.tflite"
    tflite_path.write_bytes(tflite_model)

    return {
        "tflite_path": tflite_path,
        "gesture_ids": gesture_ids_ordered,
        "trained_on": len(examples),
        "metrics": metrics,
        "model_id": model_id,
    }
