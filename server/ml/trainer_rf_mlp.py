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

NONE_GESTURE_ID = "_none"


def _generate_none_examples(
    X: np.ndarray, y: np.ndarray, per_class_count: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Synthesise negative ("none") examples so the model learns to reject
    hand poses that don't match any trained gesture.

    Three strategies, each contributing a third of the samples:
      1. **Interpolation** — lerp between examples of *different* classes.
         Produces realistic but ambiguous "in-between" hand poses.
      2. **Jitter** — add Gaussian noise to random real examples.
         Teaches the model that slightly-off poses aren't valid.
      3. **Shuffle** — randomly permute feature values.
         Creates unrealistic feature vectors well outside the real manifold.
    """
    n_total = per_class_count
    n_interp = n_total // 3
    n_jitter = n_total // 3
    n_shuffle = n_total - n_interp - n_jitter

    parts: list[np.ndarray] = []

    # 1. Interpolation between different classes
    if len(np.unique(y)) >= 2 and n_interp > 0:
        interp = np.empty((n_interp, X.shape[1]), dtype=np.float32)
        for i in range(n_interp):
            c1, c2 = rng.choice(np.unique(y), size=2, replace=False)
            a = X[rng.choice(np.where(y == c1)[0])]
            b = X[rng.choice(np.where(y == c2)[0])]
            alpha = rng.uniform(0.3, 0.7)
            interp[i] = a * alpha + b * (1 - alpha)
        parts.append(interp)

    # 2. Jittered copies
    if n_jitter > 0:
        idxs = rng.choice(len(X), size=n_jitter)
        std = np.std(X, axis=0, keepdims=True).clip(min=1e-6)
        noise = rng.normal(0, 0.5, size=(n_jitter, X.shape[1])).astype(np.float32)
        parts.append(X[idxs] + noise * std)

    # 3. Feature-shuffled
    if n_shuffle > 0:
        idxs = rng.choice(len(X), size=n_shuffle)
        shuffled = X[idxs].copy()
        for i in range(n_shuffle):
            rng.shuffle(shuffled[i])
        parts.append(shuffled)

    return np.concatenate(parts, axis=0)


def train(examples: list[dict]) -> dict:
    """
    Train an MLP on the supplied examples and export a .tflite file.

    A synthetic ``_none`` class is added automatically so the model can
    reject hand poses that don't match any trained gesture.

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

    X_real = np.array([summary_features(e["hand_film"]) for e in examples], dtype=np.float32)
    y_labels = np.array([e["gesture_id"] for e in examples])

    if len(np.unique(y_labels)) < 2:
        raise ValueError("Need examples for at least 2 different gestures to train.")

    # --- Add synthetic "_none" class ---
    # Encode real labels first so we can pass integer labels to the generator.
    real_label_enc = LabelEncoder()
    real_label_enc.fit(gesture_ids_ordered)
    y_real_int = real_label_enc.transform(y_labels)

    rng = np.random.default_rng(42)
    per_class_count = max(int(np.mean(np.bincount(y_real_int))), 5)
    X_none = _generate_none_examples(X_real, y_real_int, per_class_count, rng)

    # Combine real + none
    gesture_ids_ordered = sorted({e["gesture_id"] for e in examples}) + [NONE_GESTURE_ID]
    n_classes = len(gesture_ids_ordered)

    label_enc = LabelEncoder()
    label_enc.fit(gesture_ids_ordered)

    X = np.concatenate([X_real, X_none], axis=0)
    y = np.concatenate([
        label_enc.transform([e["gesture_id"] for e in examples]),
        np.full(len(X_none), label_enc.transform([NONE_GESTURE_ID])[0], dtype=int),
    ])

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
