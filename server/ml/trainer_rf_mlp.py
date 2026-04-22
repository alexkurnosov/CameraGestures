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

# Strategies to counter class imbalance (fewer examples for some gestures).
#   "class_weight": scale loss per-class by inverse frequency (sklearn 'balanced').
#                   Cheap, no new data, doesn't affect val split.
#   "jitter":       oversample minority classes with Gaussian-noise copies,
#                   applied only to X_train so validation metrics stay honest.
#   "none":         no balancing — useful as a baseline.
BALANCE_STRATEGIES = ("class_weight", "jitter", "none")
DEFAULT_BALANCE_STRATEGY = "class_weight"


def _jitter_augment_minority_classes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Oversample minority classes in the training set up to the majority count
    by adding Gaussian-noise copies of existing real examples.

    Only applied to training data; the validation split is never augmented,
    so val metrics reflect performance on real examples only.
    """
    classes, counts = np.unique(y_train, return_counts=True)
    target = int(counts.max())
    std = np.std(X_train, axis=0, keepdims=True).clip(min=1e-6)

    extra_X: list[np.ndarray] = []
    extra_y: list[np.ndarray] = []
    for c, count in zip(classes, counts):
        deficit = target - int(count)
        if deficit <= 0:
            continue
        idxs = rng.choice(np.where(y_train == c)[0], size=deficit)
        noise = rng.normal(0, 0.5, size=(deficit, X_train.shape[1])).astype(np.float32)
        extra_X.append(X_train[idxs] + noise * std)
        extra_y.append(np.full(deficit, c, dtype=y_train.dtype))

    if not extra_X:
        return X_train, y_train

    return (
        np.concatenate([X_train, *extra_X], axis=0),
        np.concatenate([y_train, *extra_y], axis=0),
    )


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


CONFIDENCE_THRESHOLDS = (0.5, 0.7, 0.9)


def _compute_extended_metrics(
    y_val: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    y_train: np.ndarray,
    gesture_ids_ordered: list[str],
) -> dict:
    """
    Compute the richer metrics bundle that /model/metrics exposes.

    Three groups:
      - per_class : precision / recall / F1 / support per gesture (including _none)
      - none_aware: FPR on synthetic _none + accuracy restricted to real gestures
      - thresholds: softmax distribution, precision/coverage at thresholds, AUCs
    """
    from sklearn.metrics import (
        precision_recall_fscore_support,
        roc_auc_score,
        average_precision_score,
    )

    n_classes = len(gesture_ids_ordered)
    labels = list(range(n_classes))

    # --- Per-class precision / recall / F1 / support ---
    precision, recall, f1_arr, support_val = precision_recall_fscore_support(
        y_val, y_pred, labels=labels, zero_division=0
    )
    # Train-side support so the UI can show class-imbalance context.
    train_support = np.bincount(y_train, minlength=n_classes)

    per_class = [
        {
            "gesture_id": gesture_ids_ordered[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1_arr[i]),
            "support_val": int(support_val[i]),
            "support_train": int(train_support[i]),
        }
        for i in range(n_classes)
    ]

    # --- _none-aware metrics ---
    try:
        none_idx = gesture_ids_ordered.index(NONE_GESTURE_ID)
    except ValueError:
        none_idx = None

    none_aware: dict = {}
    if none_idx is not None:
        none_mask = y_val == none_idx
        real_mask = ~none_mask
        n_none = int(none_mask.sum())
        n_real = int(real_mask.sum())

        # FPR on _none: of the synthetic negatives, how many got classified as a real gesture?
        if n_none > 0:
            false_positive = int((y_pred[none_mask] != none_idx).sum())
            none_aware["none_false_positive_rate"] = false_positive / n_none
            none_aware["none_support_val"] = n_none
        # Real-only accuracy: accuracy restricted to rows where the true label is a real gesture.
        if n_real > 0:
            none_aware["real_accuracy"] = float(
                (y_pred[real_mask] == y_val[real_mask]).mean()
            )
            none_aware["real_support_val"] = n_real

    # --- Confidence / threshold analysis ---
    top_conf = y_proba.max(axis=1)

    # Per-class softmax distribution for predictions actually made as that class.
    # Tells us how confident the model is when it chooses each gesture.
    confidence_by_class = []
    for i in range(n_classes):
        chose_i = y_proba[:, i] == top_conf
        scores = y_proba[chose_i, i]
        if scores.size == 0:
            confidence_by_class.append({
                "gesture_id": gesture_ids_ordered[i],
                "count": 0,
                "mean": None, "p10": None, "p50": None, "p90": None,
            })
            continue
        confidence_by_class.append({
            "gesture_id": gesture_ids_ordered[i],
            "count": int(scores.size),
            "mean": float(scores.mean()),
            "p10": float(np.percentile(scores, 10)),
            "p50": float(np.percentile(scores, 50)),
            "p90": float(np.percentile(scores, 90)),
        })

    # Precision and coverage at fixed thresholds: "if we only fire when conf > T, how accurate
    # are those fires, and what fraction of val examples do we fire on?".
    threshold_curves = []
    for t in CONFIDENCE_THRESHOLDS:
        fires = top_conf >= t
        n_fire = int(fires.sum())
        coverage = n_fire / len(y_val) if len(y_val) else 0.0
        if n_fire > 0:
            precision_at = float((y_pred[fires] == y_val[fires]).mean())
        else:
            precision_at = None
        threshold_curves.append({
            "threshold": t,
            "coverage": coverage,
            "precision": precision_at,
            "fires": n_fire,
        })

    # ROC-AUC / PR-AUC one-vs-rest, macro-averaged. Need at least 2 classes with support.
    aucs: dict = {}
    try:
        val_onehot = np.eye(n_classes)[y_val]
        # Drop classes that have no positives in val; sklearn errors otherwise.
        present = [i for i in range(n_classes) if val_onehot[:, i].sum() > 0]
        if len(present) >= 2:
            aucs["roc_auc_macro"] = float(
                roc_auc_score(
                    val_onehot[:, present], y_proba[:, present],
                    average="macro", multi_class="ovr",
                )
            )
            aucs["pr_auc_macro"] = float(
                average_precision_score(val_onehot[:, present], y_proba[:, present], average="macro")
            )
    except ValueError:
        # Degenerate val split (one class only) — leave AUCs absent.
        pass

    return {
        "per_class": per_class,
        "none_aware": none_aware,
        "confidence_by_class": confidence_by_class,
        "threshold_curves": threshold_curves,
        "auc": aucs,
        "val_size": int(len(y_val)),
        "train_size": int(len(y_train)),
    }


def train(
    examples: list[dict],
    balance_strategy: str = DEFAULT_BALANCE_STRATEGY,
) -> dict:
    """
    Train an MLP on the supplied examples and export a .tflite file.

    A synthetic ``_none`` class is added automatically so the model can
    reject hand poses that don't match any trained gesture.

    Parameters
    ----------
    examples : list of dicts with keys {gesture_id, hand_film}
    balance_strategy : one of BALANCE_STRATEGIES — how to counter class imbalance.

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
    from sklearn.utils.class_weight import compute_class_weight

    if balance_strategy not in BALANCE_STRATEGIES:
        raise ValueError(
            f"Unknown balance_strategy {balance_strategy!r}; "
            f"expected one of {BALANCE_STRATEGIES}."
        )

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
    gesture_ids_ordered = sorted({e["gesture_id"] for e in examples} | {NONE_GESTURE_ID})
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

    # --- Apply class-imbalance strategy (post-split so val stays untouched) ---
    class_weight_dict: dict[int, float] | None = None
    if balance_strategy == "class_weight":
        unique = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=unique, y=y_train)
        class_weight_dict = {int(c): float(w) for c, w in zip(unique, weights)}
    elif balance_strategy == "jitter":
        X_train, y_train = _jitter_augment_minority_classes(X_train, y_train, rng)

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
        class_weight=class_weight_dict,
        verbose=0,
    )

    # --- Evaluate ---
    y_proba = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)
    acc = float(accuracy_score(y_val, y_pred))
    f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_val, y_pred, labels=list(range(n_classes))).tolist()

    extended = _compute_extended_metrics(
        y_val=y_val,
        y_pred=y_pred,
        y_proba=y_proba,
        y_train=y_train,
        gesture_ids_ordered=gesture_ids_ordered,
    )

    metrics = {
        "accuracy": acc,
        "f1_weighted": float(f1),
        "confusion_matrix": cm,
        "gesture_ids": gesture_ids_ordered,
        "balance_strategy": balance_strategy,
        **extended,
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
