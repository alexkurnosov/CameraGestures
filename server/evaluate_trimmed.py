"""
Trimmed-clip evaluation — verifies Problem A (length-dependent feature skew).

Trains an MLP on full-length training clips, then evaluates the same model on
the validation set at multiple artificial trim lengths. A trim length of N
keeps only the last N non-absent frames of a clip; earlier frames are marked
is_absent so the preprocessor drops them. This mimics the short runtime buffer
the device sees at inference time (≈10 frames) while training films are
~60 frames.

Signals:
  * Feature drift  — mean L2 distance between full-length and trimmed summary
    feature vectors for the same clip. If small, the preprocessor is
    length-invariant. If large, runtime features live in a different region
    of feature space than training features.
  * Real-gesture accuracy at each trim length. If it tracks full-length
    accuracy, the fix works end-to-end. A sharp drop with smaller N points
    to genuine length-sensitivity beyond padding (Problem B).

Usage (from the server/ directory, same place the server is run):
    python evaluate_trimmed.py                       # read from DB
    python evaluate_trimmed.py --json report.json    # also write JSON
    python evaluate_trimmed.py --dry-run             # synthetic data, no DB
    python evaluate_trimmed.py --trim-lengths 8,12,15,20,30

Read-only with respect to the DB. Writes nothing except the optional JSON.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ml.preprocessor import summary_features

NONE_GESTURE_ID = "_none"
DEFAULT_TRIM_LENGTHS = (60, 30, 20, 15, 12, 10, 8)
ABSENT_LANDMARKS = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(21)]


# ---------------------------------------------------------------------------
# Clip trimming.
# ---------------------------------------------------------------------------

def trim_hand_film(hand_film: dict, n_real_frames: int) -> dict:
    """Return a shallow copy of hand_film with only the last n_real_frames
    non-absent frames preserved. All earlier originally-non-absent frames are
    rewritten as is_absent=True so the preprocessor skips them.

    The preprocessor treats absent frames as excluded from both normalisation
    and stats, so marking them absent is equivalent to deletion — but keeping
    the array shape makes debugging timestamps easier.
    """
    src = hand_film["frames"]
    keep_indices = [i for i, f in enumerate(src) if not f.get("is_absent")]
    if len(keep_indices) <= n_real_frames:
        return hand_film

    keep_set = set(keep_indices[-n_real_frames:])
    new_frames = []
    for i, f in enumerate(src):
        if f.get("is_absent") or i in keep_set:
            new_frames.append(f)
        else:
            g = dict(f)
            g["is_absent"] = True
            g["landmarks"] = ABSENT_LANDMARKS
            new_frames.append(g)
    return {**hand_film, "frames": new_frames}


def count_real_frames(hand_film: dict) -> int:
    return sum(1 for f in hand_film["frames"] if not f.get("is_absent"))


# ---------------------------------------------------------------------------
# Synthetic _none class (mirrors trainer_rf_mlp._generate_none_examples).
# ---------------------------------------------------------------------------

def generate_none_examples(
    X: np.ndarray, y: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, X.shape[1]), dtype=np.float32)

    n_interp = count // 3
    n_jitter = count // 3
    n_shuffle = count - n_interp - n_jitter
    parts: list[np.ndarray] = []

    classes = np.unique(y)
    if len(classes) >= 2 and n_interp > 0:
        interp = np.empty((n_interp, X.shape[1]), dtype=np.float32)
        for i in range(n_interp):
            c1, c2 = rng.choice(classes, size=2, replace=False)
            a = X[rng.choice(np.where(y == c1)[0])]
            b = X[rng.choice(np.where(y == c2)[0])]
            alpha = rng.uniform(0.3, 0.7)
            interp[i] = a * alpha + b * (1 - alpha)
        parts.append(interp)

    if n_jitter > 0:
        idxs = rng.choice(len(X), size=n_jitter)
        std = np.std(X, axis=0, keepdims=True).clip(min=1e-6)
        noise = rng.normal(0, 0.5, size=(n_jitter, X.shape[1])).astype(np.float32)
        parts.append(X[idxs] + noise * std)

    if n_shuffle > 0:
        idxs = rng.choice(len(X), size=n_shuffle)
        shuffled = X[idxs].copy()
        for i in range(n_shuffle):
            rng.shuffle(shuffled[i])
        parts.append(shuffled)

    return np.concatenate(parts, axis=0) if parts else np.zeros((0, X.shape[1]), dtype=np.float32)


# ---------------------------------------------------------------------------
# Train + predict. Thin wrapper over Keras; kept local so the eval is
# self-contained and doesn't drag in training-side side effects (tflite
# export, UUID assignment, etc.).
# ---------------------------------------------------------------------------

def train_mlp(
    X_train: np.ndarray, y_train: np.ndarray, n_classes: int, epochs: int
):
    import tensorflow as tf
    from sklearn.utils.class_weight import compute_class_weight

    unique = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=unique, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(unique, weights)}

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=min(32, len(X_train)),
        class_weight=class_weight,
        verbose=0,
    )
    return model


def predict(model, X: np.ndarray) -> np.ndarray:
    return model.predict(X, verbose=0)


# ---------------------------------------------------------------------------
# Per-class metrics.
# ---------------------------------------------------------------------------

def per_class_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, label_order: list[str],
    label_to_idx: dict[str, int],
) -> list[dict]:
    out: list[dict] = []
    for name in label_order:
        idx = label_to_idx[name]
        mask = y_true == idx
        support = int(mask.sum())
        if support == 0:
            out.append({"gesture_id": name, "support": 0, "accuracy": None})
            continue
        acc = float((y_pred[mask] == idx).mean())
        out.append({"gesture_id": name, "support": support, "accuracy": acc})
    return out


# ---------------------------------------------------------------------------
# Core evaluation.
# ---------------------------------------------------------------------------

def evaluate(examples: list[dict], args: argparse.Namespace) -> dict:
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    real_gestures = sorted({e["gesture_id"] for e in examples})
    if len(real_gestures) < 2:
        print("Need ≥ 2 real gesture classes to evaluate.", file=sys.stderr)
        sys.exit(2)

    labels = real_gestures + [NONE_GESTURE_ID]
    label_enc = LabelEncoder().fit(labels)
    label_to_idx = {name: int(label_enc.transform([name])[0]) for name in labels}
    n_classes = len(labels)

    # Stratified split on REAL examples only. We trim val clips; train always
    # uses full-length features.
    indices = np.arange(len(examples))
    y_real_str = np.array([e["gesture_id"] for e in examples])
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y_real_str
    )
    train_examples = [examples[i] for i in train_idx]
    val_examples   = [examples[i] for i in val_idx]

    # --- Features ---
    X_train_real = np.array(
        [summary_features(e["hand_film"]) for e in train_examples], dtype=np.float32
    )
    y_train_real = np.array(
        [label_to_idx[e["gesture_id"]] for e in train_examples], dtype=np.int64
    )

    rng = np.random.default_rng(42)
    per_class_count = max(int(np.mean(np.bincount(y_train_real))), 5)
    X_none = generate_none_examples(X_train_real, y_train_real, per_class_count, rng)
    y_none = np.full(len(X_none), label_to_idx[NONE_GESTURE_ID], dtype=np.int64)

    X_train = np.concatenate([X_train_real, X_none], axis=0)
    y_train = np.concatenate([y_train_real, y_none], axis=0)

    # Validation: full-length features once, then trimmed variants.
    X_val_full = np.array(
        [summary_features(e["hand_film"]) for e in val_examples], dtype=np.float32
    )
    y_val = np.array(
        [label_to_idx[e["gesture_id"]] for e in val_examples], dtype=np.int64
    )

    # --- Train ---
    epochs = min(100, max(20, len(X_train) * 2))
    model = train_mlp(X_train, y_train, n_classes=n_classes, epochs=epochs)

    # --- Sweep trim lengths ---
    real_frame_counts = [count_real_frames(e["hand_film"]) for e in val_examples]
    max_real = max(real_frame_counts) if real_frame_counts else 0

    trim_lengths = sorted(set(args.trim_lengths), reverse=True)
    rows: list[dict] = []

    # Full-length baseline.
    y_full_proba = predict(model, X_val_full)
    y_full_pred = np.argmax(y_full_proba, axis=1)
    acc_full = float((y_full_pred == y_val).mean())
    rows.append({
        "trim_length": None,
        "label": "full",
        "accuracy": acc_full,
        "mean_real_frames": float(np.mean(real_frame_counts)) if real_frame_counts else 0.0,
        "feature_drift_l2_mean": 0.0,
        "per_class": per_class_accuracy(y_val, y_full_pred, real_gestures, label_to_idx),
    })

    for N in trim_lengths:
        trimmed_films = [trim_hand_film(e["hand_film"], N) for e in val_examples]
        X_trim = np.array(
            [summary_features(hf) for hf in trimmed_films], dtype=np.float32
        )
        drift = np.linalg.norm(X_trim - X_val_full, axis=1)
        y_trim_pred = np.argmax(predict(model, X_trim), axis=1)
        acc_trim = float((y_trim_pred == y_val).mean())
        effective_frames = [min(N, c) for c in real_frame_counts]
        rows.append({
            "trim_length": N,
            "label": f"trim_{N}",
            "accuracy": acc_trim,
            "mean_real_frames": float(np.mean(effective_frames)),
            "feature_drift_l2_mean": float(drift.mean()),
            "feature_drift_l2_p90": float(np.percentile(drift, 90)),
            "per_class": per_class_accuracy(y_val, y_trim_pred, real_gestures, label_to_idx),
        })

    report: dict[str, Any] = {
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val_full)),
        "gesture_ids": real_gestures,
        "max_real_frames_in_val": int(max_real),
        "rows": rows,
    }
    return report


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------

def print_report(report: dict) -> None:
    print("=" * 72)
    print("Trimmed-clip evaluation")
    print("=" * 72)
    print()
    print(f"Train size (with synthetic _none): {report['train_size']}")
    print(f"Val size (real only):              {report['val_size']}")
    print(f"Max real frames in val:            {report['max_real_frames_in_val']}")
    print(f"Real gestures: {', '.join(report['gesture_ids'])}")
    print()

    print(f"  {'trim':>6s}  {'mean_frames':>11s}  {'accuracy':>8s}  "
          f"{'drift_mean':>10s}  {'drift_p90':>9s}")
    for row in report["rows"]:
        trim_label = "full" if row["trim_length"] is None else str(row["trim_length"])
        drift_mean = f"{row['feature_drift_l2_mean']:.4f}"
        drift_p90 = f"{row.get('feature_drift_l2_p90', 0.0):.4f}" if row["trim_length"] else "   n/a"
        print(
            f"  {trim_label:>6s}  {row['mean_real_frames']:>11.1f}  "
            f"{row['accuracy']:>8.4f}  {drift_mean:>10s}  {drift_p90:>9s}"
        )

    print()
    print("Per-class accuracy by trim length:")
    gestures = report["gesture_ids"]
    header = f"  {'trim':>6s}  " + "  ".join(f"{g[:12]:>12s}" for g in gestures)
    print(header)
    for row in report["rows"]:
        trim_label = "full" if row["trim_length"] is None else str(row["trim_length"])
        cells = []
        for g in gestures:
            entry = next((p for p in row["per_class"] if p["gesture_id"] == g), None)
            cells.append(f"{entry['accuracy']:>12.3f}" if entry and entry["accuracy"] is not None else f"{'n/a':>12s}")
        print(f"  {trim_label:>6s}  " + "  ".join(cells))

    print()
    print("Interpretation:")
    print("  * accuracy ≈ full-length across trim lengths → Problem A (length-")
    print("    dependent feature skew) is not impacting real-gesture classification.")
    print("  * accuracy drops sharply as trim length shrinks → length sensitivity")
    print("    remains (Problem B). Consider adding duration feature, time-warp")
    print("    augmentation, or a motion-gated variable-length capture.")
    print("  * drift_mean large but accuracy stable → feature vectors shift but")
    print("    model is robust (likely due to MLP generalisation or class-weighted")
    print("    training). Still worth monitoring on device.")


# ---------------------------------------------------------------------------
# Data loading.
# ---------------------------------------------------------------------------

async def load_from_db() -> list[dict]:
    from storage.example_store import load_all_examples
    return await load_all_examples()


def generate_synthetic_examples() -> list[dict]:
    """Four gestures × 10 films × ~50 frames each. Each gesture has a
    distinctive hand pose so a trivial classifier can separate them.
    """
    rng = random.Random(42)
    gestures = {
        "gesture_a": 1.0,
        "gesture_b": 2.0,
        "gesture_c": 3.0,
        "gesture_d": 4.0,
    }

    def frame(offset: float, idx: int) -> dict:
        landmarks = []
        for j in range(21):
            angle = (j / 21.0) * math.pi * 2.0 + offset
            r = 0.1
            landmarks.append({
                "x": 0.5 + math.cos(angle) * r + rng.gauss(0, 0.001),
                "y": 0.5 + math.sin(angle) * r + rng.gauss(0, 0.001),
                "z": 0.01 * j + rng.gauss(0, 0.001),
            })
        # Force wrist → landmark 9 to be well-defined for normalisation.
        landmarks[0] = {"x": 0.5, "y": 0.5, "z": 0.0}
        landmarks[9] = {"x": 0.5, "y": 0.35 + offset * 0.02, "z": 0.0}
        return {
            "landmarks": landmarks,
            "timestamp": idx / 30.0,
            "left_or_right": "right",
            "is_absent": False,
        }

    examples: list[dict] = []
    for gid, offset in gestures.items():
        for s in range(10):
            frames = [frame(offset + rng.gauss(0, 0.02), i) for i in range(50)]
            examples.append({
                "id": f"synthetic-{gid}-{s}",
                "gesture_id": gid,
                "session_id": f"session-{s}",
                "user_id": "synthetic",
                "hand_film": {"frames": frames, "start_time": 0.0},
                "created_at": 0.0,
            })
    return examples


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use a synthetic dataset instead of reading from the database.",
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Also write the full report as JSON to this file.",
    )
    parser.add_argument(
        "--trim-lengths", type=str, default=None,
        help=(
            "Comma-separated integer trim lengths in frames. "
            f"Default: {','.join(str(n) for n in DEFAULT_TRIM_LENGTHS)}."
        ),
    )
    args = parser.parse_args()

    if args.trim_lengths:
        args.trim_lengths = [int(x) for x in args.trim_lengths.split(",") if x.strip()]
    else:
        args.trim_lengths = list(DEFAULT_TRIM_LENGTHS)

    if args.dry_run:
        examples = generate_synthetic_examples()
        print(f"(dry-run) generated {len(examples)} synthetic examples", file=sys.stderr)
    else:
        examples = asyncio.run(load_from_db())
        print(f"Loaded {len(examples)} examples from database", file=sys.stderr)

    if not examples:
        print("No examples to evaluate.", file=sys.stderr)
        sys.exit(1)

    report = evaluate(examples, args)
    print_report(report)

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"Wrote JSON report to {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
