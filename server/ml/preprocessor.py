"""
HandFilm → numpy feature matrix.

Mirrors FeaturePreprocessor.swift exactly:
  - Normalise all 21 landmark xyz relative to landmark 0 (wrist)
  - Compute frame-to-frame velocity (zero for first frame)
  - Pad to TARGET_FRAMES (zero-pad) or trim to last TARGET_FRAMES frames
  - Output shape: (TARGET_FRAMES, 126)
    - 63 normalised coords  (21 landmarks × 3)
    - 63 velocity values    (21 landmarks × 3)
"""

from __future__ import annotations

import numpy as np

TARGET_FRAMES = 60
LANDMARKS_PER_FRAME = 21
COORDS_PER_LANDMARK = 3
FEATURES_PER_FRAME = LANDMARKS_PER_FRAME * COORDS_PER_LANDMARK * 2  # 126


def feature_matrix(hand_film: dict) -> np.ndarray:
    """
    Convert a HandFilm dict (as stored/received from the client) to a
    float32 numpy array of shape (TARGET_FRAMES, 126).

    hand_film format:
        {
          "frames": [
            {
              "landmarks": [{"x": ..., "y": ..., "z": ...}, ...],  # 21 items
              "timestamp": ...,
              "left_or_right": ...
            },
            ...
          ],
          "start_time": ...
        }
    """
    frames = hand_film["frames"]

    # --- Extract raw coords: shape (n_frames, 21, 3) ---
    n = len(frames)
    raw = np.zeros((n, LANDMARKS_PER_FRAME, COORDS_PER_LANDMARK), dtype=np.float32)
    for i, frame in enumerate(frames):
        for j, lm in enumerate(frame["landmarks"][:LANDMARKS_PER_FRAME]):
            raw[i, j, 0] = lm["x"]
            raw[i, j, 1] = lm["y"]
            raw[i, j, 2] = lm["z"]

    # --- Normalise relative to wrist (landmark 0) ---
    wrist = raw[:, 0:1, :]           # (n, 1, 3)
    normalised = raw - wrist         # (n, 21, 3)  wrist itself becomes (0,0,0)

    # --- Velocity: frame-to-frame delta, zero for first frame ---
    velocity = np.zeros_like(normalised)
    if n > 1:
        velocity[1:] = normalised[1:] - normalised[:-1]

    # --- Concatenate coords + velocity: (n, 21, 6) → (n, 126) ---
    combined = np.concatenate([normalised, velocity], axis=2)  # (n, 21, 6)
    flat = combined.reshape(n, FEATURES_PER_FRAME)              # (n, 126)

    # --- Pad / trim to TARGET_FRAMES ---
    if n >= TARGET_FRAMES:
        result = flat[-TARGET_FRAMES:]
    else:
        pad = np.zeros((TARGET_FRAMES - n, FEATURES_PER_FRAME), dtype=np.float32)
        result = np.concatenate([flat, pad], axis=0)

    return result.astype(np.float32)


def summary_features(hand_film: dict) -> np.ndarray:
    """
    Compute the 256-feature statistical summary vector used by the MLP trainer.

    Features (256 total):
      - mean of each of 63 normalised coord dims across 60 frames  →  63
      - std  of each of 63 normalised coord dims across 60 frames  →  63
      - mean of each of 63 velocity dims across 60 frames          →  63
      - std  of each of 63 velocity dims across 60 frames          →  63
      - net displacement (last_frame − first_frame) for wrist xyz  →   3
      - dominant motion axis magnitude                             →   1
      Total: 256
    """
    mat = feature_matrix(hand_film)          # (60, 126)
    coords = mat[:, :63]                     # normalised positions
    vels   = mat[:, 63:]                     # velocities

    coord_mean = coords.mean(axis=0)         # (63,)
    coord_std  = coords.std(axis=0)          # (63,)
    vel_mean   = vels.mean(axis=0)           # (63,)
    vel_std    = vels.std(axis=0)            # (63,)

    # Net wrist displacement (landmark 0 xyz in normalised space is always 0
    # after per-frame normalisation, so use raw wrist motion instead)
    frames = hand_film["frames"]
    if len(frames) >= 2:
        first = frames[0]["landmarks"][0]
        last  = frames[-1]["landmarks"][0]
        displacement = np.array(
            [last["x"] - first["x"], last["y"] - first["y"], last["z"] - first["z"]],
            dtype=np.float32,
        )
    else:
        displacement = np.zeros(3, dtype=np.float32)

    # Dominant motion axis: max absolute mean velocity across x/y/z
    # averaged over all landmarks
    vel_xyz = vels.reshape(TARGET_FRAMES, LANDMARKS_PER_FRAME, 3)
    mean_axis = np.abs(vel_xyz.mean(axis=(0, 1)))   # (3,)
    dominant = np.array([mean_axis.max()], dtype=np.float32)

    return np.concatenate(
        [coord_mean, coord_std, vel_mean, vel_std, displacement, dominant]
    ).astype(np.float32)
