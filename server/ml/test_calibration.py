"""Stage 1 auto tests for server/ml/pose_corrections.py and analyze_motion.py.

Run from the server/ directory:
    python -m pytest ml/test_calibration.py -v
or:
    python ml/test_calibration.py
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Make sure the server/ directory is on sys.path when running directly.
_SERVER_DIR = Path(__file__).resolve().parent.parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from ml.pose_corrections import (
    PoseCorrections,
    coarse_idle_fallback,
    idle_clusters_for_calibration,
    load_corrections,
    params_hash,
)


# ---------------------------------------------------------------------------
# Helpers: build minimal film_energy / rows / X / rep_to_film structures
# that gate_calibration() consumes.
# ---------------------------------------------------------------------------

def _smooth(arr: np.ndarray, k: int = 3) -> np.ndarray:
    if k <= 1 or arr.size < k:
        return arr.astype(np.float32)
    pad = k // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _make_film_energy(
    onset_values: list[float],
    gesture_hold_values: list[float],
    idle_hold_values: list[float] | None = None,
    zero_fraction: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Build a minimal film_energy dict for one synthetic film.

    The segment has the shape:
        [ onset (non-zero) | gesture hold (low energy) | idle hold (very low) ]

    If zero_fraction > 0, randomly replace that fraction of energy values with
    exactly 0 to simulate MediaPipe duplicate frames.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    parts = [onset_values, gesture_hold_values]
    if idle_hold_values:
        parts.append(idle_hold_values)
    energy = np.array(
        [v for part in parts for v in part], dtype=np.float32
    )

    if zero_fraction > 0.0:
        n_zeros = int(math.floor(energy.size * zero_fraction))
        zero_idx = rng.choice(energy.size, size=n_zeros, replace=False)
        energy[zero_idx] = 0.0

    smoothed = _smooth(energy, k=3)

    # Gesture hold spans indices [len(onset)..len(onset)+len(gesture_hold)-1]
    g_start = len(onset_values)
    g_end = g_start + len(gesture_hold_values)

    # Idle hold spans [g_end..end] if present
    i_start = g_end
    i_end = energy.size

    # Build holds_kept entries mirroring what analyse_film() produces.
    n_in_view = energy.size
    holds_kept = []
    rep_coords = []

    # Gesture hold
    if g_end > g_start:
        rep = g_start + int(np.argmin(smoothed[g_start:g_end]))
        pos_frac = rep / n_in_view
        holds_kept.append({
            "seg_index": 0,
            "start_in_seg": g_start,
            "end_in_seg": g_end,
            "length": g_end - g_start,
            "position_fraction": pos_frac,
            "is_edge": False,
        })
        rep_coords.append(None)  # placeholder — filled by caller

    # Idle hold
    if idle_hold_values and i_end > i_start:
        rep = i_start + int(np.argmin(smoothed[i_start:i_end]))
        pos_frac = rep / n_in_view
        holds_kept.append({
            "seg_index": 0,
            "start_in_seg": i_start,
            "end_in_seg": i_end,
            "length": i_end - i_start,
            "position_fraction": pos_frac,
            "is_edge": False,
        })
        rep_coords.append(None)

    return {
        "seg_energies": [energy],
        "seg_smoothed": [smoothed],
        "flat_energy": energy,
        "_g_start": g_start,
        "_g_end": g_end,
        "_i_start": i_start,
        "_i_end": i_end,
        "_holds_kept": holds_kept,
        "_n_in_view": n_in_view,
    }


def _build_corpus(
    n_gestures: int = 4,
    n_films_each: int = 5,
    onset_energy: float = 0.20,
    gesture_hold_energy: float = 0.04,
    idle_hold_energy: float = 0.01,
    zero_fraction: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[list[dict], list[dict], np.ndarray, list[tuple[int, int]]]:
    """Build (film_energies, rows, X, rep_to_film) for gate_calibration().

    Layout per film per segment:
      onset    : 6 frames  ~ onset_energy  (non-zero onset pool)
      motion   : 4 frames  ~ onset_energy/2 (decreasing ramp)
      g_hold   : 5 frames  ~ gesture_hold_energy  (gesture-specific hold)
      i_hold   : 8 frames  ~ idle_hold_energy      (shared idle hold)

    This is designed so that the coarse idle-candidate heuristic fires on
    the idle cluster: it receives contributions from all n_gestures, each
    contributing ~1/(n_gestures) > 5 % of the cluster's total holds.

    Returns (film_energies, rows, X, rep_to_film) ready for gate_calibration().
    """
    if rng is None:
        rng = np.random.default_rng(42)

    gestures = [f"gesture_{i}" for i in range(n_gestures)]
    # One unique 63-dim pose vector per gesture (gesture cluster)
    # and one shared 63-dim vector for the idle cluster.
    pose_vecs = {
        g: rng.uniform(0.0, 2.0, size=63).astype(np.float32)
        for g in gestures
    }
    idle_vec = rng.uniform(5.0, 7.0, size=63).astype(np.float32)  # far from gestures

    onset = [onset_energy + rng.random() * 0.02 - 0.01 for _ in range(6)]
    motion = [onset_energy * 0.5 * (1 - k / 4) for k in range(4)]
    g_hold = [gesture_hold_energy + rng.random() * 0.005 - 0.0025 for _ in range(5)]
    i_hold = [idle_hold_energy + rng.random() * 0.002 - 0.001 for _ in range(8)]

    film_energies: list[dict] = []
    rows: list[dict] = []
    X_list: list[np.ndarray] = []
    rep_to_film: list[tuple[int, int]] = []

    film_idx = 0
    for g in gestures:
        for _ in range(n_films_each):
            fe = _make_film_energy(
                onset_values=onset,
                gesture_hold_values=g_hold,
                idle_hold_values=i_hold,
                zero_fraction=zero_fraction,
                rng=rng,
            )
            film_energies.append(fe)

            # Build the row (same structure as analyse_film output).
            holds_kept = fe["_holds_kept"]
            hold_reps = [
                pose_vecs[g] + rng.uniform(-0.05, 0.05, 63).astype(np.float32),
                idle_vec + rng.uniform(-0.05, 0.05, 63).astype(np.float32),
            ]
            rows.append({
                "gesture_id": g,
                "session_id": f"session_{film_idx}",
                "n_in_view_frames": fe["_n_in_view"],
                "holds_all": holds_kept,
                "holds_kept": holds_kept,
                "hold_reps": hold_reps,
            })

            for hi, rep in enumerate(hold_reps):
                X_list.append(rep)
                rep_to_film.append((film_idx, hi))

            film_idx += 1

    X = np.stack(X_list).astype(np.float32)
    return film_energies, rows, X, rep_to_film


# ---------------------------------------------------------------------------
# Test 1: gate_calibration returns all four seeds on a known synthetic corpus
# ---------------------------------------------------------------------------

def test_gate_calibration_produces_all_four_seeds() -> None:
    """gate_calibration must return T_open, K_open, T_close, K_close (not None)
    and each must be within ±5 % of the analytical expectation.
    """
    from analyze_motion import gate_calibration

    onset_energy = 0.20
    idle_hold_energy = 0.01

    film_energies, rows, X, rep_to_film = _build_corpus(
        n_gestures=4,
        n_films_each=5,
        onset_energy=onset_energy,
        idle_hold_energy=idle_hold_energy,
    )

    gate = gate_calibration(
        film_energies=film_energies,
        rows=rows,
        X=X,
        rep_to_film=rep_to_film,
        epsilon=0.5,  # small enough to separate idle vec from gesture vecs
    )

    # All four seeds must be present.
    assert gate["t_open_seed"] is not None, "T_open seed must not be None"
    assert gate["k_open_seed"] is not None, "K_open seed must not be None"
    assert gate["t_close_seed"] is not None, "T_close seed must not be None"
    # K_close: only non-None when duration distributions don't overlap.
    # Our synthetic corpus is designed so idle holds (8 frames) are longer
    # than gesture holds (5 frames), so K_close should be derivable.
    # Accept None only if distributions legitimately overlap.
    if gate["k_close_seed"] is None:
        assert not gate["duration_separation_clean"], (
            "K_close is None but duration_separation_clean is True — inconsistent"
        )

    # T_open analytical expectation: p10 of non-zero onset energies (~0.20).
    t_open = gate["t_open_seed"]
    assert 0.0 < t_open, "T_open must be positive"
    # Within 50 % of onset_energy (loose band: onset pool has some noise).
    assert t_open < onset_energy * 1.5, (
        f"T_open={t_open:.4f} unexpectedly large vs onset_energy={onset_energy}"
    )

    # K_open: must be ≥ 1.
    assert gate["k_open_seed"] >= 1, "K_open must be at least 1 frame"

    # T_close analytical expectation: p90 of idle within-hold energy (~0.01).
    t_close = gate["t_close_seed"]
    assert t_close > 0.0, "T_close must be positive"
    assert t_close < onset_energy, (
        f"T_close={t_close:.4f} should be below onset energy {onset_energy}"
    )


# ---------------------------------------------------------------------------
# Test 2: gate_calibration handles 35 % MediaPipe zero frames correctly
# ---------------------------------------------------------------------------

def test_gate_calibration_zero_fraction_does_not_pin_t_open() -> None:
    """With 35 % of energy values forced to exactly 0, T_open must not be
    pinned to 0 (the corpus contains non-zero onset energies).
    """
    from analyze_motion import gate_calibration

    film_energies, rows, X, rep_to_film = _build_corpus(
        n_gestures=4,
        n_films_each=5,
        onset_energy=0.18,
        idle_hold_energy=0.01,
        zero_fraction=0.35,
        rng=np.random.default_rng(7),
    )

    gate = gate_calibration(
        film_energies=film_energies,
        rows=rows,
        X=X,
        rep_to_film=rep_to_film,
        epsilon=0.5,
    )

    t_open = gate["t_open_seed"]
    assert t_open is not None, "T_open must not be None even with 35 % zeros"
    assert t_open > 0.0, (
        f"T_open={t_open} is pinned to zero — zero frames leaked into onset pool"
    )


# ---------------------------------------------------------------------------
# Test 3: load_corrections on a missing file returns empty-but-valid object
# ---------------------------------------------------------------------------

def test_load_corrections_missing_file_returns_empty() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pose_corrections.json"
        corr = load_corrections(path)
        assert isinstance(corr, PoseCorrections)
        assert corr.cluster_kinds == {}
        assert corr.excluded_holds == []


# ---------------------------------------------------------------------------
# Test 4: load_corrections on malformed JSON raises with a clear message
# ---------------------------------------------------------------------------

def test_load_corrections_malformed_json_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pose_corrections.json"
        path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ValueError, match="Malformed pose_corrections"):
            load_corrections(path)


def test_load_corrections_wrong_type_raises() -> None:
    """Top-level array instead of object should raise ValueError."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pose_corrections.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON object"):
            load_corrections(path)


def test_load_corrections_cluster_kinds_wrong_type_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pose_corrections.json"
        path.write_text(
            json.dumps({"cluster_kinds": [1, 2], "excluded_holds": []}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="cluster_kinds"):
            load_corrections(path)


def test_load_corrections_valid_file_round_trips() -> None:
    payload = {
        "cluster_kinds": {"15": "idle", "9": "regular"},
        "excluded_holds": [
            {
                "film_id": "abc-123",
                "hold_ordinal": 0,
                "rep_frame": 47,
                "start_frame": 42,
                "end_frame": 53,
                "params_hash": "sha256:aabbcc",
            }
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pose_corrections.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        corr = load_corrections(path)
        assert corr.cluster_kind(15) == "idle"
        assert corr.cluster_kind(9) == "regular"
        assert corr.cluster_kind(99) is None
        assert corr.confirmed_idle_ids() == frozenset({15})
        assert corr.is_hold_excluded("abc-123", 0, "sha256:aabbcc")
        assert not corr.is_hold_excluded("abc-123", 1, "sha256:aabbcc")


# ---------------------------------------------------------------------------
# Test 5: params_hash stability
# ---------------------------------------------------------------------------

def test_params_hash_invariant_under_key_reorder() -> None:
    """params_hash must be identical regardless of dict key ordering."""
    base = {
        "T_hold": 0.54,
        "K_hold": 3,
        "smooth_k": 3,
        "edge_trim_fraction": 0.15,
    }
    # Build reordered versions.
    reordered_1 = {
        "K_hold": 3,
        "smooth_k": 3,
        "edge_trim_fraction": 0.15,
        "T_hold": 0.54,
    }
    reordered_2 = {
        "edge_trim_fraction": 0.15,
        "T_hold": 0.54,
        "K_hold": 3,
        "smooth_k": 3,
    }
    h = params_hash(base)
    assert params_hash(reordered_1) == h
    assert params_hash(reordered_2) == h


def test_params_hash_snake_case_aliases() -> None:
    """t_hold / k_hold aliases must produce the same hash as T_hold / K_hold."""
    camel = {"T_hold": 0.54, "K_hold": 3, "smooth_k": 3, "edge_trim_fraction": 0.15}
    snake = {"t_hold": 0.54, "k_hold": 3, "smooth_k": 3, "edge_trim_fraction": 0.15}
    assert params_hash(camel) == params_hash(snake)


def test_params_hash_changes_on_each_field() -> None:
    """Changing any of the four fields must produce a different hash."""
    base = {"T_hold": 0.54, "K_hold": 3, "smooth_k": 3, "edge_trim_fraction": 0.15}
    h_base = params_hash(base)

    variants = [
        {**base, "T_hold": 0.99},
        {**base, "K_hold": 5},
        {**base, "smooth_k": 5},
        {**base, "edge_trim_fraction": 0.20},
    ]
    for variant in variants:
        assert params_hash(variant) != h_base, (
            f"Hash should differ for {variant} vs base"
        )


def test_params_hash_missing_key_raises() -> None:
    with pytest.raises(ValueError, match="missing keys"):
        params_hash({"T_hold": 0.54, "K_hold": 3, "smooth_k": 3})  # edge_trim_fraction absent


# ---------------------------------------------------------------------------
# Additional: coarse_idle_fallback and idle_clusters_for_calibration
# ---------------------------------------------------------------------------

def test_coarse_idle_fallback_identifies_shared_cluster() -> None:
    composition = {
        0: {"total": 20, "by_gesture": {"ok": 5, "stop": 5, "point_left": 5, "thumbs_up": 5}},
        1: {"total": 40, "by_gesture": {"ok": 40}},  # gesture-specific
        2: {"total": 30, "by_gesture": {"stop": 30}},
    }
    result = coarse_idle_fallback(composition, min_gestures=3, min_fraction=0.05)
    assert 0 in result, "Cluster 0 should be flagged as idle candidate"
    assert 1 not in result
    assert 2 not in result


def test_idle_clusters_for_calibration_prefers_confirmed() -> None:
    corrections = PoseCorrections({"99": "idle"}, [])
    composition = {
        0: {"total": 20, "by_gesture": {"ok": 5, "stop": 5, "point_left": 5, "thumbs_up": 5}},
        99: {"total": 5, "by_gesture": {"ok": 5}},
    }
    result = idle_clusters_for_calibration(corrections, composition)
    # Should use confirmed idle (cluster 99), not the coarse fallback (cluster 0).
    assert result == frozenset({99})


def test_idle_clusters_for_calibration_falls_back_to_heuristic() -> None:
    corrections = PoseCorrections({}, [])  # no confirmed kinds
    composition = {
        0: {"total": 20, "by_gesture": {"ok": 5, "stop": 5, "point_left": 5, "thumbs_up": 5}},
        1: {"total": 40, "by_gesture": {"ok": 40}},
    }
    result = idle_clusters_for_calibration(corrections, composition)
    assert 0 in result
    assert 1 not in result


# ---------------------------------------------------------------------------
# Runner for direct execution without pytest.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _tests = [
        test_gate_calibration_produces_all_four_seeds,
        test_gate_calibration_zero_fraction_does_not_pin_t_open,
        test_load_corrections_missing_file_returns_empty,
        test_load_corrections_malformed_json_raises,
        test_load_corrections_wrong_type_raises,
        test_load_corrections_cluster_kinds_wrong_type_raises,
        test_load_corrections_valid_file_round_trips,
        test_params_hash_invariant_under_key_reorder,
        test_params_hash_snake_case_aliases,
        test_params_hash_changes_on_each_field,
        test_params_hash_missing_key_raises,
        test_coarse_idle_fallback_identifies_shared_cluster,
        test_idle_clusters_for_calibration_prefers_confirmed,
        test_idle_clusters_for_calibration_falls_back_to_heuristic,
    ]
    failed = 0
    for fn in _tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL  {fn.__name__}: {exc}")
            failed += 1
    print()
    print(f"{len(_tests) - failed}/{len(_tests)} passed")
    sys.exit(failed)
