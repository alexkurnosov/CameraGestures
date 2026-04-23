"""
One-off calibration script for Phase 2 (key-pose) parameter tuning.

Reads all stored training examples, computes per-frame motion energy in the
normalised coord space, detects hold segments, and prints a report that feeds
Phase 2 parameter choices in Plans/PLAN_3phase_recognition.md.

Output is printed to stdout; with --json it also dumps the full statistics as
machine-readable JSON so you can paste it back into the design discussion.

Usage (from the server/ directory, same place the server is run):

    python analyze_motion.py                       # read real data from the DB
    python analyze_motion.py --json report.json    # also dump JSON
    python analyze_motion.py --dry-run             # use synthetic data (no DB)

The script is read-only. It does not modify the database or any server state.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import py_mini_racer
from sklearn.cluster import AgglomerativeClustering


# ---------------------------------------------------------------------------
# JS preprocessor + helper for in-view normalised coords per frame.
# ---------------------------------------------------------------------------

_JS_PREPROCESSOR_PATH = Path(__file__).parent / "ml" / "preprocessor.js"

# Adds one function to the preprocessor's global scope that returns, for each
# non-absent frame, (original_index, normalised_coords_63). The normalisation
# pipeline is the same as featureMatrix() — wrist-relative, scale-normalised,
# rotation-aligned — but without velocity, padding, or trimming.
_JS_HELPER = r"""
function inViewNormalisedCoords(handFilm) {
    var src = handFilm.frames;
    var out = [];
    for (var i = 0; i < src.length; i++) {
        if (src[i].is_absent) continue;
        var landmarks = src[i].landmarks;
        var wrist = landmarks[0];
        var xSign = src[i].left_or_right === "left" ? -1 : 1;

        var rel = [];
        for (var j = 0; j < 21; j++) {
            rel.push([
                (landmarks[j].x - wrist.x) * xSign,
                landmarks[j].y - wrist.y,
                landmarks[j].z - wrist.z
            ]);
        }

        var lm9 = rel[9];
        var scale = Math.sqrt(lm9[0]*lm9[0] + lm9[1]*lm9[1] + lm9[2]*lm9[2]);
        if (scale < 1e-6) scale = 1.0;
        for (var j = 0; j < 21; j++) {
            rel[j][0] /= scale; rel[j][1] /= scale; rel[j][2] /= scale;
        }

        var up = [rel[9][0], rel[9][1], rel[9][2]];
        var rApprox = [
            rel[17][0] - rel[5][0],
            rel[17][1] - rel[5][1],
            rel[17][2] - rel[5][2]
        ];
        var dotRU = rApprox[0]*up[0] + rApprox[1]*up[1] + rApprox[2]*up[2];
        var right = [
            rApprox[0] - dotRU * up[0],
            rApprox[1] - dotRU * up[1],
            rApprox[2] - dotRU * up[2]
        ];
        var rMag = Math.sqrt(right[0]*right[0] + right[1]*right[1] + right[2]*right[2]);
        if (rMag < 1e-6) { right = [1, 0, 0]; }
        else { right[0] /= rMag; right[1] /= rMag; right[2] /= rMag; }
        var forward = [
            up[1]*right[2] - up[2]*right[1],
            up[2]*right[0] - up[0]*right[2],
            up[0]*right[1] - up[1]*right[0]
        ];

        var row = [];
        for (var j = 0; j < 21; j++) {
            var p = rel[j];
            row.push(
                p[0]*right[0]   + p[1]*right[1]   + p[2]*right[2],
                p[0]*up[0]      + p[1]*up[1]      + p[2]*up[2],
                p[0]*forward[0] + p[1]*forward[1] + p[2]*forward[2]
            );
        }
        out.push({ orig_idx: i, coords: row });
    }
    return out;
}
"""


def _build_js_context() -> py_mini_racer.MiniRacer:
    ctx = py_mini_racer.MiniRacer()
    ctx.eval(_JS_PREPROCESSOR_PATH.read_text(encoding="utf-8"))
    ctx.eval(_JS_HELPER)
    return ctx


def normalise_film(ctx: py_mini_racer.MiniRacer, hand_film: dict) -> list[tuple[int, np.ndarray]]:
    """Return [(orig_frame_idx, coords63_float32), ...] for every non-absent frame."""
    raw = ctx.call("inViewNormalisedCoords", hand_film)
    return [(int(r["orig_idx"]), np.asarray(r["coords"], dtype=np.float32)) for r in raw]


# ---------------------------------------------------------------------------
# Motion energy, smoothing, hold detection.
# ---------------------------------------------------------------------------

def split_into_segments(
    normalised: list[tuple[int, np.ndarray]],
) -> list[list[tuple[int, np.ndarray]]]:
    """Split into runs of consecutive-in-time in-view frames (absent frames break the run)."""
    segments: list[list[tuple[int, np.ndarray]]] = []
    if not normalised:
        return segments
    current = [normalised[0]]
    for prev, cur in zip(normalised, normalised[1:]):
        if cur[0] - prev[0] == 1:
            current.append(cur)
        else:
            segments.append(current)
            current = [cur]
    segments.append(current)
    return segments


def segment_energy(segment: list[tuple[int, np.ndarray]]) -> np.ndarray:
    """Per-frame L2 motion energy inside a segment.
    Returns an array of length len(segment)-1 (one delta per consecutive pair).
    """
    if len(segment) < 2:
        return np.zeros(0, dtype=np.float32)
    coords = np.stack([s[1] for s in segment])          # (n, 63)
    delta = np.diff(coords, axis=0)                      # (n-1, 63)
    return np.linalg.norm(delta, axis=1).astype(np.float32)


def smooth(arr: np.ndarray, k: int = 3) -> np.ndarray:
    """Centered moving average with edge padding. k must be odd and >= 1."""
    if k <= 1 or arr.size < k:
        return arr.astype(np.float32)
    pad = k // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def find_holds(smoothed: np.ndarray, t_hold: float, k_hold: int) -> list[dict]:
    """Runs where smoothed < t_hold for at least k_hold consecutive samples.
    Returns list of dicts with start, end (exclusive), length, rep (argmin index).
    Indices are into the smoothed/energy array; to map to a coord-frame index
    inside the segment, use `start` (energy[i] is the delta leading to frame i+1,
    so the hold covers frames [start..end] in coord-frame indexing).
    """
    holds = []
    below = smoothed < t_hold
    i, n = 0, len(below)
    while i < n:
        if below[i]:
            j = i
            while j < n and below[j]:
                j += 1
            if j - i >= k_hold:
                rep = i + int(np.argmin(smoothed[i:j]))
                holds.append({"start": i, "end": j, "length": j - i, "rep": rep})
            i = j
        else:
            i += 1
    return holds


# ---------------------------------------------------------------------------
# Per-film energy computation (independent of T_hold — done once, reused by
# every threshold in the sweep).
# ---------------------------------------------------------------------------

def compute_film_energy(
    ctx: py_mini_racer.MiniRacer,
    hand_film: dict,
    smooth_k: int,
) -> dict:
    """Normalise, segment, and compute smoothed energy for one film.
    Output is reused for every T_hold value we evaluate.
    """
    normalised = normalise_film(ctx, hand_film)
    segments = split_into_segments(normalised)
    seg_energies = [segment_energy(seg) for seg in segments]
    seg_smoothed = [smooth(e, k=smooth_k) for e in seg_energies]
    # Flat array of all energy deltas (for pooled percentile stats).
    flat_energy = (
        np.concatenate(seg_energies) if seg_energies else np.zeros(0, dtype=np.float32)
    )
    return {
        "normalised": normalised,
        "n_in_view": len(normalised),
        "segments": segments,
        "seg_energies": seg_energies,
        "seg_smoothed": seg_smoothed,
        "flat_energy": flat_energy,
    }


# ---------------------------------------------------------------------------
# Film-level hold detection. Pure function of (pre-computed energy, T_hold).
# ---------------------------------------------------------------------------

def analyse_film(
    film_energy: dict,
    gesture_id: str,
    session_id: str | None,
    t_hold: float,
    k_hold: int,
    edge_trim_fraction: float,
) -> dict:
    n_in_view = film_energy["n_in_view"]
    segments = film_energy["segments"]
    seg_smoothed = film_energy["seg_smoothed"]

    # Edge zone = first or last edge_trim_fraction of in-view frames.
    edge_lo = int(math.floor(n_in_view * edge_trim_fraction))
    edge_hi = int(math.ceil(n_in_view * (1.0 - edge_trim_fraction)))

    holds_all: list[dict] = []
    holds_kept: list[dict] = []
    hold_reps_normalised: list[np.ndarray] = []
    in_view_counter_offset = 0

    for seg, smoothed in zip(segments, seg_smoothed):
        seg_holds = find_holds(smoothed, t_hold=t_hold, k_hold=k_hold)
        for h in seg_holds:
            rep_coord_frame = h["rep"]
            in_view_ordinal = in_view_counter_offset + rep_coord_frame
            is_edge = (in_view_ordinal < edge_lo) or (in_view_ordinal >= edge_hi)
            h_out = {
                "in_view_ordinal": in_view_ordinal,
                "length": h["length"],
                "is_edge": is_edge,
                "position_fraction": (
                    in_view_ordinal / n_in_view if n_in_view > 0 else 0.0
                ),
            }
            holds_all.append(h_out)
            if not is_edge:
                holds_kept.append(h_out)
                hold_reps_normalised.append(seg[rep_coord_frame][1])
        in_view_counter_offset += len(seg)

    return {
        "gesture_id": gesture_id,
        "session_id": session_id,
        "n_in_view_frames": n_in_view,
        "n_segments": len(segments),
        "holds_all": holds_all,
        "holds_kept": holds_kept,
        "hold_reps": hold_reps_normalised,
    }


# ---------------------------------------------------------------------------
# Extra diagnostics: intra-film hold-pair distance, per-gesture centroids,
# hold position distribution, duplicate-frame fraction.
# ---------------------------------------------------------------------------

def intra_film_hold_pair_distances(rows: list[dict]) -> list[float]:
    """Distances between consecutive hold representatives *within the same film*.
    Tells us whether multi-hold films contain genuinely different poses (large
    distance) or duplicate/noisy holds (small distance).
    """
    distances: list[float] = []
    for r in rows:
        reps = r["hold_reps"]
        if len(reps) < 2:
            continue
        for i in range(len(reps) - 1):
            distances.append(float(np.linalg.norm(reps[i] - reps[i + 1])))
    return distances


def per_gesture_centroids(rows: list[dict]) -> dict[str, np.ndarray]:
    """Mean hold-rep coord per gesture (over all kept holds in all films)."""
    by_gesture: dict[str, list[np.ndarray]] = defaultdict(list)
    for r in rows:
        by_gesture[r["gesture_id"]].extend(r["hold_reps"])
    return {
        gid: np.mean(np.stack(reps), axis=0)
        for gid, reps in by_gesture.items()
        if reps
    }


def inter_gesture_distance_matrix(
    centroids: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray]:
    """Pairwise L2 distance between per-gesture centroids. Returns (labels, matrix)."""
    labels = sorted(centroids.keys())
    n = len(labels)
    mat = np.zeros((n, n), dtype=np.float32)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            mat[i, j] = float(np.linalg.norm(centroids[a] - centroids[b]))
    return labels, mat


def hold_position_histogram(rows: list[dict], n_bins: int = 4) -> list[int]:
    """Distribution of hold positions within the film, binned into n_bins quantiles."""
    bins = [0] * n_bins
    for r in rows:
        for h in r["holds_kept"]:
            idx = min(int(h["position_fraction"] * n_bins), n_bins - 1)
            bins[idx] += 1
    return bins


def hold_position_by_ordinal(rows: list[dict]) -> dict[int, list[float]]:
    """Group position_fraction values by the hold's ordinal index within its film.
    Returns {0: [positions of 1st hold in each film], 1: [...2nd...], ...}.
    Used to test whether the first hold in a film tends to land near the start
    (the "hand-entering-view" hypothesis).
    """
    by_ordinal: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        for i, h in enumerate(r["holds_kept"]):
            by_ordinal[i].append(float(h["position_fraction"]))
    return by_ordinal


def duplicate_frame_fraction(film_energies: list[dict]) -> float:
    """Fraction of consecutive frame pairs with exactly zero energy (very likely
    MediaPipe re-emitting the prior detection). Reported for diagnostics only —
    these samples are excluded from the T_hold percentile calc so the auto
    threshold isn't pinned to zero.
    """
    total = 0
    zeros = 0
    for fe in film_energies:
        for energy in fe["seg_energies"]:
            total += energy.size
            zeros += int(np.count_nonzero(energy == 0.0))
    return (zeros / total) if total else 0.0


# ---------------------------------------------------------------------------
# Clustering: agglomerative over all kept hold representatives.
# ---------------------------------------------------------------------------

def flatten_hold_reps(
    rows: list[dict],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Flatten every film's hold reps into one (N, 63) array.
    Returns (X, rep_to_film) where rep_to_film[k] = (film_index, hold_index).
    """
    flat: list[np.ndarray] = []
    rep_to_film: list[tuple[int, int]] = []
    for fi, r in enumerate(rows):
        for hi, rep in enumerate(r["hold_reps"]):
            flat.append(rep)
            rep_to_film.append((fi, hi))
    if not flat:
        return np.zeros((0, 0), dtype=np.float32), rep_to_film
    return np.stack(flat).astype(np.float32), rep_to_film


def cluster_hold_reps(X: np.ndarray, epsilon: float) -> np.ndarray:
    """Agglomerative clustering with distance threshold = epsilon.
    Average-link over Euclidean distance; no pre-set cluster count.
    Returns an int label per row.
    """
    if X.shape[0] < 2:
        return np.zeros(X.shape[0], dtype=np.int32)
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=epsilon,
        linkage="average",
    )
    return clusterer.fit_predict(X).astype(np.int32)


def film_templates_from_labels(
    n_films: int,
    labels: np.ndarray,
    rep_to_film: list[tuple[int, int]],
) -> list[list[int]]:
    """Per film, the ordered sequence of cluster ids across its kept holds."""
    templates: list[list[int]] = [[] for _ in range(n_films)]
    for label, (fi, _hi) in zip(labels.tolist(), rep_to_film):
        templates[fi].append(int(label))
    return templates


def template_consistency_per_gesture(
    rows: list[dict],
    film_templates: list[list[int]],
) -> dict[str, dict]:
    """For each gesture, the distribution of observed (ordered) cluster-id
    sequences across its films. Includes the modal template and what fraction
    of films produce it.
    """
    by_gesture: dict[str, list[tuple[int, ...]]] = defaultdict(list)
    for r, t in zip(rows, film_templates):
        if t:
            by_gesture[r["gesture_id"]].append(tuple(t))

    out: dict[str, dict] = {}
    for gid, seqs in by_gesture.items():
        counter = Counter(seqs)
        modal_seq, modal_n = counter.most_common(1)[0]
        out[gid] = {
            "n_films": len(seqs),
            "unique_templates": len(counter),
            "modal_template": list(modal_seq),
            "modal_fraction": modal_n / len(seqs),
            "all_templates": [
                {"template": list(k), "count": v}
                for k, v in counter.most_common()
            ],
        }
    return out


def cluster_composition(
    rows: list[dict],
    labels: np.ndarray,
    rep_to_film: list[tuple[int, int]],
) -> dict[int, dict]:
    """For each cluster id, the count of holds from each gesture plus total."""
    composition: dict[int, Counter] = defaultdict(Counter)
    for label, (fi, _hi) in zip(labels.tolist(), rep_to_film):
        composition[int(label)][rows[fi]["gesture_id"]] += 1
    return {
        cid: {
            "total": sum(cnt.values()),
            "by_gesture": dict(cnt),
        }
        for cid, cnt in composition.items()
    }


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------

def describe_percentiles(values: np.ndarray, pcts=(1, 5, 10, 25, 50, 75, 90, 95, 99)) -> dict:
    if values.size == 0:
        return {f"p{p}": None for p in pcts}
    return {f"p{p}": float(np.percentile(values, p)) for p in pcts}


def group_by_gesture(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["gesture_id"]].append(r)
    return out


def intra_gesture_spread(rows: list[dict]) -> float | None:
    """For each gesture with >=2 films AND >=2 holds total, compute pairwise distances
    among its hold representatives and return the 95th percentile across all gestures.
    This drives the ε threshold for agglomerative clustering.
    """
    all_distances: list[float] = []
    for gid, gesture_rows in group_by_gesture(rows).items():
        reps = [r for row in gesture_rows for r in row["hold_reps"]]
        if len(reps) < 2:
            continue
        arr = np.stack(reps)  # (m, 63)
        # Compute all pairwise distances (flat).
        m = arr.shape[0]
        if m > 200:
            # cap cost for very popular gestures
            idx = np.random.default_rng(0).choice(m, size=200, replace=False)
            arr = arr[idx]
            m = 200
        for i in range(m):
            for j in range(i + 1, m):
                all_distances.append(float(np.linalg.norm(arr[i] - arr[j])))
    if not all_distances:
        return None
    return float(np.percentile(np.array(all_distances), 95))


def per_gesture_hold_stats(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for gid, gesture_rows in group_by_gesture(rows).items():
        counts = np.array([len(r["holds_kept"]) for r in gesture_rows])
        out[gid] = {
            "n_films": int(counts.size),
            "mean_holds": float(counts.mean()),
            "median_holds": float(np.median(counts)),
            "min_holds": int(counts.min()),
            "max_holds": int(counts.max()),
            "zero_hold_pct": float((counts == 0).mean() * 100.0),
        }
    return out


def print_report(
    rows: list[dict],
    params: dict,
    pooled_energy: np.ndarray,
    nonzero_energy: np.ndarray,
    dup_frac: float,
    sweep_results: list[dict],
    report_json: dict,
) -> None:
    print("=" * 72)
    print("Key-pose calibration report")
    print("=" * 72)

    # --- Inventory ---
    gestures = group_by_gesture(rows)
    sessions = {r["session_id"] for r in rows if r["session_id"] is not None}
    print()
    print(f"Films analysed   : {len(rows)}")
    print(f"Distinct gestures: {len(gestures)}")
    print(f"Distinct sessions: {len(sessions)}")
    print("Per-gesture film counts:")
    for gid in sorted(gestures):
        print(f"  {gid:30s}  {len(gestures[gid]):4d}")

    # --- Duplicate-frame diagnostic ---
    print()
    print(
        f"Consecutive frame pairs with exactly zero energy: "
        f"{dup_frac * 100:.1f}% of total deltas"
    )
    print(
        "  (Likely MediaPipe re-emitting the previous detection. These samples "
        "are excluded from\n   the non-zero percentile stats below so the "
        "auto T_hold isn't pinned to zero.)"
    )
    report_json["duplicate_frame_fraction"] = dup_frac

    # --- Motion energy distribution ---
    print()
    print("Per-frame motion energy, normalised coord space (all pairs pooled):")
    pct_all = describe_percentiles(pooled_energy)
    pct_nz = describe_percentiles(nonzero_energy)
    keys = ("p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99")
    print(f"  {'percentile':>10s}  {'all':>10s}  {'non-zero':>10s}")
    for k in keys:
        va = pct_all.get(k)
        vn = pct_nz.get(k)
        sa = f"{va:.5f}" if va is not None else "n/a"
        sn = f"{vn:.5f}" if vn is not None else "n/a"
        print(f"  {k:>10s}  {sa:>10s}  {sn:>10s}")
    report_json["energy_percentiles_all"] = pct_all
    report_json["energy_percentiles_nonzero"] = pct_nz

    # --- T_hold sweep ---
    print()
    print(f"T_hold sweep (K_hold={params['k_hold']}, smooth={params['smooth_k']}, "
          f"edge_trim={int(params['edge_trim_fraction']*100)}%):")
    sweep_header = "  " + f"{'T_hold':>8s}  " + "".join(
        f"{gid[:14]:>15s}" for gid in sorted(gestures)
    ) + "  pooled_median"
    print(sweep_header)
    sweep_summary: list[dict] = []
    for sr in sweep_results:
        gesture_medians = [
            sr["per_gesture"].get(gid, {}).get("median_holds", 0.0)
            for gid in sorted(gestures)
        ]
        pooled_med = np.median([
            len(r["holds_kept"]) for r in sr["rows"]
        ])
        row_str = "  " + f"{sr['t_hold']:>8.4f}  " + "".join(
            f"{m:>15.1f}" for m in gesture_medians
        ) + f"  {pooled_med:>12.1f}"
        print(row_str)
        sweep_summary.append({
            "t_hold": sr["t_hold"],
            "per_gesture_median": dict(zip(sorted(gestures), gesture_medians)),
            "pooled_median": float(pooled_med),
        })
    report_json["t_hold_sweep"] = sweep_summary

    # --- Chosen-threshold detail ---
    print()
    print(
        f"Detail at chosen T_hold={params['t_hold']:.5f}:"
    )
    print(f"  {'gesture':30s}  {'n':>4s}  {'mean':>5s}  {'med':>4s}  "
          f"{'min':>4s}  {'max':>4s}  {'zero%':>6s}")
    gesture_stats = per_gesture_hold_stats(rows)
    for gid in sorted(gestures):
        s = gesture_stats[gid]
        print(
            f"  {gid:30s}  {s['n_films']:4d}  "
            f"{s['mean_holds']:5.2f}  {s['median_holds']:4.1f}  "
            f"{s['min_holds']:4d}  {s['max_holds']:4d}  "
            f"{s['zero_hold_pct']:5.0f}%"
        )
    report_json["per_gesture_hold_stats"] = gesture_stats

    # --- Intra-film hold-pair distance (the key artefact diagnostic) ---
    pair_dists = intra_film_hold_pair_distances(rows)
    print()
    if pair_dists:
        arr = np.array(pair_dists, dtype=np.float32)
        print(f"Intra-film hold-pair distance (n={arr.size}):")
        for k in ("p10", "p25", "p50", "p75", "p90"):
            p = int(k[1:])
            print(f"  {k:>4s}  {np.percentile(arr, p):.4f}")
        print(
            "  Interpretation: if the median is small (< a few coord units), the "
            "2nd hold in\n  a film is basically the same pose as the 1st — likely "
            "the hand-entering-view\n  artefact. If the median is large, the "
            "holds are genuinely different poses."
        )
        report_json["intra_film_hold_pair_percentiles"] = {
            k: float(np.percentile(arr, int(k[1:]))) for k in ("p10", "p25", "p50", "p75", "p90")
        }
    else:
        print("Intra-film hold-pair distance: n/a (no films with ≥2 holds)")
        report_json["intra_film_hold_pair_percentiles"] = None

    # --- Hold position in the film ---
    print()
    bins = hold_position_histogram(rows, n_bins=4)
    total_holds = sum(bins) or 1
    print("Hold position in film (quartiles of in-view duration):")
    quartile_labels = ("Q1 [0-25%)", "Q2 [25-50%)", "Q3 [50-75%)", "Q4 [75-100%]")
    for label, count in zip(quartile_labels, bins):
        print(f"  {label:>14s}  {count:5d}  ({100 * count / total_holds:5.1f}%)")
    report_json["hold_position_quartiles"] = dict(zip(quartile_labels, bins))

    # --- Position by hold ordinal (tests the "hand entering view" hypothesis) ---
    print()
    by_ord = hold_position_by_ordinal(rows)
    print("Hold position by ordinal (which hold in the film, vs its position):")
    print(f"  {'ordinal':>8s}  {'n':>4s}  {'mean':>6s}  {'med':>6s}  {'p25':>6s}  {'p75':>6s}")
    ordinal_stats: dict[str, dict] = {}
    for ord_i in sorted(by_ord):
        positions = np.array(by_ord[ord_i], dtype=np.float32)
        if positions.size == 0:
            continue
        label = f"hold#{ord_i + 1}"
        s = {
            "n": int(positions.size),
            "mean_position": float(positions.mean()),
            "median_position": float(np.median(positions)),
            "p25": float(np.percentile(positions, 25)),
            "p75": float(np.percentile(positions, 75)),
        }
        ordinal_stats[label] = s
        print(
            f"  {label:>8s}  {s['n']:4d}  {s['mean_position']:6.3f}  "
            f"{s['median_position']:6.3f}  {s['p25']:6.3f}  {s['p75']:6.3f}"
        )
    print(
        "  Interpretation: if hold#1's median is near 0.15-0.3, it's landing just "
        "after the\n  edge-trim at the film start — consistent with a "
        "'hand-entering-view' artefact.\n  If hold#1's median sits mid-film, the "
        "first hold is a real gesture phase."
    )
    report_json["hold_position_by_ordinal"] = ordinal_stats

    # --- Inter-gesture centroid distances ---
    centroids = per_gesture_centroids(rows)
    print()
    if len(centroids) >= 2:
        labels, mat = inter_gesture_distance_matrix(centroids)
        print("Inter-gesture centroid distances (hold-rep coord space):")
        header = " " * 18 + "".join(f"{lbl[:14]:>15s}" for lbl in labels)
        print(header)
        for i, a in enumerate(labels):
            row = f"  {a[:16]:16s}" + "".join(f"{mat[i, j]:>15.4f}" for j in range(len(labels)))
            print(row)
        # Off-diagonal min gives the closest gesture pair.
        off = mat.copy()
        np.fill_diagonal(off, np.inf)
        min_idx = int(np.argmin(off))
        i_min, j_min = divmod(min_idx, mat.shape[1])
        print(
            f"\n  Closest pair: {labels[i_min]} ↔ {labels[j_min]}  "
            f"distance={mat[i_min, j_min]:.4f}"
        )
        report_json["inter_gesture_distance_min"] = float(mat[i_min, j_min])
        report_json["inter_gesture_closest_pair"] = [labels[i_min], labels[j_min]]
    else:
        print("Inter-gesture centroid distances: n/a (need ≥2 gestures with holds)")

    # --- ε for agglomerative clustering ---
    eps = intra_gesture_spread(rows)
    print()
    if eps is None:
        print("ε (agglomerative clustering threshold): n/a — not enough holds per gesture")
    else:
        print(f"ε suggestion (95th pct of within-gesture hold-rep pairwise distance): {eps:.4f}")
    report_json["epsilon_suggestion"] = eps

    # --- Template length draft ---
    print()
    print("Suggested gesture → template length (median hold count at chosen T_hold):")
    templates_draft: dict[str, int] = {}
    for gid, s in gesture_stats.items():
        tlen = int(round(s["median_holds"]))
        templates_draft[gid] = tlen
        tag = "  (no-pose-gate)" if tlen == 0 else ""
        print(f"  {gid:30s}  len={tlen}{tag}")
    report_json["template_length_draft"] = templates_draft

    # --- Edge-trim effect ---
    n_all_holds = sum(len(r["holds_all"]) for r in rows)
    n_kept = sum(len(r["holds_kept"]) for r in rows)
    trimmed_pct = 100.0 * (1.0 - n_kept / n_all_holds) if n_all_holds else 0.0
    print()
    print(
        f"Edge trim impact: {n_all_holds} total holds detected, "
        f"{n_kept} kept after trimming first/last "
        f"{int(params['edge_trim_fraction']*100)}%  "
        f"({trimmed_pct:.1f}% dropped)"
    )
    report_json["edge_trim"] = {
        "total_holds": n_all_holds,
        "kept_holds": n_kept,
        "dropped_pct": trimmed_pct,
    }

    # --- Clustering: agglomerative over all kept hold reps ---
    X, rep_to_film = flatten_hold_reps(rows)
    cluster_reports: list[dict] = []
    if X.shape[0] >= 2 and params.get("epsilon_grid"):
        print()
        print("Clustering (agglomerative, average-link, Euclidean):")
        for eps_val in params["epsilon_grid"]:
            labels = cluster_hold_reps(X, eps_val)
            n_clusters = int(labels.max()) + 1 if labels.size else 0
            templates = film_templates_from_labels(len(rows), labels, rep_to_film)
            consistency = template_consistency_per_gesture(rows, templates)
            composition = cluster_composition(rows, labels, rep_to_film)

            # Cluster size distribution (sorted desc).
            sizes = sorted(
                (c["total"] for c in composition.values()), reverse=True
            )
            print()
            print(f"  ε = {eps_val:.3f}   →   {n_clusters} clusters   "
                  f"(sizes top-10: {sizes[:10]})")

            # Per-gesture template consistency.
            print(f"  {'gesture':>12s}  {'films':>6s}  {'uniq':>5s}  "
                  f"{'modal %':>8s}  modal template")
            for gid in sorted(consistency):
                s = consistency[gid]
                print(
                    f"  {gid:>12s}  {s['n_films']:>6d}  {s['unique_templates']:>5d}  "
                    f"{s['modal_fraction']*100:>7.1f}%  {s['modal_template']}"
                )

            # Top clusters by size: which gestures contribute?
            top = sorted(composition.items(), key=lambda kv: -kv[1]["total"])[:8]
            print("  cluster_id → gesture composition (top 8 by size):")
            for cid, comp in top:
                gesture_parts = ", ".join(
                    f"{g}:{n}" for g, n in sorted(
                        comp["by_gesture"].items(), key=lambda kv: -kv[1]
                    )
                )
                print(f"    [{cid:>3d}]  n={comp['total']:>3d}   {gesture_parts}")

            cluster_reports.append({
                "epsilon": float(eps_val),
                "n_clusters": n_clusters,
                "cluster_sizes": sizes,
                "per_gesture_consistency": {
                    gid: {
                        "n_films": s["n_films"],
                        "unique_templates": s["unique_templates"],
                        "modal_fraction": s["modal_fraction"],
                        "modal_template": s["modal_template"],
                        "all_templates": s["all_templates"],
                    }
                    for gid, s in consistency.items()
                },
                "cluster_composition": {
                    str(cid): comp for cid, comp in composition.items()
                },
            })
    else:
        print()
        print("Clustering skipped (not enough hold reps or no epsilon grid set).")
    report_json["clustering"] = cluster_reports

    # --- Final suggested parameters ---
    # T_hold based on NON-ZERO percentiles (excludes MediaPipe duplicates).
    p25_nz = pct_nz.get("p25") or 0.0
    p95_nz = pct_nz.get("p95") or 0.0
    t_hold_suggestion = max(1.5 * p25_nz, 0.5 * p95_nz)
    print()
    print("-" * 72)
    print("Suggested starting values for Phase 2:")
    print(f"  T_hold          = {t_hold_suggestion:.5f}   "
          f"(currently tested: {params['t_hold']:.5f})")
    print(f"  K_hold          = 3 frames")
    print(f"  smooth_k        = 3 frames")
    print(f"  edge_trim       = 15%")
    print(f"  T_commit        = 300 ms  (runtime, after gate-close)")
    print(f"  ε (cluster)     = {eps:.4f}" if eps is not None else "  ε (cluster)     = (need more data)")
    print("-" * 72)
    report_json["suggested_parameters"] = {
        "t_hold": t_hold_suggestion,
        "k_hold": 3,
        "smooth_k": 3,
        "edge_trim_fraction": 0.15,
        "t_commit_ms": 300,
        "epsilon": eps,
    }


# ---------------------------------------------------------------------------
# Data loading (real DB and synthetic for --dry-run).
# ---------------------------------------------------------------------------

async def load_examples_from_db() -> list[dict]:
    """Load every training example from the server's PostgreSQL database.
    Requires the server's .env to be present so config.Settings can initialise.
    """
    from storage.example_store import load_all_examples  # imported lazily on purpose
    return await load_all_examples()


def generate_synthetic_examples() -> list[dict]:
    """Tiny synthetic dataset for --dry-run. Five gestures × 8 films each.

    Each film is 30 frames. Gestures have different hold structure:
      static_fist        : 1 long hold throughout
      open_then_close    : 2 holds with a transition
      wave_dynamic       : no holds — continuous motion
      three_stage        : 3 holds
      flick_plus_hold    : 1 hold after a motion burst
    """
    rng = random.Random(42)

    def frame(coords_63: list[float], idx: int, is_absent: bool = False) -> dict:
        landmarks = [
            {"x": coords_63[i * 3], "y": coords_63[i * 3 + 1], "z": coords_63[i * 3 + 2]}
            for i in range(21)
        ]
        return {
            "landmarks": landmarks,
            "timestamp": float(idx) * (1.0 / 30.0),
            "left_or_right": "right",
            "is_absent": is_absent,
        }

    def pose_coords(pose_id: int, jitter: float) -> list[float]:
        # Deterministic per-pose coords in raw landmark space; wrist at origin
        # with middle_MCP on +y axis so normalisation stays well-defined.
        base = []
        wrist = (0.5, 0.5, 0.0)
        middle_mcp = (0.5, 0.35, 0.0)
        # 21 landmarks; seed offsets per pose_id for variety.
        prng = random.Random(pose_id * 1000 + 7)
        for j in range(21):
            if j == 0:
                x, y, z = wrist
            elif j == 9:
                x, y, z = middle_mcp
            else:
                angle = (j / 21.0) * math.pi + pose_id * 0.7
                r = 0.1 + 0.01 * prng.random() + pose_id * 0.01
                x = 0.5 + math.cos(angle) * r
                y = 0.5 - math.sin(angle) * r
                z = (prng.random() - 0.5) * 0.02
            x += rng.gauss(0, jitter)
            y += rng.gauss(0, jitter)
            z += rng.gauss(0, jitter)
            base.extend([x, y, z])
        return base

    def film_for(schedule: list[tuple[int, int]]) -> dict:
        # schedule: list of (pose_id, n_frames). Holds use tiny jitter,
        # motion segments interpolate between poses with medium jitter.
        frames = []
        idx = 0
        for i, (pose_id, n) in enumerate(schedule):
            if i == 0 or schedule[i - 1][0] == pose_id:
                # static hold on this pose
                for _ in range(n):
                    frames.append(frame(pose_coords(pose_id, jitter=0.001), idx))
                    idx += 1
            else:
                # motion segment: interpolate from prev pose to this pose
                prev_pose = schedule[i - 1][0]
                a = pose_coords(prev_pose, jitter=0.0)
                b = pose_coords(pose_id,  jitter=0.0)
                for k in range(n):
                    t = (k + 1) / n
                    interp = [
                        (1 - t) * a[m] + t * b[m] + rng.gauss(0, 0.005)
                        for m in range(63)
                    ]
                    frames.append(frame(interp, idx))
                    idx += 1
        return {"frames": frames, "start_time": 0.0}

    examples: list[dict] = []
    recipes = {
        "static_fist":     [(1, 30)],
        "open_then_close": [(2, 8), (3, 6), (4, 10), (4, 6)],   # hold, move, move-tail, hold
        "wave_dynamic":    [(5, 10), (6, 10), (5, 10)],         # pure motion ABA, no real hold
        "three_stage":     [(7, 6), (8, 4), (8, 6), (9, 4), (9, 6), (7, 4)],
        "flick_plus_hold": [(10, 4), (11, 6), (12, 4), (12, 12)],
    }
    for gid, schedule in recipes.items():
        for s in range(8):
            examples.append({
                "id": f"synthetic-{gid}-{s}",
                "gesture_id": gid,
                "session_id": f"session-{s}",
                "user_id": "synthetic-user",
                "hand_film": film_for(schedule),
                "created_at": 0.0,
            })
    return examples


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

def run(examples: list[dict], args: argparse.Namespace) -> dict:
    ctx = _build_js_context()

    # Pass 1: compute per-film energy once. Every T_hold in the sweep reuses it.
    film_energies: list[dict] = []
    for ex in examples:
        film_energies.append(
            compute_film_energy(ctx, ex["hand_film"], smooth_k=args.smooth_k)
        )

    pooled_arr = np.concatenate([
        fe["flat_energy"] for fe in film_energies if fe["flat_energy"].size > 0
    ]) if film_energies else np.zeros(0, dtype=np.float32)
    if pooled_arr.size == 0:
        print("No energy samples — all films have zero usable in-view frames.", file=sys.stderr)
        sys.exit(2)

    nonzero_arr = pooled_arr[pooled_arr > 0.0]
    dup_frac = duplicate_frame_fraction(film_energies)

    # T_hold chosen from NON-ZERO percentiles to bypass the MediaPipe-duplicate floor.
    if nonzero_arr.size == 0:
        print("All energy samples are zero — check for pathological data.", file=sys.stderr)
        sys.exit(2)
    p25_nz = float(np.percentile(nonzero_arr, 25))
    p95_nz = float(np.percentile(nonzero_arr, 95))
    auto_t_hold = max(1.5 * p25_nz, 0.5 * p95_nz)
    t_hold = args.t_hold if args.t_hold is not None else auto_t_hold

    # T_hold grid for the sweep. Default grid is geometric around the auto value;
    # overridable with --t-hold-grid=0.08,0.15,0.3,0.6.
    if args.t_hold_grid:
        grid = [float(x) for x in args.t_hold_grid.split(",")]
    else:
        grid = sorted({
            round(auto_t_hold * f, 5)
            for f in (0.25, 0.5, 1.0, 2.0, 4.0)
        })
        # Always include the currently-chosen t_hold so its column is visible.
        if t_hold not in grid:
            grid = sorted(grid + [round(t_hold, 5)])

    # Epsilon grid for clustering. If --epsilon-grid is set it wins; otherwise
    # a single ε from --epsilon (default 0.8).
    if args.epsilon_grid:
        eps_grid = [float(x) for x in args.epsilon_grid.split(",")]
    else:
        eps_grid = [float(args.epsilon)]

    params = {
        "t_hold": t_hold,
        "k_hold": args.k_hold,
        "smooth_k": args.smooth_k,
        "edge_trim_fraction": args.edge_trim,
        "epsilon_grid": eps_grid,
    }

    def analyse_all(t: float) -> list[dict]:
        return [
            analyse_film(
                fe,
                ex["gesture_id"],
                ex.get("session_id"),
                t_hold=t,
                k_hold=args.k_hold,
                edge_trim_fraction=args.edge_trim,
            )
            for ex, fe in zip(examples, film_energies)
        ]

    # Pass 2: T_hold sweep.
    sweep_results: list[dict] = []
    for t in grid:
        sweep_rows = analyse_all(t)
        sweep_results.append({
            "t_hold": t,
            "rows": sweep_rows,
            "per_gesture": per_gesture_hold_stats(sweep_rows),
        })

    # Pass 3: detailed analysis at the chosen T_hold.
    rows = analyse_all(t_hold)

    report_json: dict[str, Any] = {"params_used": params}
    print_report(
        rows=rows,
        params=params,
        pooled_energy=pooled_arr,
        nonzero_energy=nonzero_arr,
        dup_frac=dup_frac,
        sweep_results=sweep_results,
        report_json=report_json,
    )
    return report_json


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
        "--t-hold", type=float, default=None,
        help="Override the motion-energy threshold. Default: derived from non-zero percentiles.",
    )
    parser.add_argument(
        "--t-hold-grid", type=str, default=None,
        help="Comma-separated T_hold values for the sweep. Default: auto-derived.",
    )
    parser.add_argument("--k-hold", type=int, default=3, help="Min consecutive frames for a hold.")
    parser.add_argument("--smooth-k", type=int, default=3, help="Moving-average window for energy.")
    parser.add_argument(
        "--edge-trim", type=float, default=0.15,
        help="Fraction of first/last in-view frames to exclude from hold acceptance.",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.8,
        help="Agglomerative-clustering distance threshold on hold reps. Default: 0.8.",
    )
    parser.add_argument(
        "--epsilon-grid", type=str, default=None,
        help="Comma-separated ε values for a clustering sweep. Overrides --epsilon.",
    )
    args = parser.parse_args()

    if args.dry_run:
        examples = generate_synthetic_examples()
        print(f"(dry-run) generated {len(examples)} synthetic examples", file=sys.stderr)
    else:
        examples = asyncio.run(load_examples_from_db())
        print(f"Loaded {len(examples)} examples from database", file=sys.stderr)

    if not examples:
        print("No examples to analyse.", file=sys.stderr)
        sys.exit(1)

    report = run(examples, args)

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"Wrote JSON report to {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
