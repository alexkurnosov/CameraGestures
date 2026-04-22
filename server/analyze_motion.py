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
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import py_mini_racer


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
# Film-level analysis. Produces per-film summary rows for all downstream stats.
# ---------------------------------------------------------------------------

def analyse_film(
    ctx: py_mini_racer.MiniRacer,
    hand_film: dict,
    gesture_id: str,
    session_id: str | None,
    t_hold: float,
    k_hold: int,
    smooth_k: int,
    edge_trim_fraction: float,
) -> dict:
    normalised = normalise_film(ctx, hand_film)
    segments = split_into_segments(normalised)

    all_energy: list[float] = []
    holds_in_view: list[dict] = []   # holds after edge-trimming (what we'd use for Phase 2)
    holds_all: list[dict] = []       # all holds, including edge ones (for trim-rate metric)
    n_in_view = len(normalised)

    # Compute in-view frame index ordering to evaluate edge trim.
    # "edge" = first or last edge_trim_fraction of in_view frames across the whole film.
    edge_lo = int(math.floor(n_in_view * edge_trim_fraction))
    edge_hi = int(math.ceil(n_in_view * (1.0 - edge_trim_fraction)))

    in_view_counter_offset = 0  # running count of in-view frames across prior segments
    hold_reps_normalised: list[np.ndarray] = []

    for seg in segments:
        energy = segment_energy(seg)
        all_energy.extend(energy.tolist())

        smoothed = smooth(energy, k=smooth_k)
        seg_holds = find_holds(smoothed, t_hold=t_hold, k_hold=k_hold)

        for h in seg_holds:
            # Map smoothed-index rep → coord-frame index inside segment → in-view ordinal.
            rep_coord_frame = h["rep"]           # 0..len(seg)-1
            in_view_ordinal = in_view_counter_offset + rep_coord_frame
            is_edge = (in_view_ordinal < edge_lo) or (in_view_ordinal >= edge_hi)
            h_out = {
                "in_view_ordinal": in_view_ordinal,
                "length": h["length"],
                "is_edge": is_edge,
            }
            holds_all.append(h_out)
            if not is_edge:
                holds_in_view.append(h_out)
                hold_reps_normalised.append(seg[rep_coord_frame][1])

        in_view_counter_offset += len(seg)

    return {
        "gesture_id": gesture_id,
        "session_id": session_id,
        "n_in_view_frames": n_in_view,
        "n_segments": len(segments),
        "energy": all_energy,
        "holds_all": holds_all,
        "holds_kept": holds_in_view,
        "hold_reps": hold_reps_normalised,   # for cross-film clustering
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


def print_report(
    rows: list[dict],
    params: dict,
    all_energy: np.ndarray,
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

    # --- Motion energy distribution ---
    print()
    print("Per-frame motion energy, normalised coord space (all films pooled):")
    pct = describe_percentiles(all_energy)
    for k in ("p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"):
        v = pct[k]
        print(f"  {k:>4s}  {v:.5f}" if v is not None else f"  {k:>4s}  n/a")
    report_json["energy_percentiles"] = pct

    # --- Hold-count distribution under the chosen params ---
    print()
    print(
        f"Holds per film (T_hold={params['t_hold']:.5f}, "
        f"K_hold={params['k_hold']}, smooth={params['smooth_k']}, "
        f"edge_trim={int(params['edge_trim_fraction']*100)}%)"
    )
    print(f"  {'gesture':30s}  {'n':>4s}  {'mean':>5s}  {'med':>4s}  {'min':>4s}  {'max':>4s}  {'zero%':>6s}")
    gesture_hold_stats: dict[str, dict] = {}
    for gid in sorted(gestures):
        counts = np.array([len(r["holds_kept"]) for r in gestures[gid]])
        stats = {
            "n_films": int(counts.size),
            "mean_holds": float(counts.mean()),
            "median_holds": float(np.median(counts)),
            "min_holds": int(counts.min()),
            "max_holds": int(counts.max()),
            "zero_hold_pct": float((counts == 0).mean() * 100.0),
        }
        gesture_hold_stats[gid] = stats
        print(
            f"  {gid:30s}  {stats['n_films']:4d}  "
            f"{stats['mean_holds']:5.2f}  {stats['median_holds']:4.1f}  "
            f"{stats['min_holds']:4d}  {stats['max_holds']:4d}  "
            f"{stats['zero_hold_pct']:5.0f}%"
        )
    report_json["per_gesture_hold_stats"] = gesture_hold_stats

    # --- Suggested template length per gesture ---
    print()
    print("Suggested gesture → template length (median hold count):")
    templates_draft: dict[str, int] = {}
    for gid, stats in gesture_hold_stats.items():
        tlen = int(round(stats["median_holds"]))
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

    # --- ε for agglomerative clustering ---
    eps = intra_gesture_spread(rows)
    print()
    if eps is None:
        print("ε (agglomerative clustering threshold): n/a — not enough holds per gesture")
    else:
        print(f"ε suggestion (95th pct of within-gesture hold-rep pairwise distance): {eps:.4f}")
    report_json["epsilon_suggestion"] = eps

    # --- Final suggested parameters ---
    print()
    print("-" * 72)
    print("Suggested starting values for Phase 2:")
    # Noise-floor-based T_hold: 1.5 × p95 of low-energy samples.
    # Use p25 of all-energy as the "low-energy" floor proxy (25% of samples
    # are in holds or near-holds under reasonable assumptions).
    p95 = pct.get("p95") or 0.0
    p25 = pct.get("p25") or 0.0
    t_hold_suggestion = max(1.5 * p25, 0.5 * p95)
    print(f"  T_hold          = {t_hold_suggestion:.5f}   (currently tested: {params['t_hold']:.5f})")
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

    # First pass: compute pooled motion energy to pick a sensible candidate T_hold.
    pooled: list[float] = []
    for ex in examples:
        normalised = normalise_film(ctx, ex["hand_film"])
        for seg in split_into_segments(normalised):
            pooled.extend(segment_energy(seg).tolist())
    pooled_arr = np.asarray(pooled, dtype=np.float32)
    if pooled_arr.size == 0:
        print("No energy samples — all films have zero usable in-view frames.", file=sys.stderr)
        sys.exit(2)

    p25 = float(np.percentile(pooled_arr, 25))
    p95 = float(np.percentile(pooled_arr, 95))
    # Candidate T_hold sits between the low-energy floor and median motion.
    t_hold = args.t_hold if args.t_hold is not None else max(1.5 * p25, 0.5 * p95)

    params = {
        "t_hold": t_hold,
        "k_hold": args.k_hold,
        "smooth_k": args.smooth_k,
        "edge_trim_fraction": args.edge_trim,
    }

    # Second pass: full analysis per film with the chosen params.
    rows = [
        analyse_film(
            ctx,
            ex["hand_film"],
            ex["gesture_id"],
            ex.get("session_id"),
            t_hold=params["t_hold"],
            k_hold=params["k_hold"],
            smooth_k=params["smooth_k"],
            edge_trim_fraction=params["edge_trim_fraction"],
        )
        for ex in examples
    ]

    report_json: dict[str, Any] = {"params_used": params}
    print_report(rows, params, pooled_arr, report_json)
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
        help="Override the motion-energy threshold. Default: derived from percentiles.",
    )
    parser.add_argument("--k-hold", type=int, default=3, help="Min consecutive frames for a hold.")
    parser.add_argument("--smooth-k", type=int, default=3, help="Moving-average window for energy.")
    parser.add_argument(
        "--edge-trim", type=float, default=0.15,
        help="Fraction of first/last in-view frames to exclude from hold acceptance.",
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
