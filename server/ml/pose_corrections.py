"""Pose corrections loader and params_hash utility (Stage 1.2 + 1.3).

pose_corrections.json records human reviewer decisions:
  cluster_kinds    — explicit kind per cluster id ("idle" / "regular" / "unconfirmed")
  excluded_holds   — per-hold exclusions with params_hash for migration

Clusters not listed in cluster_kinds are implicitly unconfirmed and excluded
from inference (rejected at runtime, excluded from training labels and templates).

The coarse_idle_fallback() implements the pre-heuristic used for gate calibration
when reviewer-confirmed idle clusters are not yet available.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# params_hash (Stage 1.3)
# ---------------------------------------------------------------------------

def params_hash(params: dict) -> str:
    """Canonical sha256 of {T_hold, K_hold, smooth_k, edge_trim_fraction}.

    Accepts both camelCase (T_hold / K_hold) and snake_case (t_hold / k_hold)
    key names. The output is invariant under dict key ordering.
    Returns a "sha256:<hex>" string.
    """
    t_hold = params.get("T_hold", params.get("t_hold"))
    k_hold = params.get("K_hold", params.get("k_hold"))
    smooth_k = params.get("smooth_k")
    edge_trim = params.get("edge_trim_fraction")
    if any(v is None for v in (t_hold, k_hold, smooth_k, edge_trim)):
        missing = [
            k for k, v in [
                ("T_hold", t_hold), ("K_hold", k_hold),
                ("smooth_k", smooth_k), ("edge_trim_fraction", edge_trim),
            ]
            if v is None
        ]
        raise ValueError(f"params_hash: missing keys {missing}")
    canonical = {
        "T_hold": float(t_hold),
        "K_hold": int(k_hold),
        "smooth_k": int(smooth_k),
        "edge_trim_fraction": float(edge_trim),
    }
    j = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(j.encode()).hexdigest()


# ---------------------------------------------------------------------------
# PoseCorrections (Stage 1.2)
# ---------------------------------------------------------------------------

class PoseCorrections:
    """In-memory representation of pose_corrections.json."""

    def __init__(self, cluster_kinds: dict, excluded_holds: list):
        # Normalise cluster_kinds keys to str so lookups work for both int
        # and str cluster ids.
        self.cluster_kinds: dict[str, str] = {str(k): v for k, v in cluster_kinds.items()}
        self.excluded_holds: list[dict] = excluded_holds

    # --- cluster_kinds lookup -----------------------------------------------

    def cluster_kind(self, cluster_id: int | str) -> str | None:
        """Explicit kind for this cluster, or None if not listed."""
        return self.cluster_kinds.get(str(cluster_id))

    def confirmed_idle_ids(self) -> frozenset[int]:
        """Cluster ids explicitly marked idle."""
        return frozenset(
            int(k) for k, v in self.cluster_kinds.items() if v == "idle"
        )

    # --- excluded_holds lookup -----------------------------------------------

    def is_hold_excluded(
        self, film_id: str, hold_ordinal: int, phash: str
    ) -> bool:
        """True iff the corrections file explicitly excludes this hold."""
        for entry in self.excluded_holds:
            if (
                entry.get("film_id") == film_id
                and entry.get("hold_ordinal") == hold_ordinal
                and entry.get("params_hash") == phash
            ):
                return True
        return False


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_corrections(path: Path | str) -> PoseCorrections:
    """Load pose_corrections.json.

    Returns an empty-but-valid PoseCorrections when the file does not exist.
    Raises ValueError with a descriptive message on malformed JSON or wrong
    top-level structure.
    """
    p = Path(path)
    if not p.exists():
        return PoseCorrections({}, [])

    text = p.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Malformed pose_corrections.json at {p}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"pose_corrections.json must be a JSON object, got {type(data).__name__}"
        )

    cluster_kinds = data.get("cluster_kinds", {})
    excluded_holds = data.get("excluded_holds", [])

    if not isinstance(cluster_kinds, dict):
        raise ValueError(
            "pose_corrections.json: 'cluster_kinds' must be a JSON object"
        )
    if not isinstance(excluded_holds, list):
        raise ValueError(
            "pose_corrections.json: 'excluded_holds' must be a JSON array"
        )

    return PoseCorrections(cluster_kinds, excluded_holds)


# ---------------------------------------------------------------------------
# Coarse idle-candidate fallback (Stage 1.2 / plan:130–133)
# ---------------------------------------------------------------------------

def coarse_idle_fallback(
    composition: dict[int, dict],
    min_gestures: int = 3,
    min_fraction: float = 0.05,
) -> frozenset[int]:
    """Identify idle-candidate clusters via the coarse pre-heuristic.

    A cluster qualifies if ≥ min_gestures distinct gestures each contribute
    ≥ min_fraction of the cluster's total holds.  Used for T_close gate
    calibration before reviewer-confirmed cluster_kinds are available.

    composition maps cluster_id → {"total": int, "by_gesture": {gid: count}}.
    """
    result: set[int] = set()
    for cid, comp in composition.items():
        total = comp.get("total", 0)
        if total <= 0:
            continue
        qualifying = sum(
            1
            for n in comp.get("by_gesture", {}).values()
            if n / total >= min_fraction
        )
        if qualifying >= min_gestures:
            result.add(int(cid))
    return frozenset(result)


def idle_clusters_for_calibration(
    corrections: PoseCorrections,
    composition: dict[int, dict],
    min_gestures: int = 3,
    min_fraction: float = 0.05,
) -> frozenset[int]:
    """Idle cluster ids to use for Phase 1 gate calibration.

    Uses confirmed idle clusters from corrections when available; falls back
    to coarse_idle_fallback otherwise.
    """
    confirmed = corrections.confirmed_idle_ids()
    if confirmed:
        return confirmed
    return coarse_idle_fallback(
        composition, min_gestures=min_gestures, min_fraction=min_fraction
    )
