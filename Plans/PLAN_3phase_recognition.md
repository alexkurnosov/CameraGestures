# 3-Phase Gesture Recognition — Design Plan

## Context

### Current state (as of 2026-04-25)

- Preprocessor filters absent frames before feature extraction (`nonAbsentFrames` in
  `preprocessor.js`). Result: false-positive rate dropped significantly.
- **Phase 3 padding-ratio skew (Problem A): solved.** `summaryFeatures` now divides
  coord/velocity stats by the real-frame count instead of `TARGET_FRAMES` (PR #34).
  A clip with 10 real frames and a clip with 60 real frames of the same gesture now
  produce comparable feature vectors.
- **Phase 3 runtime buffer: bumped 10 → 30 frames** (PR #37) after `evaluate_trimmed.py`
  showed real-gesture accuracy collapses below trim=20 (58 % at 20, 16 % at 10).
  30 frames at 30 fps = 1.0 s, matching mean training-clip length (~38 real frames)
  and the existing `temporalWindow`.
- Remaining structural problems:
  1. **Phase 3 length sensitivity below ~20 frames (Problem B)**: dynamic gestures
     (`ok`, `stop`, `thumbs_up`) lose discriminative signal when truncated. Static-pose
     gestures (`point_left`) survive aggressive trimming because their info lives in
     a single frame. Mitigations belong to Phase 1 or to feature/training changes —
     see Phase 3 below.
  2. **Per-frame prediction**: recognition fires on every camera frame (~30×/s), causing
     duplicate detections of the same gesture. Phase 1 cooldown will fix this.
  3. **No gesture boundary detection**: no concept of "gesture started / ended". Phase 1
     motion gate addresses this.

### Key design constraint (revised 2026-04-25)

**Dynamic gestures need ~30 frames (≈ 1 s) of motion to be classifiable.**
`evaluate_trimmed.py` on the 151-film corpus shows real-gesture accuracy is 100 %
at trim=30 and collapses below trim=20:

| trim | accuracy |
|---:|---:|
| 30 | 1.000 |
| 20 | 0.516 |
| 15 | 0.290 |
| 10 | 0.129 |

The earlier draft of this plan claimed gestures last 0.3–0.5 s and recommended
shrinking the training window to match. **That recommendation is refuted by the
data** and has been removed. Mean training clip = 38 real frames; runtime buffer
is now 30 frames. Both sit in the regime where the model classifies correctly.

---

## Architecture

```
Camera frames (30 fps)
        │
        ▼
┌───────────────────┐
│  Phase 1          │  per-frame, always-on
│  Motion Gate      │──── no motion → discard, keep watching
└───────────────────┘
        │ motion detected, buffer frames
        ▼
┌───────────────────┐
│  Phase 2          │  on each detected hold (low-energy run)
│  Key-Pose         │──── unknown pose → discard, reset gate
│  Classifier       │     else: extend observed sequence, prefix-match
└───────────────────┘      candidate gesture templates
        │ complete template match, candidate set S
        ▼
┌───────────────────┐
│  Phase 3          │  run once on the buffered handfilm
│  Handfilm Model   │──── low confidence → discard
│                   │     output restricted to S
└───────────────────┘
        │ confirmed gesture, class C ∈ S
        ▼
   fire callback  (cooldown ~1 s before next detection)
```

---

## Phase 1 — Motion Gate

### Purpose
Prevent Phases 2 and 3 from running on idle or slowly-drifting hands.

### Signal
Per-frame **motion energy**: sum of L2 distances of all 21 landmarks between
consecutive frames, in the wrist-relative normalised coordinate space already
computed by the preprocessor.

### Implementation options

| Option | Cost | Notes |
|---|---|---|
| **Threshold heuristic** | trivial | motion\_energy > T for K consecutive frames; T and K tuned empirically |
| Trained binary classifier | medium | would need negative examples (idle hands); likely overkill for this signal |

**Recommendation**: start with heuristic. Escalate to a trained model only if heuristic
produces too many spurious gates (tremor, slow movement).

### Hysteresis
Open the gate when energy exceeds `T_open` for `K_open` frames; close only when it falls
below `T_close < T_open` for `K_close` frames. Prevents flapping on noisy input.

### Threshold calibration from the existing corpus

All four thresholds can be seeded offline from the 151-film corpus by extending
`server/analyze_motion.py`. Run the script after the Phase 2 clustering pass
completes — it reuses the per-film energies, cluster labels, and idle-candidate
identification already computed for Phase 2.

Idle-pose detection (Phase 2) is now the primary commit signal at runtime, so
`T_close` / `K_close` serve as a *fallback* — relevant only when the user
neither releases into an idle pose nor triggers another hold. Seeding them
still matters (the gate needs *some* close condition for that path), but the
tolerances are looser than if the low-energy close were the main signal.

**Open side (`T_open`, `K_open`)** — from segment-onset energy.
- For each in-view segment, take the first N raw per-frame energies (default
  N=6 ≈ 200 ms), pool across the corpus, exclude the MediaPipe duplicate-frame
  zeros. `T_open` seed = p10 of the resulting non-zero distribution: the hand
  has just started moving and the gate should open promptly.
- `K_open` seed: distribution of frames from segment start until smoothed
  energy first crosses `T_open`. `K_open` ≤ p10 of that distribution so
  fast-onset captures aren't missed.

**Close side (`T_close`, `K_close`)** — from idle-candidate clusters.
The Phase 2 calibration (§Calibration observation) identified one cluster
(n=131 holds on the current corpus) that is "hand in view, relaxed, doing
nothing". Once a reviewer marks it `idle` via the inspector, the calibration
can read the confirmed set directly from `server/data/pose_corrections.json`
(`cluster_kinds`). Until that file exists, the calibration falls back to a
coarse pre-heuristic — clusters contributed to by ≥ 3 gestures with each
gesture accounting for ≥ 5 % of the cluster's holds — which approximates the
idle-pose signal well enough for seed values.
- `T_close` seed = p90 of within-hold smoothed energy pooled across all
  idle-candidate holds: idle frames fall below it 90 % of the time.
- `K_close` seed: compare idle-hold run-length vs gesture-specific-hold
  run-length distributions. If idle p25 > gesture p90, the populations
  separate cleanly on duration and `K_close` sits in the gap. If they
  overlap, duration alone can't distinguish them — rely on absent-hand
  and on Phase 2's idle-pose commit as the real close signals and set
  `K_close` conservatively.

Both gesture holds and idle holds are low-energy by construction, so energy
alone can't perfectly distinguish them; `K_close` relies on duration. The
calibration report flags whether the gap is clean.

If the idle-candidate data turns out to be too thin or the duration gap
overlaps, add a one-off "idle capture" mode to the training app (5–10 s films
of relaxed hand in view, flagged as idle) and re-run calibration.

### Open questions
- Should absent frames (no hand in view) reset the gate immediately, or tolerate short
  occlusions? Likely: reset immediately, since a missing hand cannot be "mid-gesture".
- **Gate-close vs hold-detect signal collision.** `T_close` and `T_hold` both
  gate on low energy; if `T_close ≥ T_hold` or `K_close ≤ K_hold`, every
  Phase 2 hold closes the gate. Mitigation: set `T_close` well below
  `T_hold` (calibration output seeds this) and make `K_close` longer than any
  realistic gesture hold, or rely on absent-hand as the primary close signal
  and treat the low-energy path as a fallback.
- Cooldown duration after a confirmed detection (architecture diagram says
  ~1 s) — not yet parameterised.
- Does Phase 1 share the motion-energy computation with Phase 2's runtime
  hold detector, or recompute? One source of truth is cheaper and avoids
  drift.

---

## Phase 2 — Key-Pose Classifier

### Purpose
Identify the stable hand configurations ("key poses") that make up a gesture and
narrow down the candidate class set for Phase 3. Each gesture is defined as an
**ordered sequence of pose ids** (its template). At inference, Phase 2 fires on
each detected hold, appends the predicted `pose_id` to an observed sequence, and
prefix-matches against the template manifest.

### Why key poses rather than a single initial pose

Gestures are naturally structured as *holds* separated by *movements* (the
Movement-Hold model from sign-language linguistics). Some gestures are truly
multi-phase (e.g. a hypothetical "shut up" = open palm → fingers together).
Calibration also shows that even static gestures produce a multi-hold pattern
in practice (see *Calibration observation* below). Templates as ordered lists
of pose ids handle both cases uniformly.

### Key-pose extraction pipeline (training)

For each stored film:
1. Filter out absent frames (already done by `preprocessor.js`).
2. Compute normalised coords per in-view frame (wrist-relative + scale-normalised
   + rotation-aligned — the first 63 dims of `featureMatrix`, without velocity or
   padding).
3. Split into in-view segments, breaking across absent-frame gaps.
4. Per segment, compute motion energy as the L2 norm of consecutive-frame
   coord deltas.
5. Smooth energy with a 3-frame moving average.
6. Detect **holds** as maximal runs where smoothed energy < `T_hold` and the
   run length is ≥ `K_hold`.
7. The hold's **representative frame** is the argmin of smoothed energy within
   the run.
8. Drop holds whose representative falls in the first or last 15 % of the
   in-view duration (enters/exits).

Pool all hold representatives across all films and cluster with **agglomerative
clustering** (average-link, Euclidean, `distance_threshold = ε`). Each cluster
is a `pose_id`.

Apply the **idle-pose heuristic** to flag candidates for human review (never
auto-exclude — flags surface in the inspector; clusters are only treated as
idle after explicit confirmation):

> Score each cluster with three signals:
> - *Gesture-entropy*: Shannon entropy of the cluster's gesture distribution,
>   normalised by `log(n_gestures)`. 0 = pure (gesture-specific), 1 = fully
>   shared.
> - *Tail-position*: the cluster appears as the last element in the modal
>   template of ≥ 2 distinct gestures.
> - *Temporal-position*: median `position_fraction` of the cluster's holds
>   > 0.5.
>
> A cluster is flagged **suspected_idle** when any two of the three signals
> trigger. The flag is advisory only.

**Human confirmation via inspector** (see *Inspection tool* section). Each
cluster has a `kind` of `regular`, `idle`, or `unconfirmed`. Unconfirmed
clusters default to `regular` behaviour at both training and inference. Only
clusters explicitly marked `idle` are treated as gesture-end markers.

For each gesture, the **canonical template** is the modal sequence across its
films, with any holds assigned to `idle`-marked clusters stripped out. Non-modal
films fall into one of: fewer holds than the modal, extra holds beyond the
modal, or a genuinely different pattern. If the modal fraction is ≥ 70 %, the
gesture template is the modal; otherwise flag the gesture for capture review.

### Feature vector

Each hold representative is a single frame's normalised coords — 63 floats
(21 landmarks × 3, wrist-relative + scale-normalised + rotation-aligned). The
classifier predicts the `pose_id` assigned to that frame by the clustering
pass.

### Runtime flow

```
motion gate OPENS (Phase 1, energy > T_open for K_open frames)
  │  buffer every in-view frame
  ▼
monitor motion energy continuously
  │
  on detected hold  (smoothed energy < T_hold for K_hold frames)
  ▼
Phase 2 MLP on the hold's argmin frame
  │
  ├─ top pose confidence < τ_commit                     → discard capture, reset gate
  │
  ├─ pose is IDLE  (cluster kind == "idle")
  │     observed == []                                  → reset gate, keep watching
  │     observed is a COMPLETE match of some template   → commit, restrict Phase 3 to S
  │     observed is a live prefix (no complete match)   → commit to the longest completed
  │                                                        ancestor prefix if any, else
  │                                                        discard and reset gate
  │
  └─ pose is REGULAR (unconfirmed clusters default here)
        append pose_id to observed = [p1, p2, ...]
        prefix-match observed against gesture_templates
        ├─ no template starts with observed             → discard capture, reset gate
        ├─ completion exists, no longer prefix possible → commit  ✓
        └─ completion exists but longer prefixes remain → keep buffering,
                                                           start T_commit timer
              new hold arrives before timer fires       → extend observed, re-match
              timer fires                               → commit the completed prefix

  on motion gate CLOSES (Phase 1, energy < T_close for K_close frames)
        if observed has any complete match              → commit  ✓
        else                                            → discard, reset gate
  ↓
Phase 3 runs on the buffered film, output restricted to S
```

Idle-pose detection is the **primary** commit signal for the happy path: the
user performs the gesture, relaxes into an idle configuration, commit fires
immediately. `T_commit` stays as a fallback for users who hold the gesture
without releasing. Gate-close is the final fallback for cases where neither
an idle pose nor another hold arrives.

### Parameters

Empirically chosen from the 151-film corpus (`ok`, `stop`, `point_left`,
`thumbs_up`) using `server/analyze_motion.py`:

| Parameter | Value | Rationale |
|---|---|---|
| `T_hold` | **0.54** | p75 of non-zero energy. `T_hold` sweep showed a flat plateau over 0.5–2.1 — threshold is not fragile. |
| `K_hold` | **3 frames** | ~100 ms at 30 fps; excludes jitter without missing short holds. |
| `smooth_k` | **3 frames** | Tight smoothing; preserves hold boundaries. |
| `edge_trim` | **15 %** | Drops 3.5 % of detected holds on the current corpus — removes enter/exit artefacts without undershooting. |
| `T_commit` | **300 ms** | After hold-close, wait this long for another hold before committing to a match. Handles prefix collisions (short templates that are prefixes of longer ones). |
| `ε` (clustering) | **0.8** | Below the inter-gesture centroid minimum (1.08 between `ok` and `stop`). Produces 29 clusters on the current corpus, with clean gesture-specific groupings for three of four gestures. |
| `τ_commit` (confidence) | **0.6** | Initial guess; tune after first real run. Holds with top pose confidence below this are rejected as "unknown pose". |
| `τ_idle_suspicion` | **entropy ≥ 0.8 · log(n_gestures)** | Part of the idle-pose heuristic (combined with tail-position and temporal-position signals). Flags clusters as `suspected_idle` for human review; never auto-excludes. |

Calibration uses non-zero energy percentiles because MediaPipe re-emits the
previous detection unchanged ~35 % of the time, which would pin any
percentile-based threshold to zero.

### Classifier

- **Architecture**: `Dense(64, relu) → Dense(32, relu) → Dense(n_pose_clusters, softmax)`
- **Input**: 63-dim hold representative (one frame of normalised coords).
- **Output space**: all pose clusters, *including idle*. The classifier must
  recognise idle poses because runtime uses them as gesture-end markers.
- **Output decision**: argmax over `pose_id` with confidence. If top
  confidence < `τ_commit`, the hold is rejected as "unknown pose".
- **Training labels**: `pose_id`s produced by the clustering pass (not
  `gesture_id`s directly). Idle-marked clusters are *included* in training.
  Holds explicitly excluded via the inspector's per-hold action are dropped.
- **Train/test split**: by `session_id` — never per-frame, which would leak
  near-duplicate holds from the same film across the split.

### Template manifest (on-disk format alongside the model)

```json
{
  "version": 1,
  "pose_clusters": {
    "2":  { "label": "pose_point",  "kind": "regular", "suspected_idle": false, "n_samples": 13, "centroid": [...63 floats...] },
    "9":  { "label": "pose_ok",     "kind": "regular", "suspected_idle": false, "n_samples": 45, "centroid": [...] },
    "10": { "label": "pose_stop",   "kind": "regular", "suspected_idle": false, "n_samples": 36, "centroid": [...] },
    "15": { "label": "pose_relax",  "kind": "idle",    "suspected_idle": true,  "n_samples": 131, "centroid": [...] }
  },
  "idle_poses": [15],
  "gesture_templates": {
    "ok":         [9],
    "stop":       [10],
    "point_left": [2]
  },
  "parameters": {
    "t_hold": 0.54, "k_hold_frames": 3, "smooth_k": 3,
    "edge_trim_fraction": 0.15, "t_commit_ms": 300,
    "epsilon": 0.8, "tau_commit": 0.6,
    "tau_idle_suspicion_entropy": 0.8
  }
}
```

`kind` is one of:
- `regular` — a gesture-specific pose; appears in templates and is used for prefix matching.
- `idle` — a neutral / end-of-gesture pose; never appears in templates; detection triggers commit at runtime.
- `unconfirmed` — default for newly discovered clusters. Treated as `regular` for safety; `suspected_idle = true` surfaces it in the inspector for human review.

`suspected_idle` is set by the idle-pose heuristic (gesture-entropy + tail-position + temporal-position) and is advisory only — it never changes `kind` without a human decision.

Draft templates derived from the current corpus, after human review marks
cluster 15 as idle: all four present gestures collapse to **length-1
templates** (single `pose_id` per gesture). The multi-pose machinery is
retained in the design for future genuinely multi-phase gestures but has no
active users in today's corpus.

### Calibration observation (151-film corpus, 2026-04-23)

Initial calibration found a surprising **two-hold pattern** in three of four
gestures (`ok`, `stop`, `thumbs_up`) — median 2 holds per film with an
intra-film hold-pair distance (median 5.80 coord units) far exceeding the
inter-gesture centroid minimum (1.08). The two holds are genuinely distinct
configurations, not noise.

Clustering with `ε = 0.8` resolved the cause: one cluster (id 15 in that run,
n=131 holds) is a **neutral/relax pose** shared by all four gestures. Modal
templates were `[9, 15]` for `ok`, `[10, 15]` for `stop`, `[2]` or `[2, 15]`
for `point_left`. The gesture-specific cluster consistently appears as the
**first** hold; cluster 15 as the second.

Combined with hold-ordinal position stats (hold #1 median at 0.30 of in-view
duration, hold #2 at 0.60), the capture protocol produces:
1. User enters view and performs the gesture as they stabilise (hold #1,
   gesture-specific pose, position ≈ 0.30).
2. User releases into a neutral hand before exiting view (hold #2, shared
   cluster, position ≈ 0.60).

The idle-pose heuristic flags cluster 15 (entropy ≈ log 4 — all four gestures
contribute; tail-position in the modal templates of `ok`, `stop`, `thumbs_up`;
median `position_fraction` = 0.60 > 0.5 — all three signals fire). After a
reviewer marks cluster 15 as `idle` via the inspector, the templates become
length-1 for all four present gestures. Cluster 15 remains in the pose
vocabulary (so the classifier can detect it) and functions at runtime as the
primary gesture-end signal.

`thumbs_up` remains fragmented across ~20 gesture-specific clusters at ε = 0.8
because its training films were shot from different camera angles — the user
plans to retake or retire this gesture; no further ε tuning needed.

### Server-side changes

- New `server/ml/trainer_pose.py` (or extension to `trainer_rf_mlp.py`):
  1. Extract hold reps from every stored film using the calibrated parameters.
  2. Cluster representatives.
  3. Run the idle-pose heuristic to compute `suspected_idle` per cluster.
  4. Merge in `server/data/pose_corrections.json` (human review output) to set
     each cluster's `kind` and to drop any per-hold exclusions.
  5. Build the template manifest (idle-marked clusters are kept in
     `pose_clusters` and `idle_poses`, but stripped from every
     `gesture_templates` entry).
  6. Train the MLP on all labelled hold reps (including idle — the classifier
     must recognise them — but excluding per-hold-excluded samples).
  7. Export `pose_model.tflite` + `pose_manifest.json`.
- New endpoints:
  - `POST /train/pose` — triggers the pose-model training pipeline.
  - `GET /model/pose/download` — returns the tflite model.
  - `GET /model/pose/manifest` — returns the template manifest.
  - `POST /analyze/holds` — reusable endpoint for the inspection tool
    (below); takes a `HandFilm`, returns detected hold intervals and
    representative frame indices.
- `server/analyze_motion.py` becomes both a diagnostic tool and the source of
  truth for hold-detection + clustering logic that the trainer calls.

### iOS-side changes

- `GestureModel` gains a **single-frame prediction path** for pose
  classification (the existing path is handfilm-oriented).
- Second model download mirrors the existing handfilm model, with its manifest
  cached alongside.
- `HandGestureRecognizing` runtime:
  - Ring-buffer in-view frames while the motion gate is open.
  - Monitor motion energy to detect holds.
  - On each hold, run Phase 2 on the hold's argmin frame.
  - Maintain `observed_sequence: [pose_id]` and `candidate_gestures: set`.
  - Track the `T_commit` timer after the motion gate closes.

### Inspection tool: HandFilmsView pose inspector

Interactive inspector on the existing `HandFilmsView` in the training app.
Confirms visually what the clustering decided and lets a reviewer set each
cluster's `kind`. Heuristic flags surface but **never change `kind`
automatically** — human decision is required for `idle` or `regular`.

#### Per-film view

- Use `POST /analyze/holds` on the server (or port the detection to JS) to get
  hold intervals and representative frame indices for the film.
- Draw a timeline with coloured bands for each hold, highlighting the
  representative frame.
- Render the landmark skeleton at each hold's representative frame as a
  thumbnail, side by side.
- Label each thumbnail with its assigned `pose_id`, the cluster's current
  `kind` (`regular` / `idle` / `unconfirmed`), the distance to the cluster
  centroid, and a warning icon if the cluster is `suspected_idle`.
- **Per-hold action**: *Exclude this hold from this film's training*. Local
  correction — does not affect the cluster's overall `kind`.

#### Per-cluster view (primary confirmation surface)

- Grid of every hold thumbnail across the corpus, grouped by cluster. Within
  each group, sorted by distance from centroid so the "centre" of the cluster
  is obvious.
- Each cluster header shows:
  - Cluster id and size.
  - Gesture composition (bar or count per gesture).
  - Heuristic signals: gesture-entropy score, tail-position count, median
    temporal position, and the combined `suspected_idle` flag.
  - Current `kind`: `regular` / `idle` / `unconfirmed`.
- **Per-cluster actions**:
  - *Mark as idle* → `kind = "idle"`. Cluster is stripped from every
    `gesture_templates` entry, added to `idle_poses`, still used for
    classifier training.
  - *Mark as regular* → `kind = "regular"`. Included in templates.
  - *Reset to unconfirmed* → `kind = "unconfirmed"` (default behaviour).
- Suspected-idle clusters are visually distinct (e.g. a badge and a muted
  colour) so the reviewer can triage them first.

No cluster-level or film-level delete — heuristics surface, humans decide.

#### Corrections artefact

All inspector actions write to `server/data/pose_corrections.json` alongside
the manifest. Trainer reads it at manifest-build time:

```json
{
  "cluster_kinds": {
    "15": "idle",
    "9":  "regular"
  },
  "excluded_holds": [
    { "film_id": "abc-123", "hold_ordinal": 0 }
  ]
}
```

Keeps the training pipeline deterministic and the inspector the single source
of truth for human overrides. Clusters not listed in `cluster_kinds` stay
`unconfirmed` and behave as `regular`.

### Open questions

- **τ_commit** (pose confidence gate at inference): 0.6 is an initial guess;
  tune after first on-device run.
- **Re-clustering cadence**: do we re-cluster every time the corpus grows by
  some threshold, or only on demand via `POST /train/pose`? Re-clustering
  changes cluster ids, which invalidates the corrections file — so probably
  on demand, with a migration step for `cluster_kinds` via nearest-centroid
  mapping.
- **Default for `unconfirmed` clusters**: currently treat as `regular`. As
  the idle heuristic matures we could consider switching the default to
  "excluded until confirmed" to keep training cleaner; that change would
  require the inspector to be part of the standard training workflow.
- **Idle-pose heuristic thresholds** (entropy, tail-position count, median
  position): current values are tuned to 4 gestures; revisit at 10+.
- **Runtime edge case** (idle detected while `observed` is a live prefix
  without a complete match): current design commits to the longest
  complete-template ancestor if one exists, else discards. Validate with
  real data.
- **Handling non-modal training films** (~5 % at ε = 0.8): exclude
  per-hold via the inspector, or tolerate? Start with inspector-driven
  exclusion; revisit if undertraining.
- **LSTM trainer** for genuinely dynamic gestures with multiple ordered holds
  — deferred; current corpus doesn't need it.

---

## Phase 3 — Handfilm Classifier

### Status

**Solved**

- *Problem A (padding-ratio skew)* — `summaryFeatures` now uses real-frame count
  as the denominator (PR #34). Feature vectors of short and long clips of the same
  gesture are now comparable.
- *Runtime buffer too short* — `gestureBufferSize` 10 → 30 (PR #37). At 30 fps this
  matches `temporalWindow = 1.0 s` and the mean training clip length (~38 real frames).
- *Eval reproducibility* — `evaluate_trimmed.py` is seeded (PR #38). Produces a
  stable regression baseline; small ±1-example jitter remains from CPU oneDNN
  nondeterminism but does not affect conclusions.

**Open**

- *Problem B (length sensitivity below ~20 frames)*. The model classifies dynamic
  gestures correctly only when ≥ 20 real frames are available. Brief gestures or
  partial captures will misclassify. Per-class trim sweep:

  | trim | ok | point_left | stop | thumbs_up |
  |---:|---:|---:|---:|---:|
  | 30 | 1.00 | 1.00 | 0.875–1.00 | 1.00 |
  | 20 | 0.30–0.60 | 1.00 | 0.25 | 0.50–0.60 |
  | 10 | 0.10 | 0.67–1.00 | 0.00 | 0.10 |

  Static-pose gestures (`point_left`) are robust; dynamic ones need the full motion.
  Mitigations, in increasing complexity:
  1. Add gesture duration as an explicit feature so the MLP can condition on it.
  2. Time-warp augmentation in training (0.7×–1.4× resampling) to teach speed
     invariance.
  3. Motion-gated variable-length capture (Phase 1) — captures the gesture exactly
     as it happens regardless of length, and naturally bounds it to ≤ 1 s by the
     existing `temporalWindow`. Cleanest end-state.

  Recommendation: defer #1 and #2 until Phase 1 lands. If Phase 1's motion gate
  reliably catches every gesture's full motion arc, length sensitivity stops
  mattering at runtime.

- *Per-frame prediction*. Model fires on every camera frame, producing duplicate
  detections. Will be addressed by Phase 1 cooldown (~1 s after a confirmed
  detection).

- *Remaining gesture-to-gesture misclassification*. Status pending on-device
  re-evaluation after PR #34, #37 are in production. Current trim=30 numbers from
  the eval-MLP show `stop` occasionally drops to 0.875 (1 of 8 examples
  misclassified) and `thumbs_up` to 0.667 in some seeded runs (small-sample
  variance on 3-sample support). Treat as inconclusive until device data.

### Discarded recommendations (from earlier draft)

| Recommendation | Status | Reason |
|---|---|---|
| Shrink training window 2.0 s → 0.4 s | Discarded | Trim eval shows trim=12 → 23 % accuracy, trim=15 → 29 %. Films at 12 frames lack discriminative signal. |
| `gestureBufferSize: 10 → 15` | Superseded | Shipped 10 → 30 (PR #37). |
| `temporalWindow: 1.0 → 0.4` | Discarded | 1.0 s matches the new 30-frame buffer at 30 fps. |
| `captureWindow: 2.0 → 0.4` | Discarded | Training films at 2 s with mean 38 real frames work well. |
| `minInViewDuration: 1.2 → 0.25` | Discarded | Tied to the discarded capture-window change. |
| Trim existing 2 s films to middle 0.4 s for retraining | Discarded | Same as above — short clips lose signal. |

### Model

Existing MLP (`Dense(128) → Dropout(0.3) → Dense(64) → Dense(n_classes, softmax)` on
the 256-dim summary feature vector) is retained. Architecture change is not justified
by current data — full-clip and trim=30 accuracy are both 100 %.

LSTM upgrade: still deferred. Reconsider only if Phase 1 + duration feature don't
close the dynamic-gesture gap.

### Required follow-up

After PR #34 and #37 are in production:
1. Retrain on the existing corpus (no data changes; the new preprocessor produces
   length-invariant features).
2. Distribute the new `.tflite` to device.
3. Test on device. Expect runtime accuracy on the four trained gestures to track the
   eval-MLP's trim=30 numbers (~97–100 %).
4. If a stable misclassification persists, Phase 2 pose-gating will filter it before
   Phase 3 runs.

---

## Implementation Order

1. **Phase 3 alignment** — *done.*
   - `summaryFeatures` length-invariance fix (PR #34).
   - `gestureBufferSize: 10 → 30` (PR #37). `temporalWindow` stays at 1.0 s; capture
     window stays at 2.0 s; `minInViewDuration` unchanged.
   - Pending operational follow-up: retrain on production server, redistribute
     `.tflite`, on-device verification.

2. **Motion gate** (Phase 1):
   - Run `server/analyze_motion.py` to produce seed values for `T_open`,
     `K_open`, `T_close`, `K_close` from the existing corpus (gate-calibration
     section at the end of the report). Capture idle films only if the report
     flags insufficient idle-candidate data or overlapping duration
     distributions.
   - Implement heuristic in `HandGestureRecognizing.handleHandshot`.
   - Only buffer frames when the gate is open.
   - Add cooldown after each confirmed detection.

3. **Inspection tool** (Phase 2 prerequisite):
   - Server: `POST /analyze/holds` endpoint that reuses
     `server/analyze_motion.py`'s detection logic.
   - iOS: hold overlay on `HandFilmsView`, plus per-cluster grid view once
     clustering is available.
   - This lets a human verify before committing to the extracted pose vocabulary.

4. **Pose classifier** (Phase 2):
   - Server: `server/ml/trainer_pose.py` — extract holds, cluster, apply
     shared-cluster filter, build template manifest, train single-frame MLP.
   - Server: `POST /train/pose`, `GET /model/pose/download`, `GET /model/pose/manifest`.
   - iOS: second model download + manifest loading; single-frame prediction
     path in `GestureModel`.
   - iOS runtime: hold detection in `HandGestureRecognizing`, observed-sequence
     buffer, prefix matcher, T_commit timer.

5. **Evaluate** on real device, tune `τ_commit` and re-verify parameters on
   the grown corpus.

---

## Files That Will Change

| File | Change |
|---|---|
| `iOS/ModelTraining/.../CameraViewModel.swift` | default `captureWindow`, `pauseInterval` |
| `iOS/HandGestureRecognizing/.../Types.swift` | `gestureBufferSize`, `temporalWindow` defaults |
| `iOS/HandGestureRecognizing/.../HandGestureRecognizing.swift` | motion gate + hold-detection + prefix matcher + T_commit timer |
| `iOS/GestureModel/.../GestureModel.swift` | single-frame prediction path for pose classifier; second-model load |
| `iOS/ModelTraining/.../GestureRecognizerWrapper.swift` | updated config defaults |
| `iOS/ModelTraining/.../HandFilmsView.swift` | hold-timeline overlay, per-hold skeleton thumbnails, per-hold exclude action |
| `iOS/ModelTraining/.../PoseInspectorView.swift` (new) | per-cluster grid view, idle/regular marking actions |
| `iOS/ModelTraining/.../ViewModels/` | hold + cluster data fetch (via `/analyze/holds`); read/write corrections |
| `server/ml/trainer_pose.py` (new) | Phase 2 trainer: extract, cluster, apply corrections, train, export manifest |
| `server/ml/preprocessor.js` | (unchanged — Phase 2 uses existing normalised coords) |
| `server/analyze_motion.py` | source of truth for hold-detection + clustering logic used by trainer and `/analyze/holds` |
| `server/data/pose_corrections.json` (new) | reviewer output: `cluster_kinds`, `excluded_holds`; consumed by trainer |
| `server/routers/` | new `/train/pose`, `/model/pose/download`, `/model/pose/manifest`, `/analyze/holds`, `/pose/corrections` (GET/PUT) |

---

## Open Questions Summary

- [ ] Motion gate thresholds `T_open`, `T_close`, `K_open`, `K_close` — seeded from `server/analyze_motion.py` gate-calibration pass (onset energy + idle-candidate holds); confirm on-device.
- [ ] Gate-close vs hold-detect signal collision — `T_close` and `T_hold` use the same signal. Phase 2's idle-pose detection is now the primary commit path; the low-energy close is a fallback. Mitigate by keeping `T_close` well below `T_hold` and `K_close` above typical gesture-hold length.
- [ ] Phase 1 cooldown duration after confirmed detection (currently "~1 s" in the architecture diagram).
- [x] Phase 2 capture-protocol artefact — resolved: captures include a trailing neutral hold, flagged by the idle-pose heuristic and marked `idle` via the inspector.
- [x] Phase 2 parameters — calibrated (T_hold, K_hold, smooth_k, edge_trim, ε, T_commit).
- [ ] Phase 2 `τ_commit` (pose confidence gate) — tune on first on-device run.
- [ ] Phase 2 re-clustering cadence — on demand vs periodic. Re-clustering renumbers clusters, so `cluster_kinds` needs a nearest-centroid migration step.
- [ ] Idle-pose heuristic thresholds (entropy ≥ 0.8·log(n_gestures), tail-position ≥ 2, median position > 0.5) — revisit as vocabulary grows beyond 4 gestures.
- [ ] Default behaviour for `unconfirmed` clusters — currently `regular`; consider "excluded until confirmed" once the inspector is part of the standard workflow.
- [ ] Runtime edge case — idle detected while `observed` is a live prefix without a complete match: commit to longest complete-template ancestor vs discard. Validate on-device.
- [x] Phase 3 padding-ratio skew (Problem A) — fixed in PR #34.
- [x] Phase 3 runtime buffer length — bumped to 30 frames in PR #37; trim eval validates the choice.
- [ ] Phase 3 length sensitivity below ~20 frames (Problem B) — deferred until Phase 1 motion-gating lands; revisit if the gate doesn't bound captures to ≥20 real frames in practice.
- [ ] Phase 3 remaining misclassification — pending on-device re-evaluation after retrain and `.tflite` redistribution.
- [ ] `thumbs_up` retake — deferred to training-data curation.
- [ ] When to add the LSTM trainer for genuinely dynamic gestures with ordered multi-hold templates.
