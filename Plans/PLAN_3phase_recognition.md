# 3-Phase Gesture Recognition — Design Plan

## Context

### Current state (as of 2026-04-21)

- Preprocessor now filters absent frames before feature extraction (`nonAbsentFrames` in
  `preprocessor.js`). Result: false-positive rate dropped significantly; one stable
  gesture-to-gesture misclassification remains (to be diagnosed after Phase 1–2 models
  land).
- Remaining structural problems:
  1. **Window mismatch**: training films are 2 s, runtime buffer is ~10 frames ≈ 0.33 s.
     The preprocessor pads both to 60 frames, so the zero-fill proportion is very
     different between training and inference.
  2. **Per-frame prediction**: recognition fires on every camera frame (~30×/s), causing
     duplicate detections of the same gesture.
  3. **No gesture boundary detection**: no concept of "gesture started / ended".

### Key design constraint

**A realistic gesture lasts 0.3–0.5 s.** The 2 s capture window was always too long.
At 30 fps:
- 0.3 s ≈ 9 frames
- 0.5 s ≈ 15 frames

The runtime `gestureBufferSize` is already 10 frames (~0.33 s) — it was accidentally
close to the right value all along. The fix is to bring *training* down to match.

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

### Open questions
- What threshold values work for the gesture set? Needs empirical tuning; SSH to VPS
  and query frame-by-frame landmark deltas from stored films would give baseline.
- Should absent frames (no hand in view) reset the gate immediately, or tolerate short
  occlusions? Likely: reset immediately, since a missing hand cannot be "mid-gesture".

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

Apply the **shared-cluster filter** to exclude neutral/rest poses:
> A cluster is "shared" if ≥ 3 distinct gestures contribute holds to it and each
> contributing gesture accounts for ≥ 5 % of the cluster's total. Shared clusters
> are removed from the pose vocabulary and from every gesture's template.

For each gesture, the **canonical template** is the modal filtered sequence
across its films. Non-modal films fall into one of: fewer holds than the modal,
extra holds beyond the modal, or a genuinely different pattern. If the modal
fraction is ≥ 70 %, the gesture template is the modal; otherwise flag the
gesture for capture review.

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
  ├─ top pose confidence < τ  → discard capture, reset gate
  │
  └─ top pose is a shared cluster  → ignore this hold, keep buffering
  │
  otherwise: append predicted pose_id to `observed = [p1, p2, ...]`
  │
  ▼
prefix-match observed against gesture_templates
  │
  ├─ no template starts with observed  → discard capture, reset gate
  │
  ├─ some templates match, none complete → keep buffering, start T_commit timer
  │     if a new hold arrives before T_commit expires, extend observed and re-match
  │     otherwise (timer fires): commit to any complete match that is a prefix
  │
  └─ at least one template equals observed  → commit to candidate set
        S = { g : template(g) == observed }
  │
  ▼
Phase 3 runs on the buffered film, output restricted to S
```

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
| shared-cluster rule | **≥ 3 gestures, each ≥ 5 %** | Currently flags exactly one cluster (the neutral/relax pose) with no false positives. |

Calibration uses non-zero energy percentiles because MediaPipe re-emits the
previous detection unchanged ~35 % of the time, which would pin any
percentile-based threshold to zero.

### Classifier

- **Architecture**: `Dense(64, relu) → Dense(32, relu) → Dense(n_pose_clusters, softmax)`
- **Input**: 63-dim hold representative (one frame of normalised coords).
- **Output**: argmax over `pose_id` with confidence. If top confidence <
  `τ_commit`, the hold is rejected as "unknown pose".
- **Training labels**: `pose_id`s produced by the clustering pass (not
  `gesture_id`s directly). Shared clusters are excluded from training.
- **Train/test split**: by `session_id` — never per-frame, which would leak
  near-duplicate holds from the same film across the split.

### Template manifest (on-disk format alongside the model)

```json
{
  "version": 1,
  "pose_clusters": {
    "2":  { "label": "pose_point",  "n_samples": 13, "centroid": [...63 floats...] },
    "9":  { "label": "pose_ok",     "n_samples": 45, "centroid": [...] },
    "10": { "label": "pose_stop",   "n_samples": 36, "centroid": [...] }
  },
  "shared_clusters_excluded": [15],
  "gesture_templates": {
    "ok":         [9],
    "stop":       [10],
    "point_left": [2]
  },
  "parameters": {
    "t_hold": 0.54, "k_hold_frames": 3, "smooth_k": 3,
    "edge_trim_fraction": 0.15, "t_commit_ms": 300,
    "epsilon": 0.8, "tau_commit": 0.6
  }
}
```

Draft templates derived from the current corpus after applying the shared-cluster
filter: all four present gestures collapse to **length-1 templates** (single
`pose_id` per gesture). The multi-pose machinery is retained in the design for
future genuinely multi-phase gestures but has no active users in today's corpus.

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

The shared-cluster filter strips the trailing neutral from templates, leaving
clean length-1 templates for all four present gestures. `thumbs_up` remains
fragmented across ~20 gesture-specific clusters at ε = 0.8 because its training
films were shot from different camera angles — the user plans to retake or
retire this gesture; no further ε tuning needed.

### Server-side changes

- New `server/ml/trainer_pose.py` (or extension to `trainer_rf_mlp.py`):
  1. Extract hold reps from every stored film using the calibrated parameters.
  2. Cluster representatives.
  3. Apply the shared-cluster filter.
  4. Build the template manifest.
  5. Train the MLP on labelled hold reps (excluding shared-cluster samples).
  6. Export `pose_model.tflite` + `pose_manifest.json`.
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

### Inspection tool: HandFilmsView hold overlay

Read-only overlay on the existing `HandFilmsView` in the training app. Confirms
visually what the clustering decided.

**Per-film display:**
- Use `POST /analyze/holds` on the server (or port the detection to JS) to get
  hold intervals and representative frame indices for the film.
- Draw a timeline with coloured bands for each hold, highlighting the
  representative frame.
- Render the landmark skeleton at each hold's representative frame as a
  thumbnail, side by side.
- Label each thumbnail with its assigned `pose_id` (after clustering) plus
  the distance to the cluster centroid, and mark shared-cluster holds with
  a distinctive colour so they're easy to separate visually from the real
  gesture holds.

**Per-cluster display:**
- A separate view that groups every detected hold across the corpus by
  `pose_id`.
- Each cluster rendered as a grid of skeleton thumbnails, sorted by distance
  from the cluster centroid. Makes it obvious at a glance whether a cluster is
  coherent or contaminated.
- Shared clusters are displayed but marked clearly as excluded from templates.

Interactive labelling (flag a hold as "wrong pose" → corrections file feeding
back into training) is a second-pass feature — ship the overlay first.

### Open questions

- **τ_commit** (pose confidence gate at inference): 0.6 is an initial guess;
  tune after first on-device run.
- **Re-clustering cadence**: do we re-cluster every time the corpus grows by
  some threshold, or only on demand via `POST /train/pose`?
- **Shared-cluster threshold** (≥ 3 gestures, ≥ 5 % each): works for 4
  gestures; revisit as vocabulary grows. The "3" may need to scale.
- **Handling non-modal training films** (~5 % at ε = 0.8): exclude from training
  (cleaner) or tolerate (more data)? Start with exclusion; revisit if undertraining.
- **LSTM trainer** for genuinely dynamic gestures with multiple ordered holds
  — deferred; current corpus doesn't need it.

---

## Phase 3 — Handfilm Classifier

### Key change: shorten the capture window

Reduce `captureWindow` from 2.0 s → **0.4 s** (configurable 0.3–0.5 s).
This directly fixes the window-mismatch bug:

| | Current | After fix |
|---|---|---|
| Training window | 2.0 s / ~60 real frames | 0.4 s / ~12 real frames |
| Runtime buffer (`gestureBufferSize`) | 10 frames (~0.33 s) | 12–15 frames (~0.4–0.5 s) |
| Runtime `temporalWindow` | 1.0 s | 0.4 s |
| Zero-padding in preprocessor | 50 / 60 frames zeros at runtime | ~0 frames zeros (matched) |

The `summaryFeatures` function already handles variable-length inputs (pads or trims to
60). Once training and runtime windows match, the feature distribution aligns without
any other changes.

`minInViewDuration` should be scaled proportionally: currently 1.2 s / 2.0 s = 60 %.
At 0.4 s window → `minInViewDuration` = **0.25 s**.

### Model options

| Option | Pro | Con |
|---|---|---|
| **Retrain existing MLP on 0.4 s films** | no architecture change, fast | need to re-collect or confirm existing films can be trimmed |
| Train second MLP separately | clean separation | doubles model count |
| Upgrade to LSTM on 60 × 126 feature matrix | captures full temporal dynamics | 10–50× longer training, larger model, Phase 2 LSTM trainer is still a stub |

**Recommendation**: retrain existing MLP on newly collected 0.4 s films. Existing 2 s
films can be used by trimming to the middle 0.4 s (frames 25–37 of 60), but fresh data
with the shorter window will be cleaner.

### Remaining misclassification
One gesture is consistently misclassified as another. Likely diagnosis after retraining:
- If it persists → overlapping poses (Phase 2 will help gate it).
- If it resolves → it was caused by window-mismatch or absent-frame contamination.
- If it improves but doesn't resolve → need more training examples for that pair.

---

## Implementation Order

1. **Capture window reduction** (Phase 3 prerequisite, highest leverage):
   - `CameraViewModel.captureWindow` default: `2.0` → `0.4`
   - `AppSettings.minInViewDuration` default: `1.2` → `0.25`
   - `GestureRecognizerWrapper.gestureBufferSize`: `10` → `15`
   - `GestureModelConfig.temporalWindow`: `1.0` → `0.4`
   - Re-collect training data and retrain.

2. **Motion gate** (Phase 1):
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
| `iOS/ModelTraining/.../HandFilmsView.swift` | hold-timeline overlay, per-hold skeleton thumbnails |
| `iOS/ModelTraining/.../ViewModels/` | hold data fetch (via `/analyze/holds`) + per-cluster grid view |
| `server/ml/trainer_pose.py` (new) | Phase 2 trainer: extract, cluster, filter, train, export manifest |
| `server/ml/preprocessor.js` | (unchanged — Phase 2 uses existing normalised coords) |
| `server/analyze_motion.py` | becomes source of truth for hold-detection + clustering used by trainer |
| `server/routers/` | new `/train/pose`, `/model/pose/download`, `/model/pose/manifest`, `/analyze/holds` |

---

## Open Questions Summary

- [ ] Motion gate thresholds `T_open`, `T_close`, `K_open`, `K_close` — empirical, pending on-device testing.
- [x] Phase 2 capture-protocol artefact — resolved: captures include a trailing neutral hold, excluded by the shared-cluster filter.
- [x] Phase 2 parameters — calibrated (T_hold, K_hold, smooth_k, edge_trim, ε, T_commit).
- [ ] Phase 2 `τ_commit` (pose confidence gate) — tune on first on-device run.
- [ ] Phase 2 re-clustering cadence — on demand vs periodic.
- [ ] Phase 2 shared-cluster threshold as vocabulary grows (currently `≥ 3 gestures, ≥ 5 % each`).
- [ ] Phase 3: retrain on new short films vs trim existing films.
- [ ] Remaining misclassification pair: which gestures, and does it resolve after the window fix + Phase 2 gating.
- [ ] `thumbs_up` retake — deferred to training-data curation.
- [ ] When to add the LSTM trainer for genuinely dynamic gestures with ordered multi-hold templates.
