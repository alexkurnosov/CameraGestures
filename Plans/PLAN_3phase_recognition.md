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
        │ motion detected
        ▼
┌───────────────────┐
│  Phase 2          │  on first frame of motion window
│  Initial Pose     │──── unknown pose → discard, reset gate
└───────────────────┘
        │ known pose, class C
        ▼
┌───────────────────┐
│  Phase 3          │  collect 0.3–0.5 s handfilm, run once
│  Handfilm Model   │──── low confidence → discard
└───────────────────┘
        │ confirmed gesture, class C
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

## Phase 2 — Initial Pose Classifier

### Purpose
Identify the starting hand configuration and narrow down candidate gesture classes.
Acts as a fast, cheap "are we looking at a known gesture?" gate before committing to
a full handfilm capture.

### Training data
**Already exists.** Every stored `TrainingExample` contains a `HandFilm` whose frames
are labeled with `gesture_id`. For mostly-static gestures, individual frames carry the
same label as the film.

Decomposition:
- Take each `HandFilm`, filter absent frames (already done by preprocessor).
- Trim the first and last ~15 % of frames (transitional poses, hand entering/leaving
  view).
- Each remaining frame becomes one training example with the film's `gesture_id`.
- Result: ~30–50 examples per film → large multiplier on existing data.

**Important**: split train/test by `session_id` (or `user_id`), not by frame.
Frame-level split leaks near-duplicate poses from the same film into both sets.

### Feature vector
Single frame → `coordsOnly` (63 floats: 21 landmarks × 3, wrist-relative +
scale-normalised + rotation-aligned). Already computed as the first 63 dims of
`featureMatrix` row 0. No new preprocessing needed; just slice.

### Model size
The pose space is simpler than the full temporal space → smaller model is likely
sufficient:
- Candidate: `Dense(64, relu) → Dense(32, relu) → Dense(n_classes, softmax)`
- Compare against the existing MLP's 128 → 64 → n_classes to confirm smaller is enough.

### Server-side changes needed
- New trainer function `train_pose_classifier(examples)` in `ml/`.
- New `/train_pose` endpoint (or a `mode=pose` parameter on existing `/train`).
- New model download endpoint on iOS (separate file from handfilm model).

### Open questions
- How many frames per film to use? Middle 70 % is a reasonable starting point.
- Should the pose classifier output a confidence per class, or just a binary
  "known / unknown"? Per-class confidence is more useful — Phase 3 can use it to
  narrow its own output space.
- Do we train one pose model for all gestures, or per-gesture? One model for all is
  simpler and works unless gestures share identical starting poses.

### Observation: unexpected second hold in static-gesture films (2026-04-23)

The Phase 2 calibration script (`server/analyze_motion.py`) was run against the
current training corpus (151 films, 4 gestures: `ok`, `stop`, `thumbs_up`,
`point_left`). Summary findings:

- Three of the four gestures (`ok`, `stop`, `thumbs_up`) — all conceptually
  single-pose — produce **two detected holds per film** (median = 2) under any
  `T_hold` in a wide band (0.5–2.1). `point_left` produces median = 1.
- Intra-film hold-pair distance has p10 = 5.36, median 5.80 in normalised coord
  units. Inter-gesture centroid distances are 1.08–2.60. The two holds within a
  single film are **~5× further apart than the centroids of different
  gestures** — they cannot be near-duplicates or noise.
- MediaPipe re-emits the previous detection unchanged ~35 % of the time.
  Energy percentiles used for threshold selection now exclude these zero-delta
  samples.

**Working hypothesis** (unverified): the capture protocol produces two distinct
stable configurations in each film because the user first *moves the hand into
view and stabilises it*, then *performs the gesture*. Under this hypothesis,
hold #1 is the in-view positioning pose and hold #2 is the actual gesture.

**Indirect evidence to check**: if the hypothesis is correct, hold #1 should
cluster temporally near the film start (just past the 15 % edge-trim, say
position_fraction ≈ 0.15–0.3), and hold #2 should land later. `analyze_motion.py`
now reports position_fraction stats broken down by hold ordinal (1st, 2nd, …)
per film. Pending — run the updated script against the VPS corpus.

**Mitigation options** (deferred until the hypothesis is confirmed):
- **Phase 1 side**: once the motion gate is live, the entry-into-view phase
  will be excluded from the captured buffer because the user's hand is not yet
  "moving into the gesture" at that point — the gate would only open when real
  gesture motion starts.
- **Phase 2 side**: post-extraction, drop holds whose distance from the
  per-gesture centroid exceeds a threshold (keeps only the "real" pose). Or
  keep only the last hold in each film.
- **Training data side**: add a hand-labelled annotation indicating which hold
  is the "true" gesture pose, so the template manifest is built from the right
  hold rather than all detected holds.

For now, keep both holds. This preserves information until we've confirmed
the hypothesis and decided how to handle it. Multi-pose templates
(`gesture_id → [pose_id, ...]`) are compatible either way: once we know hold
#1 is an artefact, those single-pose gestures collapse to length-1 templates.

### Inspection tool: HandFilmsView hold overlay

To confirm or refute the hypothesis above, we need to *see* the detected holds
on each film, not just aggregate statistics. Proposal: extend the existing
`HandFilmsView` (iOS, ModelTrainingApp) with a hold overlay.

**Per-film display:**
- Run the same hold-detection algorithm the Python script uses (port the
  JS version of motion energy + smoothing + run-length detection, or have the
  server expose a `POST /analyze/holds` endpoint that takes a `HandFilm` and
  returns the hold intervals and representative frames).
- Mark each detected hold on the film's timeline: a coloured band spanning the
  hold's frame range, with the representative frame index highlighted.
- Render the landmark skeleton for each hold's representative frame as a
  thumbnail, side by side.
- Show the two distances: (a) hold-pair distance within the film, (b) each
  hold's distance to the per-gesture centroid.

**Per-cluster display** (after Phase 2 clustering is wired up):
- Group all detected hold representatives by assigned `pose_id`.
- Render each cluster as a row of skeleton thumbnails, sorted by intra-cluster
  distance from the cluster centre. Visually obvious whether the cluster is
  coherent or contaminated.
- Allow a human reviewer to flag a hold as "wrong pose" or "transitional",
  producing a corrections file that can feed back into training.

**Scope for the first version**: read-only overlay on the existing films list,
no interactive labelling yet. Interactive labelling is a follow-up once we
know the overlay produces useful signal.

**Files that will change for the inspection tool:**

| File | Change |
|---|---|
| `iOS/ModelTraining/.../HandFilmsView.swift` | timeline overlay + per-hold thumbnails |
| `iOS/ModelTraining/.../ViewModels/` | load hold data per film (from server or computed locally) |
| `server/routers/` | optional `POST /analyze/holds` endpoint that reuses `analyze_motion.py`'s hold-detection code |

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
   - Only open capture window when gate is open.
   - Add cooldown after each confirmed detection.

3. **Pose classifier** (Phase 2):
   - Server: frame-decomposition trainer.
   - iOS: second model download + `GestureModel` variant for single-frame input.
   - Wire into the gate → handfilm pipeline.

4. **Evaluate** on real device, tune thresholds.

---

## Files That Will Change

| File | Change |
|---|---|
| `iOS/ModelTraining/.../CameraViewModel.swift` | default `captureWindow`, `pauseInterval` |
| `iOS/HandGestureRecognizing/.../Types.swift` | `gestureBufferSize`, `temporalWindow` defaults |
| `iOS/HandGestureRecognizing/.../HandGestureRecognizing.swift` | motion gate logic, cooldown |
| `server/ml/trainer_rf_mlp.py` | or new `trainer_pose.py` for Phase 2 |
| `server/routers/` | new training trigger for pose model |
| `iOS/GestureModel/.../GestureModel.swift` | optional: single-frame prediction path |
| `iOS/ModelTraining/.../GestureRecognizerWrapper.swift` | updated config defaults |

---

## Open Questions Summary

- [ ] Motion gate thresholds — needs empirical data from stored films.
- [ ] Pose classifier: middle-70 % frame slice vs other sampling strategy.
- [ ] Phase 2 output: per-class confidence vs binary known/unknown.
- [ ] Phase 3: retrain on new short films vs trim existing films.
- [ ] Remaining misclassification pair: which gestures, and does it resolve after window fix.
- [ ] When to add the LSTM trainer (Phase 2 long-term for dynamic gestures).
