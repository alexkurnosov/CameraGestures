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
- `K_close` is **fixed at 0.3 s (≈ 9 frames at 30 fps)** rather than seeded
  from the corpus. Phase 2's idle-pose commit and absent-hand are the
  primary close paths; the low-energy close is a sustained-relax fallback,
  and 0.3 s is comfortably longer than the typical gesture-specific hold
  (`K_hold` = 3 frames ≈ 100 ms) so it does not race with hold detection.
  The calibration report still emits the idle-vs-gesture run-length
  comparison for diagnostic purposes.

All four parameters (`T_open`, `K_open`, `T_close`, `K_close`) plus the
post-detection cooldown are exposed in the runtime configuration, not
baked into code.

Both gesture holds and idle holds are low-energy by construction, so energy
alone can't perfectly distinguish them; `K_close` relies on duration. The
calibration report flags whether the gap is clean.

If the idle-candidate data turns out to be too thin or the duration gap
overlaps, add a one-off "idle capture" mode to the training app (5–10 s films
of relaxed hand in view, flagged as idle) and re-run calibration.

### Decisions

- **Absent frames** (no hand in view) reset the gate immediately. A missing
  hand cannot be "mid-gesture", and any subsequent re-acquisition starts a
  fresh capture.
- **Gate-close vs hold-detect** run **concurrently** when smoothed energy
  drops below `T_close`. Because `T_close < T_hold` by calibration, any
  below-`T_close` run is also below-`T_hold`, so two counters increment in
  parallel:
  - `K_hold` (3 frames) trips first → run Phase 2 pose classification,
    append the predicted `pose_id` to `observed` if it is a regular pose,
    or treat it as the idle commit signal if it is idle.
  - `K_close` (9 frames) trips later → close the gate as the fallback
    commit/discard path described in the Phase 2 runtime flow.
  This makes the same low-energy run produce up to two events in order
  (hold first, then close) instead of forcing a choice between them.
- **Post-detection cooldown** is a configurable parameter
  `cooldown_after_cycle_ms`, default **1000 ms**, applied after every cycle
  (commit or discard). See *Cross-phase cooldown* in the cross-cutting
  section.
- **Motion-energy source of truth.** Phase 1 and Phase 2's runtime hold
  detector consume the **same per-frame raw energy signal** computed once
  per frame. Each phase applies its own smoothing/threshold layer on top
  (Phase 1 favours latency on opens; Phase 2 favours stable hold
  boundaries). Tradeoff: shared raw signal keeps both phases coherent with
  the offline calibration and saves CPU; per-phase smoothing avoids
  forcing one filter to serve two different objectives.

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
>   > 0.5. (`position_fraction` = hold representative's in-view ordinal /
>   total in-view frame count for that film. 0 = first in-view frame, 1 =
>   last in-view frame.)
>
> A cluster is flagged **suspected_idle** when any two of the three signals
> trigger. The flag is advisory only.

**Human confirmation via inspector** (see *Inspection tool* section). Each
cluster has a `kind` of `regular`, `idle`, or `unconfirmed`. **Unconfirmed
clusters are excluded from inference**: their hold representatives are not
used as training labels for the pose MLP, they do not appear in any
`gesture_templates` entry, and at runtime a hold whose nearest cluster is
`unconfirmed` is rejected (treated like a sub-`τ_pose_confidence` hold).
The inspector surfaces them in a dedicated review queue so they cannot
silently leak into recognition. Only clusters explicitly marked `idle` are
treated as gesture-end markers; only clusters explicitly marked `regular`
participate in templates.

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
  ├─ top pose confidence < τ_pose_confidence            → discard capture, reset gate
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
        ├─ no completion yet, live prefix(es) exist     → keep buffering, no timer
        │                                                  (wait for next hold; nothing
        │                                                  to commit to even if timer fires)
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

**Minimum-buffer gate.** Every commit path is conditional on the buffer having
accumulated at least `T_min_buffer` (200 ms ≈ 6 frames) since gate-open. If a
commit signal fires before that, the commit is deferred until `T_min_buffer`
elapses, then re-evaluated. This prevents Phase 3 from running on a buffer
too short to classify. The threshold is a sanity floor; Phase 3's own
length-sensitivity (see Phase 3) governs the actually-useful buffer length.

**`T_min_buffer` vs `T_commit` — what each one measures.**
- `T_commit` (300 ms) is a **post-hold deferral**: it starts when a hold
  ends that produced a complete-template prefix-match, but a longer
  prefix is still possible (e.g. observed = `[A]`, both `[A]` and
  `[A, B]` exist as templates). It waits for the *next* hold; if none
  arrives before it expires, the completed prefix is committed.
- `T_min_buffer` (200 ms) is a **post-gate-open floor**: it measures
  buffer length since gate-open and applies regardless of which commit
  signal fired (idle pose, `T_commit` expiry, or gate-close).
- They do not race. `T_commit` runs in *hold-end-relative* time;
  `T_min_buffer` runs in *gate-open-relative* time. When any commit
  signal fires, `T_min_buffer` is checked first: if buffer-since-
  gate-open < `T_min_buffer`, the commit is deferred until that floor
  is met and then re-evaluated. If `T_min_buffer` is met, the commit
  proceeds immediately.

**"Reset gate"** as used throughout this section means: discard the in-view
frame buffer, clear `observed`, and return Phase 1 to its watching state.
After any commit (Phase 3 fires) or any discard, the entire pipeline enters
the cross-phase cooldown — see Phase 1.

### Parameters

Empirically chosen from the 151-film corpus (`ok`, `stop`, `point_left`,
`thumbs_up`) using `server/analyze_motion.py`:

| Parameter | Value | Rationale |
|---|---|---|
| `T_hold` | **0.54** | p75 of non-zero energy. `T_hold` sweep showed a flat plateau over 0.5–2.1 — threshold is not fragile. |
| `K_hold` | **3 frames** | ~100 ms at 30 fps; excludes jitter without missing short holds. |
| `smooth_k` | **3 frames** | Tight smoothing; preserves hold boundaries. |
| `edge_trim` | **15 %** | *Training only.* Drops 3.5 % of detected holds on the current corpus — removes enter/exit artefacts without undershooting. Not applied at runtime: Phase 1's motion gate is assumed to already exclude the equivalent entry/exit phase. May need re-tuning once on-device behaviour is observed. |
| `T_commit` | **300 ms** | After hold-close, wait this long for another hold before committing to a match. Handles prefix collisions (short templates that are prefixes of longer ones). |
| `ε` (clustering) | **0.8** | Below the inter-gesture centroid minimum (1.08 between `ok` and `stop`). Produces 29 clusters on the current corpus, with clean gesture-specific groupings for three of four gestures. |
| `τ_pose_confidence` | **0.6** (configurable) | Holds with top pose-classifier confidence below this are rejected as "unknown pose". Distinct from `T_commit`, which is a time threshold. Tuning method below. |
| `T_min_buffer` | **200 ms** (≈ 6 frames at 30 fps) | Minimum buffer duration since gate-open before Phase 2 may commit. Prevents pathological short captures where Phase 3 has too few frames to classify. If a commit signal (idle pose, prefix-only-completion, or T_commit timer) fires before this, the commit is deferred until `T_min_buffer` elapses; if no further commit signal arrives by then, fall through to gate-close behaviour. |
| `idle_entropy_threshold` | **0.8** (× `log(n_gestures)`) | Gesture-entropy threshold for the idle-pose heuristic. |
| `idle_tail_position_min` | **2** | Minimum number of distinct gestures whose modal template ends with the cluster, for the tail-position signal to fire. |
| `idle_median_position_min` | **0.5** | Minimum median `position_fraction` for the temporal-position signal to fire. |
| `idle_signals_required` | **2 of 3** | Number of idle-heuristic signals that must trigger to flag a cluster `suspected_idle`. Advisory only; never auto-excludes. |

Calibration uses non-zero energy percentiles because MediaPipe re-emits the
previous detection unchanged ~35 % of the time, which would pin any
percentile-based threshold to zero.

### Classifier

- **Architecture**: `Dense(64, relu) → Dense(32, relu) → Dense(n_pose_clusters, softmax)`
- **Input**: 63-dim hold representative (one frame of normalised coords).
- **Output space**: all pose clusters, *including idle*. The classifier must
  recognise idle poses because runtime uses them as gesture-end markers.
- **Output decision**: argmax over `pose_id` with confidence. If top
  confidence < `τ_pose_confidence`, the hold is rejected as "unknown pose".
- **Training labels**: `pose_id`s produced by the clustering pass (not
  `gesture_id`s directly). Idle-marked clusters are *included* in training.
  Holds explicitly excluded via the inspector's per-hold action are dropped.
- **Train/test split**: by `session_id` — never per-frame, which would leak
  near-duplicate holds from the same film across the split.
- **Class imbalance**: idle clusters tend to dominate (current corpus: 131
  idle samples vs 13–45 per gesture-specific cluster, ~3× ratio). Train with
  **class-weighted cross-entropy**, weights ∝ `1 / sqrt(n_samples_per_cluster)`.
  Without weighting the model defaults to predicting "idle" on borderline
  inputs, which causes Phase 2 to commit gestures prematurely instead of
  appending the predicted regular pose to `observed`.

### Tuning `τ_pose_confidence`

*Offline (from corpus):* during pose-MLP training, evaluate the validation
fold's softmax distribution. Sweep `τ` over `[0.30, 0.95]` in 0.05 steps
and plot two curves:
- **Acceptance rate** = fraction of holds whose top confidence ≥ `τ`.
- **Conditional accuracy** = accuracy on accepted holds (predicted
  `pose_id` == true label, restricted to confidence ≥ `τ`).

Pick the smallest `τ` whose conditional accuracy meets the target (initial
target: ≥ 0.95) without dropping acceptance below ~0.85. `trainer_pose.py`
emits this curve into the metrics payload so `MetricsView` can show it.

*On-device:* the holds-recognising mode in `CameraView` logs every hold's
top-pose confidence plus a reviewer tag (correct / wrong / skipped). The
log is uploaded to the server via `POST /pose/confidence-log` —
independently of film upload, as batches of
`{model_version, predicted_pose_id, confidence, reviewer_label, timestamp, film_id?}`.
An optional `film_id` cross-links the entry to a film uploaded through the
existing path, enabling landmark-level auditability without requiring it.
The server recomputes the acceptance/conditional-accuracy curve from the
received tuples and surfaces it in `MetricsView` next to the offline
curve. Divergence between the two curves is the signal that `τ` needs
adjustment.

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
    "epsilon": 0.8, "tau_pose_confidence": 0.6,
    "idle_entropy_threshold": 0.8,
    "idle_tail_position_min": 2,
    "idle_median_position_min": 0.5,
    "idle_signals_required": 2
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
  - `POST /pose/confidence-log` — accepts batches of
    `{model_version, predicted_pose_id, confidence, reviewer_label, timestamp, film_id?}`
    from device; server recomputes the on-device acceptance/conditional-accuracy
    curve for `τ_pose_confidence` tuning. `film_id` is optional; when present,
    the entry is cross-linked to the stored film for landmark-level auditability.
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
- `CameraView` (model training app, recognizing mode) gains a **mode switch**
  between two recognising modes:
  - *Handfilm* (existing) — the full pipeline runs and the recognised
    gesture is shown.
  - *Holds* (new) — Phase 1 + Phase 2 only. Surfaces in real time: motion-gate
    state (open/closed), each detected hold with its representative-frame
    skeleton thumbnail, the predicted `pose_id` and confidence (with cluster
    `kind`: regular / idle / unconfirmed), the current `observed_sequence`,
    and the matched template (or "no match"). Acts as the on-device
    instrument for verifying that hold detection and pose classification
    behave as expected before the full pipeline is wired together — it is
    how the user reads off Phase 2 confidence at runtime, complementing the
    offline corpus metrics below.

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

### Evaluation on the current corpus

Three layers of metrics; `server/analyze_motion.py` already covers the first.

1. **Clustering quality** — *implemented* in `server/analyze_motion.py`.
   Per ε: number of clusters, size distribution, per-gesture modal-template
   fraction, gesture-composition of each cluster, idle-candidate flagging.
   Answers: "do holds cluster cleanly into gesture-specific groups?"

2. **Pose-MLP recognition accuracy** — *to be produced by* `trainer_pose.py`
   (new). Train/test split by `session_id`; report per-class precision and
   recall over `pose_id`, plus the confusion matrix. Answers: "given a
   correctly-detected hold representative, does the MLP pick the right
   `pose_id`?" Critical because `τ_pose_confidence` is calibrated against
   this distribution.

3. **End-to-end Phase 2 accuracy on the corpus** (the "#2 strategy" — Phase 1+2
   only — from the strategy comparison) — *new, not yet implemented.*
   Add `server/evaluate_pose.py` as the Phase-2 analog of `evaluate_trimmed.py`.
   For each held-out film, run the full pipeline that the inference path will
   run — hold detection (using the calibrated `T_hold`/`K_hold`/`smooth_k`),
   pose classification on each hold's representative frame, idle-pose
   filtering via the manifest's `cluster_kinds`, prefix-match of the
   resulting `observed_sequence` against `gesture_templates` — and compare
   the committed gesture against the film's labelled gesture. Report:
   - per-gesture commit rate (fraction of films that produce *any* commit),
   - per-gesture commit-correct rate (committed gesture == labelled),
   - rate of "no template prefix matches" (Phase 2 discards the capture),
   - rate of premature idle commit (idle hold appears before the gesture
     hold and commits to the wrong template),
   - distribution of `observed_sequence` lengths at commit, broken down by
     commit signal (idle / `T_commit` timer / gate-close).
   - **idle-while-live-prefix rate**: count of cycles where an idle hold
     fired while `observed` was a live prefix without a complete match,
     and per such cycle whether the chosen action (commit longest
     complete-template ancestor, else discard) matched the labelled
     gesture. This is the metric that validates the runtime edge-case
     decision — if the action's success rate is materially worse than
     the headline commit-correct rate, revisit the rule.
   - **non-modal exclusion impact**: per-class pose-MLP recall computed
     once with non-modal-film holds excluded (current policy) and once
     with them retained. If exclusion drops any class's recall below
     0.90 (or below the retained-policy recall by > 5 pp), that class
     is flagged for inspector review — the signal that the "exclude
     non-modal" default needs revisiting.

   This is the metric that tells us whether Phase 2 alone could carry the
   pipeline on the current corpus. It is also what closes the loop on the
   runtime edge case of "idle while `observed` is a live prefix" — the rate
   of that case on real films is observable here.

**Both layer #2 (pose-MLP recognition accuracy) and layer #3 (end-to-end
Phase 2 accuracy = "#2 strategy" headline number) MUST be exposed through the
existing metrics surface alongside the Phase 3 numbers.** Concretely:

- `trainer_pose.py` writes them into the model-training result the same way
  `trainer_rf_mlp.py` writes Phase 3 metrics today (see
  `_compute_extended_metrics` in `server/ml/trainer_rf_mlp.py`).
- `/model/metrics` (or a sibling `/model/pose/metrics` keyed by the
  pose-model id) serves them with the same shape conventions used today —
  per-class precision/recall/F1, confusion matrix, history of past runs.
- `MetricsView` in the iOS training app gains a Phase 2 section showing the
  pose-MLP per-class breakdown, the headline end-to-end commit-correct rate,
  and the auxiliary rates (no-prefix, premature-idle) so a reviewer can spot
  regressions across retraining runs without opening the server.

Both layers are re-run on every retraining; layer #3's commit-correct rate
is the Phase 2 headline number.

### Open questions

- **τ_pose_confidence** (pose-classifier confidence gate at inference): 0.6 is
  an initial guess; tune after first on-device run.
- **Cross-phase cooldown** after every cycle (commit *or* discard, regardless of
  how many phases the cycle reached) — duration not yet parameterised; see
  Phase 1's open questions.
- **Re-clustering cadence**: always on demand via `POST /train/pose`;
  the training app surfaces a warning (never an auto-trigger) when the
  distribution-shift / recall-drift signals listed in *Post-implementation
  follow-up* trip. Every re-cluster runs the nearest-centroid migration
  of `cluster_kinds`. Cluster-id stability across re-clusters is still
  open (see *Undecided*).
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

5. **Evaluate** Phase 2:
   - *Offline:* run `server/evaluate_pose.py` on the corpus to get
     end-to-end commit-correct rate; treat this as the headline regression
     metric for retraining.
   - *On device:* use the new holds-recognising mode in `CameraView` to
     verify that hold detection, pose classification, and template matching
     behave consistently with the offline numbers; tune `τ_pose_confidence`
     where the on-device confidence distribution disagrees with the offline
     one.
   - Re-verify Phase 1 thresholds on the grown corpus.

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
| `iOS/ModelTraining/.../Views/CameraView.swift` | recognising-mode switch (handfilm / holds); holds-mode overlay surfacing motion-gate state, detected holds, predicted `pose_id`/confidence, `observed_sequence`, matched template |
| `iOS/ModelTraining/.../Views/MetricsView.swift` | new Phase 2 section: pose-MLP per-class precision/recall/F1, confusion matrix, end-to-end commit-correct rate (the "#2 strategy" headline), no-prefix and premature-idle rates, history across retraining runs; re-cluster-suggestion warning badge when any watch-list signal trips (out-of-ε rate > 10 %, per-class recall regression > 5 pp, on-device rejection-rate growth) |
| `server/evaluate_pose.py` (new) | end-to-end Phase 2 corpus evaluation: hold detection → pose argmax → template match → committed gesture vs label; reports commit rate, commit-correct rate, premature-idle rate, length distribution at commit. Output is folded into the metrics payload served by `/model/metrics` (or a sibling `/model/pose/metrics`) so it appears in `MetricsView`. |
| `iOS/ModelTraining/.../ViewModels/` | hold + cluster data fetch (via `/analyze/holds`); read/write corrections |
| `server/ml/trainer_pose.py` (new) | Phase 2 trainer: extract, cluster, apply corrections, train, export manifest |
| `server/ml/preprocessor.js` | (unchanged — Phase 2 uses existing normalised coords) |
| `server/analyze_motion.py` | source of truth for hold-detection + clustering logic used by trainer and `/analyze/holds` |
| `server/data/pose_corrections.json` (new) | reviewer output: `cluster_kinds`, `excluded_holds`; consumed by trainer |
| `server/routers/` | new `/train/pose`, `/model/pose/download`, `/model/pose/manifest`, `/analyze/holds`, `/pose/corrections` (GET/PUT), `/pose/confidence-log` (POST) |

---

## Post-implementation follow-up

Items that cannot be resolved on paper — they need device data, a grown
corpus, or a finished implementation to evaluate. Listed with the metric or
parameter to watch and the action to take if the metric drifts.

### On-device threshold confirmation

- **Phase 1 gate thresholds** (`T_open`, `K_open`, `T_close`). Seeded
  offline. After the gate ships, log `(t_onset, t_open_fired)` and
  `(t_motion_quiet_start, t_close_fired)` per session. If median open-lag
  exceeds 200 ms, lower `T_open` or `K_open`. If the gate stays open well
  after motion stops, lower `T_close`.
- **`τ_pose_confidence`**. Recompute the offline acceptance / conditional-
  accuracy curve from device-collected (confidence, reviewer-label) logs.
  Adjust `τ` to keep conditional accuracy ≥ 0.95 without acceptance
  dropping below ~0.85. The MetricsView Phase 2 panel shows offline and
  device curves side by side.
- **Cooldown** (`cooldown_after_cycle_ms`, default 1000 ms). Watch the
  duplicate-detection rate on device (same gesture firing twice within
  2 s); if non-zero, raise the cooldown.

### Phase 2 metric watch list

- **Idle-while-live-prefix-without-complete-match rate** (from
  `evaluate_pose.py` and on-device logs). If the action's success rate
  falls materially below the headline commit-correct rate, change the
  rule (e.g. always discard rather than commit longest ancestor).
- **Non-modal-exclusion impact** (per-class pose-MLP recall, exclude-vs-
  retain). If excluding non-modal-film holds drops any class's recall
  below 0.90 or by > 5 pp vs the retained policy, switch that class's
  default to *retain* in `pose_corrections.json`.
- **Idle-pose heuristic thresholds**
  (`idle_entropy_threshold`, `idle_tail_position_min`,
  `idle_median_position_min`, `idle_signals_required`). Currently tuned
  for 4 gestures. Re-evaluate at every vocabulary milestone (≥ 8, ≥ 12
  gestures) using the inspector's idle-flag confusion matrix
  (suspected_idle vs reviewer-confirmed `kind`).
- **Re-clustering cadence**. Re-clustering is always **on demand** via
  `POST /train/pose`; never auto-fired. The training app surfaces a
  **re-cluster suggestion** (warning badge in MetricsView and the
  inspector header, naming the tripped signal) when any of these trip:
  - Fraction of newly captured holds whose nearest-centroid distance
    exceeds ε crosses 10 % (primary signal — distribution shift).
  - Per-class pose-MLP recall in `evaluate_pose.py` drops by > 5 pp on
    a previously-stable class between retraining runs with no template
    or data-policy change (secondary — boundary drift).
  - On-device `τ_pose_confidence` rejection rate climbs without a
    matching shift in the offline acceptance curve (secondary —
    runtime distribution drift).
  Acting on the warning runs the re-cluster + nearest-centroid migration
  of `cluster_kinds`. The manual "Re-cluster" action is always
  available, independent of whether any warning is showing.

### Phase 3 follow-up

- **Problem B (length sensitivity)**. Phase 1's motion-gated
  variable-length capture is the primary mitigation — it bounds the
  buffer to the gesture's actual motion arc, keeping clips in the ≥ 30-
  frame regime where the model classifies correctly. Revisit only if
  device logs show accuracy still drops at lengths the gate produces in
  practice. If Phase 1 does not fully close the gap, apply the fallback
  options in *Architecture changes to consider* below.
- **Remaining gesture-to-gesture misclassification**. Re-evaluate after
  the post-PR-#34/#37 retrain reaches the device. The Phase 3 confusion
  matrix in MetricsView is the watch surface.
- **`thumbs_up` capture retake**. Decide after seeing post-Phase-2
  per-class commit-correct rate. If `thumbs_up` lags the others by more
  than 10 pp, retake or retire.

### Architecture changes to consider

Non-routine improvements that require deliberate implementation effort,
as distinct from threshold tuning and metric monitoring. Listed in
increasing complexity.

- **Duration feature** *(Phase 3 Problem B fallback, try first if Phase 1
  is insufficient)*. Append one scalar — real-frame count normalised by
  `temporalWindow` — to the 256-dim summary feature vector fed to the
  Phase 3 MLP. Gives the model an explicit signal about how much of the
  motion arc was observed. Cheapest change; no data re-collection;
  compose with time-warp augmentation if the gap persists.
- **Time-warp augmentation** *(Phase 3 Problem B fallback, try second)*.
  During training, resample each clip at 0.7×–1.4× to synthesise
  faster/slower variants before extracting summary features. Teaches
  speed invariance across the 18–30-frame band. Requires changes to
  the `trainer_rf_mlp.py` preprocessing step; inference path is
  unchanged.
- **LSTM trainer** *(Phase 2, add only when the trigger condition fires)*.
  Implement only when a future gesture's modal template has length ≥ 2
  *regular* poses (not counting idle) **and** the single-frame MLP
  cannot achieve ≥ 0.95 per-pose recall on that gesture. Current corpus
  does not meet this condition (all templates are length-1 regular pose).
- **Gesture set reassembly**. At a natural milestone (e.g. before
  expanding the vocabulary beyond the current four), review each gesture
  for utility, distinctiveness, and training-data quality. Retire
  gestures whose per-class Phase 2 commit-correct rate or Phase 3
  accuracy consistently lags the others by a meaningful margin and for
  which recapture is not worth the effort; replace them with better-
  defined alternatives if needed. `thumbs_up` is the current candidate
  to retire: its training films were shot from inconsistent camera
  angles, fragmenting its pose cluster across ~20 ids at ε = 0.8, and
  its Phase 3 accuracy at trim=30 already shows variance. No action
  until post-Phase-2 per-class numbers exist.

### Open implementation-shape questions

- Whether device-side confidence logs need their own upload endpoint or
  can ride on the existing film-upload path.

---

## Undecided

Items that are still open. Covers both questions unresolved on paper
and items deferred to data, device experience, or future
implementation. For deferred items, the linked body section
(*Post-implementation follow-up*, *Architecture changes to consider*)
is the source of truth; the entry here serves as an index.

- **Phase 2 cluster-id stability across re-clusters.** Cadence is decided
  (always on demand; the watch-list signals only raise a warning, never
  auto-fire — see *Post-implementation follow-up*). The nearest-centroid
  `cluster_kinds` migration is sketched. What is *not* decided: whether
  re-clusters preserve stable cluster ids (and the migration only adjusts
  membership), or accept renumbering and rewrite `pose_corrections.json`
  every time.

- **Cluster-kinds migration edge cases.** Nearest-centroid migration is
  named but its edge cases aren't pinned down. What is *not* decided:
  the distance threshold past which a new cluster has no usable ancestor
  and should default to `unconfirmed` rather than inherit a `kind`;
  behaviour when an old cluster has no successor in the new clustering
  (silently drop the correction, or surface it in the inspector as a
  "lost review"); behaviour when multiple new clusters map to the same
  old one (all inherit, largest only, or all reset to `unconfirmed`).

- **`excluded_holds` survival across parameter changes.** Per-hold
  exclusions in `pose_corrections.json` are keyed by `(film_id,
  hold_ordinal)`. If `T_hold`, `K_hold`, or `smooth_k` change, the set
  and ordering of detected holds shifts and the keys may no longer
  refer to the intended hold. What is *not* decided: whether to
  fingerprint exclusions by approximate frame range and re-match on
  parameter change, drop all exclusions whenever detection parameters
  change, or keep ordinals as-is and accept off-by-one drift.

- **Cooldown semantics.** `cooldown_after_cycle_ms` (default 1000 ms)
  is a parameter, but its scope is not specified. What is *not*
  decided: whether the motion gate is forced closed for the full
  cooldown duration or only gesture emission is suppressed while the
  gate operates normally; when the in-view frame buffer is cleared
  (start of cooldown vs. end); whether cooldown starts at
  gesture-emit time, gate-close time, or Phase-3-decision time
  (relevant when Phase 3 takes meaningful time to run).

- **Gate-open buffer overflow behaviour.** Plan says the variable-length
  buffer is "naturally bounded to ≤ 1 s by the existing
  `temporalWindow`" but doesn't say how. What is *not* decided:
  whether the buffer is a 30-frame ring (keep the most recent 30
  in-view frames, dropping older ones if the gate stays open longer),
  a hard cap that stops accepting frames after 30, or a sliding window
  re-anchored on detected hold boundaries. Affects what Phase 3 sees
  for gestures whose motion arc exceeds the cap.

- **Phase 3 output restriction to candidate set S.** Plan says "output
  restricted to S" without specifying the mechanism. What is *not*
  decided: argmax over masked softmax (zero out non-S logits before
  argmax), renormalised softmax over S only (probabilities sum to 1
  within S), or top-1-of-full-output-must-be-in-S-else-reject.
  Borderline cases — e.g., true class slightly below an out-of-S
  distractor — split differently across the three.

- **Phase 3 confidence threshold.** Plan mentions "low confidence →
  discard" but no parameter name, value, or tuning method, even though
  these are spelled out exhaustively for `τ_pose_confidence`. What is
  *not* decided: the parameter name (e.g., `τ_phase3_confidence`), an
  initial value, and whether tuning follows the same offline-curve
  plus on-device-log pattern used for the pose threshold.

- **`thumbs_up` interim template policy.** `thumbs_up` fragments
  across ~20 clusters at ε = 0.8 because of inconsistent capture
  angles. The retire-or-recapture decision is deferred until
  post-Phase-2 per-class numbers (see *Architecture changes to
  consider*). What is *not* decided: what the manifest does for
  `thumbs_up` in the meantime — ship a length-1 modal-cluster template
  (matches only the modal cluster's holds, leaves the other ~19
  unmatched), exclude `thumbs_up` from `gesture_templates` entirely
  until recapture (no commits possible), or allow OR-of-clusters
  templates (multiple cluster ids treated as the same logical pose).
  Affects the headline commit-correct rate Phase 2 can report before
  recapture.

- **Phase 3 remaining gesture-to-gesture misclassification.**
  Deferred to data. Re-evaluate after the post-PR-#34/#37 retrain
  reaches the device; the Phase 3 confusion matrix in MetricsView is
  the watch surface. Tracked in *Post-implementation follow-up*.

- **Gesture set reassembly.** Deferred to data. At a vocabulary
  milestone, review each gesture for utility, distinctiveness, and
  training quality; retire underperformers (current candidate:
  `thumbs_up`). Decision waits for post-Phase-2 per-class
  commit-correct numbers. Tracked in *Architecture changes to
  consider*.

- **LSTM trainer trigger condition.** Deferred to data. Sketched as
  "future gesture's modal template has length ≥ 2 *regular* poses
  (not counting idle) AND single-frame MLP cannot achieve ≥ 0.95
  per-pose recall on that gesture"; current corpus does not exercise
  this condition, so the rule is not yet committed. Tracked in
  *Architecture changes to consider*.
