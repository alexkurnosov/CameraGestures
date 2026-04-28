# 3-Phase Recognition — Implementation Stages

Companion to [PLAN_3phase_recognition.md](PLAN_3phase_recognition.md). That document is the
*design* (decisions, parameters, runtime flows). This document is the *implementation
order* — each stage broken into substages, with auto and manual tests and an exit
criterion.

Cross-reference convention: design-plan section/line numbers appear as
`(plan §Section)` or `(plan:NNN)`.

## Conventions

- **Auto tests** run from CLI without device. Server: extend `server/test_api.sh`
  (bash; `run_test NAME EXPECTED_STATUS METHOD URL [BODY]`) or add Python tests
  next to `server/ml/test_none_class.py`. Pure-Swift logic that does not touch
  AVFoundation can ship with an XCTest target if one is added; otherwise it is
  manual.
- **Manual tests** are observable steps on a real device with the ModelTraining
  app (build in Xcode, do not invoke `xcodebuild` from this plan).
- Each substage has an explicit exit criterion. A stage is "done" only when every
  substage's auto + manual tests pass.
- Stages are listed in dependency order. Stages with no dependency arrow can be
  parallelised against the previous stage.
- **Metrics snapshots**: on completing each stage, save any CLI evaluation output
  to `Metrics/3phase/stage<N>_<description>.txt` (e.g.
  `Metrics/3phase/stage0_trim_eval.txt`). These are plain-text reference
  artifacts — not code — used to compare baselines across stages as the corpus
  and pipeline evolve.

---

## Stage 0 — Phase 3 operational follow-up *(no new code)*

Pending operational work from the already-shipped Phase 3 fix (PR #34, #37). Done
before Phase 2 work begins so the Phase 3 baseline is trustworthy when Phase 2
output starts feeding into it.

### Substages

- **0.1** Retrain Phase 3 MLP on production server with the length-invariant
  preprocessor (no data changes — re-run `trainer_rf_mlp.py`).
- **0.2** Distribute new `.tflite` to device (existing `/model/download` path).
- **0.3** On-device verification — run the existing handfilm-recognising mode on
  each of the four trained gestures.

### Tests

| | What | How |
|---|---|---|
| auto | trim=30 accuracy holds | `python server/evaluate_trimmed.py --trim 30` returns ≥ 0.95 on the held-out fold (matches plan §Phase 3 Status) |
| auto | model build deterministic | re-run training with the same seed; tflite sha256 matches the prior run modulo oneDNN jitter |
| manual | on-device matches eval | each of `ok`, `stop`, `point_left`, `thumbs_up` recognised correctly across 5 deliberate captures each (≥ 18/20 pass) |

### Exit criterion
On-device per-gesture recognition rate ≥ 90 % on the four trained gestures. Any
class < 90 % is flagged for retake before moving on (plan §Phase 3 Required
follow-up).

---

## Stage 1 — Server foundations for Phase 1 + Phase 2

Pure server work. Output is consumed by every downstream stage.

### Substages

- **1.1** Extend `server/analyze_motion.py` with a `--calibrate` mode that emits
  seed values for `T_open`, `K_open`, `T_close`, `K_close` (plan §Phase 1
  Threshold calibration), reusing the per-film energy traces and idle-candidate
  cluster identification already produced for Phase 2 calibration.
- **1.2** Define `server/data/pose_corrections.json` schema and add a loader
  module that:
  - returns an empty-but-valid object when the file is missing,
  - exposes `cluster_kinds` and `excluded_holds` lookups,
  - applies the coarse pre-heuristic fallback for unconfirmed clusters
    (≥ 3 gestures contributing, each ≥ 5 % — plan:130–133).
- **1.3** Add a `params_hash` utility — canonical-JSON sha256 of
  `{T_hold, K_hold, smooth_k, edge_trim_fraction}` (plan:710–718). Surfaced in
  `server/ml/__init__.py` for trainer + corrections-migration use.

### Tests

| | What | How |
|---|---|---|
| auto | calibration produces all four gate seeds | new `server/ml/test_calibration.py` — synthetic energy trace with known onset distribution and idle run; assert returned `T_open`, `K_open`, `T_close`, `K_close` are within ±5 % of analytical expectation |
| auto | calibration handles MediaPipe duplicate zeros | synthetic trace with 35 % zero frames; assert `T_open` not pinned to 0 |
| auto | corrections loader missing file → empty | call loader on tmp dir; assert returns `{cluster_kinds: {}, excluded_holds: []}` |
| auto | corrections loader malformed JSON → raise | assert raises with a clear message |
| auto | params_hash stability | `params_hash(params)` invariant under key reorder; changes when any of the four fields change |
| manual | corpus calibration sanity | run `--calibrate` on the 151-film corpus; values land inside the bands documented in plan §Phase 1 Threshold calibration |

### Exit criterion
`analyze_motion.py --calibrate` writes a calibration report including the four
gate seeds; corrections loader and `params_hash` are unit-tested and importable
from `server.ml`.

---

## Stage 2 — Hold inspection API + per-film overlay

Server endpoint and the simpler half of the iOS inspector. Lets a human eyeball
hold detection on real films before any clustering or training is wired up.

Depends on: Stage 1.

### Substages

- **2.1** Refactor hold-detection logic out of `analyze_motion.py` into a
  reusable module so it can be called both from the trainer and from a request
  handler (plan §Phase 2 Server-side changes).
- **2.2** Add `POST /analyze/holds` to `server/routers/`. Body: `HandFilm` JSON.
  Response: `{holds: [{ordinal, start_frame, end_frame, rep_frame}], params_hash}`.
- **2.3** iOS — extend `HandFilmsView.swift` with a hold-timeline overlay.
  Calls `/analyze/holds`, draws coloured bands per hold and a marker on each
  representative frame. Each thumbnail labelled with hold ordinal and frame
  index.
- **2.4** iOS — per-hold "Exclude this hold" action that issues a corrections
  update. The PUT endpoint itself ships in Stage 5; in this stage, wire the
  button to a stub that logs the intended payload so the UI is testable.

### Tests

| | What | How |
|---|---|---|
| auto | `/analyze/holds` returns plausible structure | `server/test_api.sh` — POST a known fixture film, expect 200 + ≥ 1 hold, every `rep_frame ∈ [start_frame, end_frame]` |
| auto | `params_hash` round-trips | response `params_hash` matches what `params_hash()` returns for the configured detector params |
| auto | bad input → 422 | empty `frames`, missing fields |
| manual | timeline overlay aligns with motion | open a known film, scrub timeline, hand visibly stationary inside each coloured band |
| manual | rep-frame thumbnails are key poses | thumbnails look like meaningful poses, not transitions |

### Exit criterion
A reviewer can open any film in `HandFilmsView`, see hold bands and rep-frame
thumbnails, and click "exclude" without crashing. Server endpoint covered in
`test_api.sh`.

---

## Stage 3 — Phase 1 motion gate on iOS

Standalone — no Phase 2 dependency. Ship in parallel with Stage 2 if engineering
capacity allows. Depends on Stage 1 only for seed thresholds.

### Substages

- **3.1** Per-frame raw motion energy in `HandGestureRecognizing.swift`: L2 sum
  over consecutive-frame normalised landmark deltas. Exposed as a single value
  per `HandShot` (plan §Phase 1 Signal). Designated as the shared raw signal
  consumed by both Phase 1 and Phase 2's hold detector (plan:192–198).
- **3.2** Hysteresis state machine: states `closed | open`, parameters
  `T_open`, `K_open`, `T_close`, `K_close`, `cooldown_after_cycle_ms` exposed in
  the runtime configuration (plan:144–146). `Types.swift` gains the parameter
  struct; `HandGestureRecognizing` consumes it.
- **3.3** Buffer accumulation only when gate is open. Absent frames reset the
  gate immediately (plan:158–160). Buffer hard-capped at 30 frames (plan:617).
- **3.4** Post-cycle cooldown with the queueing behaviour from
  plan §Phase 1 Decisions: cooldown starts at cycle-end, only emission is
  suppressed during the window, the most recent commit during cooldown wins,
  buffer is cleared at cooldown start.
- **3.5** Holds-mode UI surface in `CameraView.swift`: gate-state indicator
  (open/closed) and current buffer length. The pose-id / observed-sequence /
  template-match overlays come in Stage 7; the gate indicator is independent.

### Tests

| | What | How |
|---|---|---|
| auto | energy on synthetic stream | unit test (XCTest target if one is added; otherwise a CLI test in `iOS/HandGestureRecognizing/test`): synthetic `HandShot` sequence with known landmark deltas → expected energy values |
| auto | hysteresis transitions | drive state machine with crafted energy traces → assert state and event sequence (open after `K_open` over `T_open`; close after `K_close` under `T_close`; absent frame forces close) |
| auto | cooldown queueing | inject commit during cooldown → assert it fires once, at cooldown end |
| manual | gate opens on motion | hand still in view → indicator stays closed; flick fingers → opens within ~200 ms |
| manual | gate closes on idle | relax hand → closes within ~300 ms (`K_close` ≈ 9 frames) |
| manual | absent-frame reset | open gate, move hand out of frame → indicator returns to closed immediately |
| manual | cooldown suppresses duplicates | repeat the same gesture twice within 1 s → only one emission |

### Exit criterion
Gate opens promptly on motion onset, closes within `K_close` after motion
stops, resets on absent frame, and the cooldown window suppresses the
duplicate-emission case that motivated Phase 1 in the first place.

---

## Stage 4 — Pose trainer scaffolding

Server-side trainer that builds the pose model from clustering alone, before any
inspector corrections exist. Output is consumed by Stage 7 (iOS runtime). Uses
the coarse pre-heuristic fallback (Stage 1.2) so something runnable exists
before reviewers populate corrections.

Depends on: Stage 1.

### Substages

- **4.1** Create `server/ml/trainer_pose.py`. Steps 1–4 of plan §Server-side
  changes: extract holds, cluster (agglomerative, ε = 0.8), apply coarse
  fallback for unconfirmed clusters, run idle-pose heuristic to compute
  `suspected_idle`.
- **4.2** Manifest builder: emit `pose_manifest.json` per the schema in
  plan §Template manifest. `cluster_kinds` defaulted from the coarse fallback;
  `gesture_templates` built from the modal sequence per gesture, with
  `suspected_idle` clusters stripped (so the first run produces length-1
  templates on the current corpus).
- **4.3** Train the single-frame MLP with class-weighted cross-entropy
  (`weights ∝ 1/sqrt(n_samples_per_cluster)` — plan:454–458). Train/test split
  by `session_id` (plan:451–453).
- **4.4** Export `pose_model.tflite` and bundle it with `pose_manifest.json` in
  the same artifact directory as the existing handfilm model.
- **4.5** New endpoints in `server/routers/`: `POST /train/pose`,
  `GET /model/pose/download`, `GET /model/pose/manifest`.

### Tests

| | What | How |
|---|---|---|
| auto | clustering deterministic | `server/ml/test_pose_trainer.py` — fixed-seed run on a small fixture corpus → expected cluster count and modal-template assignment |
| auto | idle heuristic flags as documented | crafted clusters with controlled gesture-entropy / tail-position / median-position → assert `suspected_idle` triggers iff `idle_signals_required` (= 2) signals fire |
| auto | manifest schema | jsonschema validation against the schema in plan §Template manifest |
| auto | per-class accuracy | held-out per-class precision/recall reported; primary watch number is per-class recall ≥ 0.85 on the 4-class corpus |
| auto | endpoints | `test_api.sh` — POST `/train/pose`, GET `/model/pose/download` returns binary > 1 KB, GET `/model/pose/manifest` returns valid JSON |
| manual | cluster-id 15 (or equivalent) flagged suspected_idle | run on the 151-film corpus; the dominant idle cluster (n ≈ 131 holds) carries `suspected_idle: true` and matches plan §Calibration observation |

### Exit criterion
`POST /train/pose` produces a `.tflite` plus a manifest whose templates and
cluster kinds match the plan §Calibration observation expectations. Per-class
recall ≥ 0.85.

---

## Stage 5 — Cluster inspector + corrections round-trip

iOS per-cluster review surface and the server PUT/GET endpoints that persist
human decisions. After this stage, corrections drive future trainer runs.

Depends on: Stages 2, 4.

### Substages

- **5.1** Add `GET /pose/corrections` and `PUT /pose/corrections` to
  `server/routers/`. Body shape is the file format in plan §Corrections
  artefact. Server validates against the current manifest's cluster ids.
- **5.2** New `iOS/ModelTraining/.../Views/PoseInspectorView.swift`. Pulls the
  manifest and the corrections file; renders a per-cluster grid grouped by
  cluster id, each thumbnail sorted by distance from centroid. Cluster header
  shows id, size, gesture composition, heuristic signals, current `kind`.
- **5.3** Per-cluster actions: *Mark as idle / regular / unconfirmed*. Each
  action issues a `PUT /pose/corrections` and refreshes the view. Clusters with
  `suspected_idle: true` are visually distinct.
- **5.4** Wire the per-hold "Exclude this hold" action from Stage 2.4 to the
  PUT endpoint; record the full `excluded_holds` entry including
  `params_hash`.
- **5.5** Re-cluster suggestion warning badge in `MetricsView.swift` — fires
  when any of the three watch-list signals from plan §Re-clustering cadence
  trip. The badge is informational; the manual "Re-cluster" trigger is always
  available.

### Tests

| | What | How |
|---|---|---|
| auto | corrections round-trip | `test_api.sh` — PUT `cluster_kinds: {15: idle}`, GET → matches; PUT excluded hold, GET → entry present with `params_hash` |
| auto | invalid cluster id rejected | PUT `cluster_kinds: {99999: idle}` (id not in manifest) → 422 |
| auto | re-cluster warning trips on synthetic distribution shift | unit test the watch-list logic with a synthetic histogram → badge fires when ≥ 10 % of holds exceed ε |
| manual | cluster grid renders | open `PoseInspectorView`, see grid for every cluster, thumbnails sorted by distance from centroid, suspected_idle cluster visually flagged |
| manual | mark cluster as idle | mark cluster 15 idle, refresh view → state preserved; rebuild trainer (Stage 6) → cluster 15 stripped from `gesture_templates` |
| manual | per-hold exclusion survives reload | exclude a hold, kill app, reopen → exclusion shows in UI |

### Exit criterion
A reviewer can confirm cluster kinds and exclude individual holds; both
decisions persist; corrections file is well-formed and validated against the
manifest.

---

## Stage 6 — Pose trainer integrates corrections

Trainer reads `pose_corrections.json` and applies cluster kinds and per-hold
exclusions. After this stage, trainer output fully reflects human review.

Depends on: Stage 5.

### Substages

- **6.1** Trainer reads `pose_corrections.json`; merges `cluster_kinds` over
  the coarse fallback (explicit overrides win, unconfirmed remains the default
  for unlisted clusters — plan:702–706).
- **6.2** Per-hold exclusion application: fast path when `params_hash`
  matches; otherwise migrate against the freshly detected hold set
  (clean / split / merge / lost — plan:708–741). For this stage, ship the
  fast path only and surface a warning if `params_hash` mismatches; the full
  migration ships in Stage 10.
- **6.3** Manifest re-emit with the corrected kinds and templates; idle-marked
  clusters move to `idle_poses` and disappear from `gesture_templates`.

### Tests

| | What | How |
|---|---|---|
| auto | cluster_kinds override fallback | fixture: corrections marks cluster A as `regular` despite coarse fallback flagging it idle; assert manifest reflects override |
| auto | excluded hold dropped from training set | fixture corrections file with one excluded hold; assert that hold's frame index is absent from the trainer's labelled hold list |
| auto | params_hash mismatch warning | invoke trainer with a corrections file whose `params_hash` does not match current params; assert warning emitted and exclusion is *not* applied (Stage 10 fixes this) |
| auto | idle stripped from templates | mark a cluster idle in corrections; assert the cluster id appears in `idle_poses` and is absent from every `gesture_templates` value |
| manual | end-to-end review loop | reviewer marks cluster 15 idle in inspector → triggers `POST /train/pose` → downloaded manifest has length-1 templates for `ok`, `stop`, `point_left`; cluster 15 in `idle_poses` |

### Exit criterion
Manifest produced by `POST /train/pose` matches the human review state in the
inspector. Re-running training without inspector changes is a no-op
(deterministic rebuild).

---

## Stage 7 — iOS pose runtime

Wires Phase 2 into the live recognition pipeline. After this stage, the device
runs the full Phase 1 + Phase 2 path; Phase 3 still runs as before but its
input is now restricted to the candidate set S produced by Phase 2.

Depends on: Stages 3, 4, 6.

### Substages

- **7.1** `GestureModel` gains a single-frame prediction path for the pose
  classifier (existing path is handfilm-oriented). Loads the second model + its
  manifest from the app's model directory.
- **7.2** Hold detection in `HandGestureRecognizing.swift`: smoothed motion
  energy (`smooth_k = 3`), hold defined as a run with energy < `T_hold` for
  ≥ `K_hold`, representative frame at the run's argmin (plan §Phase 2 §Key-pose
  extraction pipeline — same logic as offline calibration). Note: `edge_trim` is
  *training-only* and not applied at runtime (plan:387).
- **7.3** `observed_sequence: [pose_id]` buffer + prefix matcher against
  `gesture_templates` from the manifest. `T_commit` (300 ms) timer kicks off
  when a hold ends with a complete prefix and longer prefixes are still
  possible (plan §Phase 2 Runtime flow).
- **7.4** Idle-pose commit logic: when a hold's pose has `kind == idle`, apply
  the runtime-flow rules (plan:299–304) — reset on empty observed, commit on
  complete match, longest-complete-ancestor or discard on live-prefix-without-
  match.
- **7.5** `T_min_buffer` (200 ms ≈ 6 frames) deferral on every commit signal
  (plan:350–370).
- **7.6** Phase 3 output restriction to S: masked argmax with pre-mask
  unrenormalised softmax confidence as the value compared against
  `τ_phase3_confidence` (plan §Phase 3 Output restriction).
- **7.7** Recognising-mode switch in `CameraView.swift`: *Handfilm* (existing)
  and *Holds* (new). Holds-mode overlay surfaces detected holds, predicted
  `pose_id` + confidence + cluster kind, current `observed_sequence`, matched
  template (or "no match") (plan:622–634).

### Tests

| | What | How |
|---|---|---|
| auto | single-frame predict | `GestureModel` test (XCTest if available, else CLI) — feed a known 63-dim landmark vector, assert returned `pose_id` matches the trained label |
| auto | hold detection on synthetic energy | drive the detector with a crafted energy trace, assert hold start / end / rep frame |
| auto | prefix matcher commit cases | synthetic `(observed, manifest)` pairs cover: complete + no longer prefix → commit; complete + longer prefix possible → start `T_commit`; no prefix → discard; idle on empty observed → reset; idle on live prefix → longest-ancestor-or-discard |
| auto | masked argmax + confidence | given `(softmax, S)` fixtures, assert masked-argmax class and that reported confidence is the pre-mask probability |
| auto | T_min_buffer deferral | commit signal fires at `t < T_min_buffer` → assert the commit is deferred and re-evaluated at `T_min_buffer` |
| manual | gesture commits on idle release | perform `ok` and relax → fires within ~300 ms after relax |
| manual | gesture commits on T_commit | perform `ok` and freeze on the gesture pose → fires `T_commit` ≈ 300 ms after the hold |
| manual | gesture commits on gate-close | perform `ok` and exit view immediately → fires once, gate-close path |
| manual | unknown pose discards | perform a non-trained shape → no commit, holds-mode overlay shows "no match" |
| manual | holds-mode overlay matches reality | overlay reads a coherent `observed_sequence` and a sensible `template match` for each of the 4 trained gestures |

### Exit criterion
All four trained gestures recognise correctly under each of the three commit
paths (idle, `T_commit`, gate-close). The "Holds" recognising mode shows
correct telemetry on every detected hold. The "Handfilm" mode continues to
work.

---

## Stage 8 — Phase 2 evaluation harness

Offline regression baseline that lets every retraining run be compared against
the last. Required before τ tuning and migration tooling.

Depends on: Stage 6.

### Substages

- **8.1** Add `server/evaluate_pose.py`. Treat each labelled film as one
  gate-open from first to last in-view frame (no Phase 1 simulation — that's
  Stage 11). Runs hold detection → pose argmax → idle filtering → prefix match
  → commit. Reports the metric set in plan §Evaluation §Layer 3, including the
  idle-while-live-prefix rate and the non-modal-exclusion impact.
- **8.2** Wire layers #2 (per-class precision/recall, confusion matrix) and #3
  (commit-correct rate, no-prefix rate, premature-idle rate, length
  distribution at commit by signal, idle-while-live-prefix breakdown) into the
  metrics payload served by `/model/metrics` (or sibling `/model/pose/metrics`).
- **8.3** Add a Phase 2 section to `MetricsView.swift`: per-class
  precision/recall/F1, confusion matrix, headline commit-correct rate, and the
  auxiliary rates (no-prefix, premature-idle, idle-while-live-prefix); show
  history across retraining runs.

### Tests

| | What | How |
|---|---|---|
| auto | evaluate_pose.py runs cleanly | `python server/evaluate_pose.py` on the test corpus exits 0 and emits a JSON report containing every key documented in plan §Layer 3 |
| auto | commit-correct on the current corpus | report's headline commit-correct rate ≥ 0.85 (initial bar; raise after device data) |
| auto | metrics endpoint payload schema | `test_api.sh` — GET `/model/metrics` returns Phase 2 fields with the documented shape |
| manual | MetricsView Phase 2 panel | open MetricsView; Phase 2 section renders all metrics; history chart shows at least the last training run |

### Exit criterion
A single command produces the Phase 2 headline number on the corpus and the
same number is visible in MetricsView.

---

## Stage 9 — Confidence logging + τ tuning

Closes the loop on `τ_pose_confidence` (Phase 2) and `τ_phase3_confidence`
(Phase 3) — offline sweep + on-device divergence watch.

Depends on: Stages 7, 8.

### Substages

- **9.1** Add `POST /confidence-log` to `server/routers/`. Accepts batched
  entries with the `phase` discriminator (`pose | phase3`) and the per-phase
  fields documented in plan §Server-side changes and plan §Phase 3 Confidence
  threshold.
- **9.2** iOS — confidence log emission. In Holds mode, every hold logs a pose
  entry (`predicted_pose_id`, `confidence`, optional reviewer label).
  In handfilm mode, every committed cycle logs a phase3 entry
  (`candidate_set_size`, `predicted_class`, `confidence`, optional reviewer
  label). Reviewer-labelling UI is in Holds mode and on the
  post-commit confirmation overlay.
- **9.3** Server-side τ-sweep: trainer emits the offline acceptance /
  conditional-accuracy curve into the metrics payload for both phases. Server
  recomputes the on-device curve from received logs and surfaces both in
  MetricsView.

### Tests

| | What | How |
|---|---|---|
| auto | confidence-log POST round-trip | `test_api.sh` — submit a batch, assert persistence and correct phase routing |
| auto | offline sweep emitted | trainer writes `confidence_curve_pose` and `confidence_curve_phase3` to the metrics payload |
| auto | bad phase value rejected | POST `phase: "bogus"` → 422 |
| manual | reviewer tagging works | tag 10 holds in Holds mode → server shows 10 entries; tag a phase3 commit → 1 entry |
| manual | divergence visible in MetricsView | offline and on-device curves render side-by-side for both phases |

### Exit criterion
Both τ values can be tuned from MetricsView curves; offline and on-device
curves agree to within ±0.05 acceptance for the trained gestures.

---

## Stage 10 — Migration tooling

Per-hold exclusion migration (Stage 6.2 fast-path → full migration) and
cluster-id migration on re-cluster.

Depends on: Stage 6.

### Substages

- **10.1** Per-hold exclusion migration in `trainer_pose.py`: clean / split /
  merge / lost cases per plan:719–741. Each case emits a per-exclusion report
  entry. Inspector queues for `split`, `merge`, `lost` are added to
  `PoseInspectorView`.
- **10.2** Cluster-id stability across re-clusters: nearest-centroid
  inheritance with `d_inherit = ε`. Implements the four edge-case rules from
  plan §Re-clustering cadence (inheritance threshold, lost reviews, split,
  merge). Genuinely new clusters mint fresh ids from a monotonic counter.
- **10.3** Migration report exposed in MetricsView: per-cluster
  `(old_id, new_id, distance, inherited?)`; high-distance-but-inherited and
  lost-review queues surface in the inspector.
- **10.4** Bootstrap stability sanity check: 20× resample (90 % each), record
  centroid drift for stable clusters, verify p95 drift < ε. Run as part of the
  first calibration pass and on parameter changes.

### Tests

| | What | How |
|---|---|---|
| auto | exclusion migration cases | unit tests in `test_pose_trainer.py` for each of clean / split / merge / lost; assert correct outcome and report-entry shape |
| auto | inheritance threshold | synthetic re-cluster: nearest centroid at distance < ε → inherited; at distance > ε → defaults `unconfirmed` |
| auto | merge defaults to unconfirmed | two old clusters within ε of one new centroid → new centroid is `unconfirmed` regardless of old kinds |
| auto | bootstrap stability | unit test: synthetic stable distribution → p95 drift < ε; synthetic unstable distribution → p95 drift ≥ ε and a clear warning |
| manual | inspector surfaces queues | trigger a re-cluster after changing `T_hold`; review split / merge / lost queues populate; reconfirming any entry round-trips through `pose_corrections.json` |

### Exit criterion
A parameter change (any of `T_hold`, `K_hold`, `smooth_k`, `edge_trim`)
followed by re-training does not silently lose any prior corrections; every
ambiguous case appears in the inspector's review queues.

---

## Stage 11 — End-to-end pipeline evaluation

Full 1+2+3 regression baseline. Depends on every prior stage being merged.

Depends on: Stages 3, 7, 8.

### Substages

- **11.1** Add `server/evaluate_pipeline.py`. Simulates Phase 1's gate from
  each film's energy trace using the calibrated `T_open`/`K_open`/`T_close`/
  `K_close`, then runs Phase 2 + Phase 3 on the resulting buffer. Emits the
  metric set in plan §Evaluation §Layer 4: gate-open rate, gate-miss rate,
  1+2 sub-pipeline commit-correct, 1+2+3 commit-correct, Phase 3 lift,
  gate-trim impact vs `evaluate_pose.py`.
- **11.2** Wire Layer #4 metrics into `/model/pipeline/metrics` (separate
  endpoint per plan §Evaluation §Layer 4 — line 830).
- **11.3** New MetricsView panel: pipeline metrics, with the gate-trim impact
  highlighted (red if > 5 pp, plan:1217).

### Tests

| | What | How |
|---|---|---|
| auto | evaluate_pipeline.py runs cleanly | `python server/evaluate_pipeline.py` on the corpus exits 0 and emits a JSON report containing every key documented in plan §Layer 4 |
| auto | gate-trim impact bounded | difference between `evaluate_pose.py` commit-correct and `evaluate_pipeline.py`'s 1+2 commit-correct is < 5 pp on the calibrated thresholds |
| auto | Phase 3 lift non-negative | `1+2+3 commit-correct ≥ 1+2 commit-correct − 0.5 pp` (small jitter allowed) |
| auto | metrics endpoint payload | `test_api.sh` — GET `/model/pipeline/metrics` returns Layer 4 fields |
| manual | MetricsView pipeline panel | open MetricsView; pipeline panel renders all metrics; gate-trim warning behaves correctly on synthetic large gap |

### Exit criterion
Layer #4 baseline is reproducible on demand and surfaced in MetricsView.
Re-runs after every threshold or model change become routine.

---

## Stage dependency graph

```
Stage 0 ── (independent)

Stage 1 ──┬─→ Stage 2 ──┐
          │             │
          └─→ Stage 4 ──┴─→ Stage 5 ─→ Stage 6 ─→ Stage 7 ─→ Stage 9
                                       │             │
                                       └─→ Stage 8 ──┴─→ Stage 11
                                       │
                                       └─→ Stage 10

Stage 3 ──→ Stage 7 (also feeds Stage 11)
```

Stages 2 and 3 can run in parallel after Stage 1. Stages 8 and 10 can run in
parallel after Stage 6.

## Per-stage acceptance summary

| Stage | Headline exit metric |
|---|---|
| 0 | On-device per-gesture rate ≥ 90 % on the four gestures |
| 1 | `--calibrate` produces all four gate seeds; loader + hash unit-tested |
| 2 | `/analyze/holds` covered in `test_api.sh`; HandFilmsView overlay renders |
| 3 | Gate opens / closes / resets correctly; cooldown suppresses duplicates |
| 4 | `POST /train/pose` produces tflite + manifest; per-class recall ≥ 0.85 |
| 5 | Cluster + per-hold corrections persist and survive reload |
| 6 | Manifest reflects inspector state; deterministic rebuild |
| 7 | All four gestures recognise under all three commit paths |
| 8 | Phase 2 commit-correct ≥ 0.85 on corpus; visible in MetricsView |
| 9 | Both τ tunable from MetricsView curves; offline ↔ on-device agree |
| 10 | Parameter changes never silently drop corrections |
| 11 | Layer #4 baseline reproducible; visible in MetricsView |

## Open questions deferred to implementation time

- Whether iOS adds a real XCTest target (currently no `*Tests` folder under
  `iOS/`). If yes, every "auto" iOS test in this plan moves from manual to
  XCTest. If no, those tests stay manual.
- Whether `evaluate_pose.py` and `evaluate_pipeline.py` should share a CLI
  driver. Decide after both ship.
