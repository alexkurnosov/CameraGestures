# Session fixtures

Real `.cgsession` bundles (format version 1) for the Training App's Sessions tab
and for the reader regression test (`SessionFixtures.*` in `test_session.cpp`).
They hold hand landmarks only, no images or video.

| Bundle | Content |
|---|---|
| `corpus_holds.cgsession` | 20 phase3 films (first 4 of each gesture) end to end, one hand, holds mode: 605 shots, 59 s |
| `corpus_two_hand.cgsession` | the same films two at a time as hands 0 and 1: 605 shots on tracks 0 / 1 / absent, 29 s |

Both come from export snapshot `20260927T095311Z_a45264cc` and its models; the
manifests carry the model hashes and the film names.

They are derived data. Rebuild them with

```bash
core/tests/fixtures/sessions/regenerate.sh [<snapshot dir>] [<replay_rig>]
```

The same snapshot and library build give the same bundles, apart from
`created_at` / `stopped_at`. After a library change the telemetry may differ;
commit the regenerated bundles with that change.

In the two-hand bundle, absent frames from either film are recorded as absent
frames, which a real device would not do while the other hand is in view.
