#!/bin/bash
# Regenerates the committed .cgsession fixtures from an export snapshot.
#
# Usage: regenerate.sh [<snapshot dir>] [<replay_rig binary>]
#   snapshot  default: the 2026-09-27 snapshot the fixtures were made from
#   rig       default: core/build-rig/tests/replay_rig (TFLite build, see
#             docs/data-export/snapshot-replay.md in the server repo)
#
# Takes the first 4 films (by file name) of each phase3 gesture and writes
#   corpus_holds.cgsession     the films end to end, one hand (track 0)
#   corpus_two_hand.cgsession  the same films two at a time, as hands 0 and 1
# both in holds mode. Same snapshot + same library build = the same bundles,
# apart from created_at / stopped_at in the manifests.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CORE="$(cd "$HERE/../../.." && pwd)"
S="${1:-/Volumes/ExtraDisk/CameraGesturesData/20260927T095311Z_a45264cc}"
RIG="${2:-$CORE/build-rig/tests/replay_rig}"

FILMS=()
for dir in "$S"/examples/phase3/*/; do
    while IFS= read -r f; do FILMS+=("$f"); done < <(ls "$dir"*.json | sort | head -4)
done
echo "films: ${#FILMS[@]}"

MODELS=(--model "$S/models/gesture_model.tflite" --gesture-ids "$S/models/gesture_ids.json"
        --pose-model "$S/models/pose_model.tflite" --pose-manifest "$S/models/pose_manifest.json")

rm -rf "$HERE/corpus_holds.cgsession" "$HERE/corpus_two_hand.cgsession"
"$RIG" "${MODELS[@]}" --holds --session-out "$HERE/corpus_holds.cgsession" "${FILMS[@]}" > /dev/null
"$RIG" "${MODELS[@]}" --holds --pair-tracks --session-out "$HERE/corpus_two_hand.cgsession" "${FILMS[@]}" > /dev/null
du -sh "$HERE"/*.cgsession
