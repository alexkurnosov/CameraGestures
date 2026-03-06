#!/usr/bin/env bash
# test_api.sh — end-to-end smoke tests for the CameraGestures training server.
#
# Usage:
#   ./test_api.sh [BASE_URL] [--verbose]
#
# Examples:
#   ./test_api.sh
#   ./test_api.sh http://192.168.1.42:8000
#   ./test_api.sh http://localhost:8000 --verbose

set -euo pipefail

# ── Arguments ────────────────────────────────────────────────────────────────
BASE_URL="http://localhost:8000"
VERBOSE=false

for arg in "$@"; do
  case "$arg" in
    --verbose) VERBOSE=true ;;
    http*) BASE_URL="$arg" ;;
  esac
done

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# ── State ─────────────────────────────────────────────────────────────────────
PASS=0
FAIL=0
FAILURES=()

# ── Helpers ───────────────────────────────────────────────────────────────────
print_header() {
  echo ""
  echo -e "${CYAN}${BOLD}══════════════════════════════════════════${RESET}"
  echo -e "${CYAN}${BOLD}  $1${RESET}"
  echo -e "${CYAN}${BOLD}══════════════════════════════════════════${RESET}"
}

# run_test NAME EXPECTED_STATUS METHOD URL [BODY]
# Returns the response body in $RESPONSE_BODY
RESPONSE_BODY=""
run_test() {
  local name="$1"
  local expected_status="$2"
  local method="$3"
  local url="$4"
  local body="${5:-}"

  local curl_args=(-s -w "\n%{http_code}" -X "$method")
  if [[ -n "$body" ]]; then
    curl_args+=(-H "Content-Type: application/json" -d "$body")
  fi
  if $VERBOSE; then
    curl_args+=(-v)
  fi
  curl_args+=("$url")

  local raw
  raw=$(curl "${curl_args[@]}" 2>&1)

  local http_code
  http_code=$(echo "$raw" | tail -1)
  # Strip the last line (status code) — portable on both macOS and Linux
  RESPONSE_BODY=$(echo "$raw" | sed '$d')

  if [[ "$http_code" == "$expected_status" ]]; then
    echo -e "  ${GREEN}✓ PASS${RESET}  ${BOLD}$name${RESET} (HTTP $http_code)"
    PASS=$((PASS + 1))
    if $VERBOSE; then
      echo "$RESPONSE_BODY" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE_BODY"
    fi
  else
    echo -e "  ${RED}✗ FAIL${RESET}  ${BOLD}$name${RESET} — expected HTTP $expected_status, got HTTP $http_code"
    FAIL=$((FAIL + 1))
    FAILURES+=("$name (expected $expected_status, got $http_code)")
    echo "$RESPONSE_BODY" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE_BODY"
  fi
}

# ── Shared test data ──────────────────────────────────────────────────────────
make_example() {
  local gesture_id="$1"
  local session_id="${2:-test-session-1}"
  cat <<EOF
{
  "gesture_id": "$gesture_id",
  "session_id": "$session_id",
  "hand_film": {
    "start_time": 0.0,
    "frames": [
      {
        "timestamp": 0.0,
        "left_or_right": "right",
        "landmarks": [
          {"x":0.50,"y":0.50,"z":0.00},{"x":0.50,"y":0.40,"z":0.00},{"x":0.50,"y":0.30,"z":0.00},
          {"x":0.50,"y":0.20,"z":0.00},{"x":0.50,"y":0.10,"z":0.00},{"x":0.40,"y":0.50,"z":0.00},
          {"x":0.30,"y":0.50,"z":0.00},{"x":0.20,"y":0.50,"z":0.00},{"x":0.10,"y":0.50,"z":0.00},
          {"x":0.60,"y":0.50,"z":0.00},{"x":0.70,"y":0.50,"z":0.00},{"x":0.80,"y":0.50,"z":0.00},
          {"x":0.90,"y":0.50,"z":0.00},{"x":0.40,"y":0.60,"z":0.00},{"x":0.30,"y":0.70,"z":0.00},
          {"x":0.20,"y":0.80,"z":0.00},{"x":0.10,"y":0.90,"z":0.00},{"x":0.60,"y":0.60,"z":0.00},
          {"x":0.70,"y":0.70,"z":0.00},{"x":0.80,"y":0.80,"z":0.00},{"x":0.90,"y":0.90,"z":0.00}
        ]
      }
    ]
  }
}
EOF
}

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}CameraGestures API smoke tests${RESET}"
echo -e "Target: ${CYAN}$BASE_URL${RESET}"
echo ""

# ── 1. Meta ───────────────────────────────────────────────────────────────────
print_header "1 · Health & docs"

run_test "GET /health" 200 GET "$BASE_URL/health"
run_test "GET /docs (Swagger)" 200 GET "$BASE_URL/docs"
run_test "GET /redoc" 200 GET "$BASE_URL/redoc"

# ── 2. Examples — empty state ─────────────────────────────────────────────────
print_header "2 · Examples — empty state"

run_test "GET /examples/stats (empty)" 200 GET "$BASE_URL/examples/stats"
run_test "GET /model/status (idle)" 200 GET "$BASE_URL/model/status"
run_test "GET /model/download (404 — no model yet)" 404 GET "$BASE_URL/model/download"
run_test "GET /model/info (404 — no model yet)" 404 GET "$BASE_URL/model/info"

# ── 3. Upload examples ────────────────────────────────────────────────────────
print_header "3 · Upload training examples"

run_test "POST /examples — thumbs_up #1" 201 POST "$BASE_URL/examples" "$(make_example thumbs_up)"
run_test "POST /examples — thumbs_up #2" 201 POST "$BASE_URL/examples" "$(make_example thumbs_up test-session-2)"
run_test "POST /examples — wave #1"      201 POST "$BASE_URL/examples" "$(make_example wave)"
run_test "POST /examples — wave #2"      201 POST "$BASE_URL/examples" "$(make_example wave test-session-2)"

run_test "POST /examples — invalid (empty frames)" 422 POST "$BASE_URL/examples" '{
  "gesture_id": "bad",
  "session_id": "s",
  "hand_film": {"start_time": 0.0, "frames": []}
}'

run_test "GET /examples/stats (after uploads)" 200 GET "$BASE_URL/examples/stats"

# ── 4. Training ───────────────────────────────────────────────────────────────
print_header "4 · Training"

run_test "POST /train — start job" 200 POST "$BASE_URL/train"
run_test "POST /train — already running (or idle)" 200 POST "$BASE_URL/train"

echo ""
echo -e "  ${YELLOW}⏳ Waiting up to 120 s for training to complete...${RESET}"
TRAINING_DONE=false
for i in $(seq 1 24); do
  sleep 5
  STATUS_BODY=$(curl -s "$BASE_URL/model/status")
  STATUS=$(echo "$STATUS_BODY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "")
  echo -e "     ${YELLOW}[$((i*5))s]${RESET} status = $STATUS"
  if [[ "$STATUS" == "ready" || "$STATUS" == "failed" ]]; then
    TRAINING_DONE=true
    break
  fi
done

if $TRAINING_DONE; then
  if [[ "$STATUS" == "ready" ]]; then
    echo -e "  ${GREEN}✓ Training completed successfully${RESET}"
    PASS=$((PASS + 1))
  else
    echo -e "  ${RED}✗ Training failed: $(echo "$STATUS_BODY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error',''))" 2>/dev/null)${RESET}"
    FAIL=$((FAIL + 1))
    FAILURES+=("Training job ended in 'failed' state")
  fi
else
  echo -e "  ${RED}✗ Training did not finish within 120 s${RESET}"
  FAIL=$((FAIL + 1))
  FAILURES+=("Training timed out after 120 s")
fi

# ── 5. Model endpoints ────────────────────────────────────────────────────────
print_header "5 · Model endpoints (post-training)"

run_test "GET /model/status (ready)" 200 GET "$BASE_URL/model/status"
run_test "GET /model/info" 200 GET "$BASE_URL/model/info"
run_test "GET /model/download" 200 GET "$BASE_URL/model/download"

# ── 6. Cleanup — wipe everything ─────────────────────────────────────────────
print_header "6 · Cleanup"

run_test "DELETE /examples?gesture_id=thumbs_up (partial wipe)" 200 DELETE "$BASE_URL/examples?gesture_id=thumbs_up"
run_test "GET /examples/stats (thumbs_up gone)" 200 GET "$BASE_URL/examples/stats"
run_test "DELETE /examples (wipe remaining)" 200 DELETE "$BASE_URL/examples"
run_test "GET /examples/stats (all gone)" 200 GET "$BASE_URL/examples/stats"
run_test "DELETE /model (wipe all models)" 200 DELETE "$BASE_URL/model"
run_test "GET /model/status (idle again)" 200 GET "$BASE_URL/model/status"
run_test "GET /model/download (404 again)" 404 GET "$BASE_URL/model/download"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}${BOLD}══════════════════════════════════════════${RESET}"
echo -e "${BOLD}  Results: ${GREEN}$PASS passed${RESET}  ${RED}$FAIL failed${RESET}"
echo -e "${CYAN}${BOLD}══════════════════════════════════════════${RESET}"

if [[ ${#FAILURES[@]} -gt 0 ]]; then
  echo ""
  echo -e "${RED}Failed tests:${RESET}"
  for f in "${FAILURES[@]}"; do
    echo -e "  ${RED}•${RESET} $f"
  done
  echo ""
  exit 1
fi

echo ""
exit 0
