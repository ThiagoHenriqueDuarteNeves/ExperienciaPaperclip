#!/usr/bin/env bash
# Smoke test for ChromaDB Episodic Memory API
# Prerequisites: docker compose up --build -d (all services healthy)
# Usage: ./smoke_test.sh

set -euo pipefail

API="http://localhost:8001"
PASS=0
FAIL=0

green() { printf '\033[32m%s\033[0m\n' "$1"; }
red()   { printf '\033[31m%s\033[0m\n' "$1"; }

assert() {
    local label="$1" expected="$2" actual="$3"
    if [ "$actual" = "$expected" ]; then
        green "  PASS: $label"
        PASS=$((PASS + 1))
    else
        red "  FAIL: $label (expected '$expected', got '$actual')"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== ChromaDB Episodic Memory API Smoke Test ==="
echo ""

# --- Test 1: Health ---
echo "[1/5] Health endpoint"
HEALTH=$(curl -s "$API/health")
HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' "$API/health")
assert "HTTP 200"       "200"         "$HTTP_CODE"
assert "Status ok"      "ok"          "$(echo "$HEALTH" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("status",""))')"
assert "ChromaDB true"  "True"        "$(echo "$HEALTH" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("chromadb",""))')"

# --- Test 2: Store ---
echo "[2/5] Store endpoint"
STORE=$(curl -s -X POST "$API/store" \
    -H 'Content-Type: application/json' \
    -d '{"user_id":"test-user","conversation_id":"test-conv","content":"Claude Code is an AI-powered CLI tool that helps developers write and refactor code.","metadata":{"topic":"ai-tools"}}')
MEMORY_ID=$(echo "$STORE" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("memory_id",""))')
assert "Returns memory_id" "" ""  # just check it's non-empty
if [ -n "$MEMORY_ID" ]; then
    green "  PASS: memory_id=$MEMORY_ID"
    PASS=$((PASS + 1))
else
    red "  FAIL: no memory_id returned"
    FAIL=$((FAIL + 1))
fi

# --- Test 3: Retrieve (semantic search) ---
echo "[3/5] Retrieve endpoint (semantic search)"
RETRIEVE=$(curl -s -X POST "$API/retrieve" \
    -H 'Content-Type: application/json' \
    -d '{"query":"AI coding assistants","user_id":"test-user","top_k":5}')
COUNT=$(echo "$RETRIEVE" | python3 -c 'import sys,json; print(len(json.load(sys.stdin).get("memories",[])))')
if [ "$COUNT" -gt 0 ]; then
    green "  PASS: Retrieved $COUNT memories"
    PASS=$((PASS + 1))
else
    red "  FAIL: No memories retrieved"
    FAIL=$((FAIL + 1))
fi

# --- Test 4: Retrieve with user filtering ---
echo "[4/5] Retrieve endpoint (user filtering)"
RETRIEVE2=$(curl -s -X POST "$API/retrieve" \
    -H 'Content-Type: application/json' \
    -d '{"query":"AI coding assistants","user_id":"other-user","top_k":5}')
COUNT2=$(echo "$RETRIEVE2" | python3 -c 'import sys,json; print(len(json.load(sys.stdin).get("memories",[])))')
assert "Cross-user isolation" "0" "$COUNT2"

# --- Test 5: Delete ---
echo "[5/5] Delete endpoint"
HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' -X DELETE "$API/memories/$MEMORY_ID")
assert "HTTP 204" "204" "$HTTP_CODE"

# Verify gone
VERIFY=$(curl -s -X POST "$API/retrieve" \
    -H 'Content-Type: application/json' \
    -d '{"query":"AI coding assistants","user_id":"test-user","top_k":5}')
VERIFY_COUNT=$(echo "$VERIFY" | python3 -c 'import sys,json; print(len(json.load(sys.stdin).get("memories",[])))')
assert "Deleted memory gone" "0" "$VERIFY_COUNT"

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
