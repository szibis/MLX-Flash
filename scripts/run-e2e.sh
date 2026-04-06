#!/usr/bin/env bash
set -euo pipefail

# E2E test: starts Rust proxy, connects to mock Python worker, runs tests.
# Used by: docker compose run e2e

RUST_PORT=${RUST_SERVER_PORT:-8080}
PYTHON_PORT=${PYTHON_WORKER_PORT:-8081}
PYTHON_HOST=${PYTHON_WORKER_HOST:-mock-worker}

echo "=== MLX-Flash E2E Test ==="
echo "  Rust proxy:    :${RUST_PORT}"
echo "  Python worker: ${PYTHON_HOST}:${PYTHON_PORT}"
echo ""

# Wait for Python mock worker to be ready
echo "Waiting for Python worker..."
for i in $(seq 1 30); do
    if python3 -c "
import urllib.request
try:
    urllib.request.urlopen('http://${PYTHON_HOST}:${PYTHON_PORT}/health', timeout=2)
    exit(0)
except:
    exit(1)
" 2>/dev/null; then
        echo "  Python worker ready"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "  ERROR: Python worker not ready after 30s"
        exit 1
    fi
    sleep 1
done

# Start Rust proxy in background
echo "Starting Rust proxy..."
mlx-flash-server \
    --port "${RUST_PORT}" \
    --python-port "${PYTHON_PORT}" \
    --host 0.0.0.0 &
RUST_PID=$!

# Wait for Rust proxy
for i in $(seq 1 10); do
    if python3 -c "
import urllib.request
try:
    urllib.request.urlopen('http://localhost:${RUST_PORT}/health', timeout=2)
    exit(0)
except:
    exit(1)
" 2>/dev/null; then
        echo "  Rust proxy ready"
        break
    fi
    sleep 1
done

echo ""
echo "=== Running E2E tests ==="

PASS=0
FAIL=0

# Test 1: Health endpoint
echo -n "  [1] GET /health ... "
STATUS=$(python3 -c "
import urllib.request, json
r = urllib.request.urlopen('http://localhost:${RUST_PORT}/health')
data = json.loads(r.read())
print('model' in data and 'memory' in data)
" 2>&1)
if [ "$STATUS" = "True" ]; then echo "PASS"; PASS=$((PASS+1)); else echo "FAIL ($STATUS)"; FAIL=$((FAIL+1)); fi

# Test 2: Models endpoint
echo -n "  [2] GET /v1/models ... "
STATUS=$(python3 -c "
import urllib.request, json
r = urllib.request.urlopen('http://localhost:${RUST_PORT}/v1/models')
data = json.loads(r.read())
print(len(data.get('data', [])) > 0)
" 2>&1)
if [ "$STATUS" = "True" ]; then echo "PASS"; PASS=$((PASS+1)); else echo "FAIL ($STATUS)"; FAIL=$((FAIL+1)); fi

# Test 3: Workers endpoint
echo -n "  [3] GET /workers ... "
STATUS=$(python3 -c "
import urllib.request, json
r = urllib.request.urlopen('http://localhost:${RUST_PORT}/workers')
data = json.loads(r.read())
print('workers' in data and 'healthy_count' in data)
" 2>&1)
if [ "$STATUS" = "True" ]; then echo "PASS"; PASS=$((PASS+1)); else echo "FAIL ($STATUS)"; FAIL=$((FAIL+1)); fi

# Test 4: CORS headers
echo -n "  [4] CORS headers present ... "
STATUS=$(python3 -c "
import urllib.request
req = urllib.request.Request('http://localhost:${RUST_PORT}/health')
req.add_header('Origin', 'http://localhost:3000')
r = urllib.request.urlopen(req)
print('access-control-allow-origin' in str(r.headers).lower())
" 2>&1)
if [ "$STATUS" = "True" ]; then echo "PASS"; PASS=$((PASS+1)); else echo "FAIL ($STATUS)"; FAIL=$((FAIL+1)); fi

# Test 5: Chat endpoint validation (missing messages)
echo -n "  [5] POST /v1/chat/completions (validation) ... "
STATUS=$(python3 -c "
import urllib.request, json
req = urllib.request.Request(
    'http://localhost:${RUST_PORT}/v1/chat/completions',
    data=json.dumps({'model': 'test'}).encode(),
    headers={'Content-Type': 'application/json'}
)
try:
    urllib.request.urlopen(req)
    print(False)
except urllib.error.HTTPError as e:
    print(e.code == 400)
" 2>&1)
if [ "$STATUS" = "True" ]; then echo "PASS"; PASS=$((PASS+1)); else echo "FAIL ($STATUS)"; FAIL=$((FAIL+1)); fi

# Test 6: Dashboard HTML
echo -n "  [6] GET /admin (dashboard) ... "
STATUS=$(python3 -c "
import urllib.request
r = urllib.request.urlopen('http://localhost:${RUST_PORT}/admin')
html = r.read().decode()
print('MLX-Flash' in html)
" 2>&1)
if [ "$STATUS" = "True" ]; then echo "PASS"; PASS=$((PASS+1)); else echo "FAIL ($STATUS)"; FAIL=$((FAIL+1)); fi

echo ""
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="

# Cleanup
kill $RUST_PID 2>/dev/null || true

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
echo "All E2E tests passed."
