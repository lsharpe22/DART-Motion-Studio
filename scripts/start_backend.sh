#!/usr/bin/env bash
set -euo pipefail
DART_HOME="${DART_HOME:-/content/DART}"
DART_PYTHON="${DART_PYTHON:-/content/dart-env/bin/python}"
# Make the restored Conda environment find its Python standard library.
export PYTHONHOME="${PYTHONHOME:-$(dirname "$(dirname "$DART_PYTHON")")}"
LOG_DIR="$DART_HOME/motion_service/outputs/service_logs"
mkdir -p "$LOG_DIR"
[[ -n "${DART_API_KEY:-}" ]] || { echo 'Set DART_API_KEY privately before launching.' >&2; exit 1; }
[[ -x "$DART_PYTHON" ]] || { echo "Missing Python: $DART_PYTHON" >&2; exit 1; }
export PYTHONPATH="$DART_HOME:$DART_HOME/motion_service/backend${PYTHONPATH:+:$PYTHONPATH}"
export DART_WORKER_URL="http://127.0.0.1:8011"
checkport() { "$DART_PYTHON" -c 'import socket,sys; s=socket.socket(); rc=s.connect_ex(("127.0.0.1",int(sys.argv[1]))); s.close(); sys.exit(0 if rc else 1)' "$1"; }
if ! checkport 8011 || ! checkport 8012; then echo 'Port 8011 or 8012 already occupied. Verify existing processes first.' >&2; exit 1; fi
cd "$DART_HOME"
nohup "$DART_PYTHON" -u "$DART_HOME/motion_service/backend/persistent_worker.py" > "$LOG_DIR/worker.log" 2>&1 < /dev/null &
echo $! > "$LOG_DIR/worker.pid"
echo 'Starting DART worker...'
ready=0
for i in $(seq 1 120); do
 if ! kill -0 "$(cat "$LOG_DIR/worker.pid")" 2>/dev/null; then echo 'Worker exited'; tail -n 35 "$LOG_DIR/worker.log"; exit 1; fi
 if curl --noproxy '*' -fsS --max-time 2 http://127.0.0.1:8011/health > "$LOG_DIR/worker-health.json" 2>/dev/null; then ready=1; break; fi
 sleep 2
done
if [[ "$ready" != 1 ]]; then echo 'Worker not ready; see worker.log'; tail -n 35 "$LOG_DIR/worker.log"; exit 1; fi
cat "$LOG_DIR/worker-health.json"; echo
cd "$DART_HOME/motion_service/backend"
nohup "$DART_PYTHON" -u -m uvicorn api_persistent:app --host 127.0.0.1 --port 8012 > "$LOG_DIR/api.log" 2>&1 < /dev/null &
echo $! > "$LOG_DIR/api.pid"
ready=0
for i in $(seq 1 20); do
 if curl --noproxy '*' -fsS --max-time 2 http://127.0.0.1:8012/health > "$LOG_DIR/api-health.json" 2>/dev/null; then ready=1; break; fi
 sleep 1
done
if [[ "$ready" != 1 ]]; then echo 'API not ready; see api.log'; tail -n 35 "$LOG_DIR/api.log"; exit 1; fi
cat "$LOG_DIR/api-health.json"; echo
printf 'Local API ready: http://127.0.0.1:8012 (worker localhost port 8011)\n'
printf 'Log directory: %s\n' "$LOG_DIR"
