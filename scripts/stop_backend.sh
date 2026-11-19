#!/usr/bin/env bash
set -euo pipefail
DART_HOME="${DART_HOME:-/content/DART}"
LOG_DIR="$DART_HOME/motion_service/outputs/service_logs"
for name in api worker; do
 pidfile="$LOG_DIR/$name.pid"
 if [[ -f "$pidfile" ]]; then
  pid="$(cat "$pidfile")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
   echo "Stopping $name process $pid"; kill "$pid" || true
  fi
  rm -f "$pidfile"
 fi
done
# Only stops recorded processes; does not use system-wide pkill.
