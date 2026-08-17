#!/usr/bin/env bash
# Copyright (c) 2026 MiLab. All rights reserved.
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$BASE_DIR/runtime"
PID_FILE="$RUNTIME_DIR/web_backend.pid"
LOG_FILE="$RUNTIME_DIR/web_backend.log"
PORT="${XILILAKE_WEB_PORT:-${CDH_WEB_PORT:-5001}}"
PYTHON_BIN="${XILILAKE_PYTHON:-${PYTHON_BIN:-$BASE_DIR/.venv/bin/python}}"

mkdir -p "$RUNTIME_DIR"

if [[ ! -x "$PYTHON_BIN" ]] && [[ -x "$BASE_DIR/cdh-bench-env/bin/python" ]]; then
  PYTHON_BIN="$BASE_DIR/cdh-bench-env/bin/python"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python not found"
  exit 1
fi

echo "[INFO] Using python: $PYTHON_BIN"
"$PYTHON_BIN" -m pip install -q -r "$BASE_DIR/requirements-web.txt"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[INFO] Stopping existing backend (PID=$OLD_PID)"
    kill "$OLD_PID" || true
    sleep 2
  fi
fi

export XILILAKE_WEB_PORT="$PORT"
if command -v setsid >/dev/null 2>&1; then
  setsid "$PYTHON_BIN" "$BASE_DIR/web_backend.py" > "$LOG_FILE" 2>&1 < /dev/null &
else
  nohup "$PYTHON_BIN" "$BASE_DIR/web_backend.py" > "$LOG_FILE" 2>&1 < /dev/null &
fi
echo $! > "$PID_FILE"

sleep 2

NEW_PID="$(cat "$PID_FILE")"
if ! kill -0 "$NEW_PID" 2>/dev/null; then
  echo "[ERROR] Backend failed to stay running. Log:"
  tail -n 80 "$LOG_FILE" || true
  exit 1
fi

echo "[INFO] Backend started"
echo "[INFO] URL: http://0.0.0.0:$PORT"
echo "[INFO] PID: $NEW_PID"
echo "[INFO] Log: $LOG_FILE"
