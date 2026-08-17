#!/usr/bin/env bash
# Copyright (c) 2026 MiLab. All rights reserved.
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$BASE_DIR/runtime/web_backend.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "[INFO] No pid file found"
  exit 0
fi

PID="$(cat "$PID_FILE")"
if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "[INFO] Stopped backend PID=$PID"
else
  echo "[INFO] Process already stopped"
fi
rm -f "$PID_FILE"
