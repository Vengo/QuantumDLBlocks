#!/usr/bin/env bash
# Idempotent Zero-Downtime Helper for the QuantumDLBlocks Benchmarking Server (Plan 017)
# - If the zero-downtime server is already healthy on port 8765 (responds to POST /api/reload),
#   triggers an immediate hot reload (< 100 ms) without restarting or dropping connections.
# - If the server is not running (or is a pre-Plan-017 legacy process without /api/reload),
#   starts the permanent zero-downtime server in the background on 0.0.0.0:8765.

set -euo pipefail

REPO_ROOT="/home/festivity/agy/QuantumDLBlocks"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
SERVER_SCRIPT="${REPO_ROOT}/scripts/monitor_colab_runs.py"
LOG_FILE="${REPO_ROOT}/.cache/monitor_server.log"
HOST="${QDL_BENCHMARK_HOST:-0.0.0.0}"
PORT="${QDL_BENCHMARK_PORT:-8765}"
INTERVAL="${QDL_BENCHMARK_INTERVAL:-20}"

mkdir -p "${REPO_ROOT}/.cache"

if [[ "${1:-}" == "--install-systemd" ]]; then
    mkdir -p "${HOME}/.config/systemd/user"
    cp "${REPO_ROOT}/scripts/qdl-benchmark-server.service" "${HOME}/.config/systemd/user/qdl-benchmark-server.service"
    systemctl --user daemon-reload || true
    systemctl --user enable --now qdl-benchmark-server.service || true
    echo "[ensure_benchmark_server] Installed and enabled systemd user unit qdl-benchmark-server.service"
    exit 0
fi

# 1. Check if the zero-downtime server is already running and supports POST /api/reload
if RELOAD_JSON="$(curl -fsS --max-time 2 -X POST "http://127.0.0.1:${PORT}/api/reload" 2>/dev/null)"; then
    echo "[ensure_benchmark_server] Zero-downtime server healthy on port ${PORT}; hot-reloaded registry:"
    echo "  ${RELOAD_JSON}"
    exit 0
fi

# 2. If a legacy process is holding the port without /api/reload, replace it once so zero-downtime is active
if curl -fsS --max-time 1 "http://127.0.0.1:${PORT}/api/status" >/dev/null 2>&1; then
    echo "[ensure_benchmark_server] Upgrading legacy server on port ${PORT} to Plan 017 Zero-Downtime Server..."
    pkill -f "monitor_colab_runs.py --serve" || true
    sleep 1
fi

# 3. Start the permanent background server
echo "[ensure_benchmark_server] Starting permanent benchmark server on ${HOST}:${PORT}..."
nohup "${PYTHON_BIN}" "${SERVER_SCRIPT}" --serve --host "${HOST}" --port "${PORT}" --interval "${INTERVAL}" \
    >> "${LOG_FILE}" 2>&1 &
SERVER_PID=$!

# 4. Wait briefly for readiness
for _ in $(seq 1 30); do
    if STATUS_JSON="$(curl -fsS --max-time 1 -X POST "http://127.0.0.1:${PORT}/api/reload" 2>/dev/null)"; then
        echo "[ensure_benchmark_server] Server ready (PID ${SERVER_PID}): ${STATUS_JSON}"
        exit 0
    fi
    sleep 0.2
done

echo "[ensure_benchmark_server] WARNING: Server process started (PID ${SERVER_PID}) but /api/reload did not respond within 6s." >&2
exit 1
