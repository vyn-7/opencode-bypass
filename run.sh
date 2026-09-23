#!/usr/bin/env bash
# Helper to run the opencode free-tier proxy (venv-aware).
# Usage: ./run.sh [--port 18788] [-- <opencode_proxy.py args...>]
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Creating venv (.venv)..."
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi

PIDFILE="${XDG_RUNTIME_DIR:-/tmp}/opencode-proxy.pid"
case "${1:-run}" in
  run) exec .venv/bin/python opencode_proxy.py "${@:2}" ;;
  start)
    shift || true
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "Already running (pid $(cat "$PIDFILE"))"; exit 0
    fi
    nohup .venv/bin/python opencode_proxy.py "$@" >opencode-proxy.log 2>&1 &
    echo $! >"$PIDFILE"
    echo "Started (pid $!, log $PWD/opencode-proxy.log)"
    ;;
  stop)
    if [ -f "$PIDFILE" ]; then kill "$(cat "$PIDFILE")" && rm -f "$PIDFILE" && echo stopped; fi
    ;;
  restart)
    "$0" stop || true
    sleep 1
    "$0" start "${@:2}"
    ;;
  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "running (pid $(cat "$PIDFILE"))"
    else
      echo "not running"
    fi
    ;;
  logs) tail -f opencode-proxy.log ;;
esac
