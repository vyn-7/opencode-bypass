#!/usr/bin/env bash
# One-command install + autostart for the opencode free-tier proxy (Linux).
#
#   ./install.sh                 install, enable autostart, start now
#   ./install.sh --port 18789    same on a custom port
#   ./install.sh --no-autostart  install only, do not enable autostart
#   ./install.sh --uninstall     remove autostart (code + venv are kept)
#
# Autostart method (auto-detected):
#   1. systemd user service (preferred): survives reboots, restarts on failure,
#      logs to the journal (journalctl --user -u opencode-proxy.service).
#   2. cron @reboot fallback: for hosts without a user systemd session
#      (minimal servers, some WSL setups).
set -euo pipefail
cd "$(dirname "$0")"

PORT=18788
AUTOSTART=1
UNINSTALL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --no-autostart) AUTOSTART=0; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1 (try --help)"; exit 1 ;;
  esac
done

SERVICE=opencode-proxy.service
UNIT_DIR="$HOME/.config/systemd/user"

have_user_systemd() {
  command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1
}

cron_uninstall() {
  if command -v crontab >/dev/null 2>&1; then
    crontab -l 2>/dev/null | grep -v "opencode-bypass/run.sh start" | crontab - || true
  fi
}

systemd_uninstall() {
  if have_user_systemd; then
    systemctl --user disable --now "$SERVICE" >/dev/null 2>&1 || true
  fi
  rm -f "$UNIT_DIR/$SERVICE"
  if have_user_systemd; then
    systemctl --user daemon-reload >/dev/null 2>&1 || true
  fi
}

if [ "$UNINSTALL" -eq 1 ]; then
  echo "Removing autostart..."
  systemd_uninstall
  cron_uninstall
  ./run.sh stop || true
  echo "Autostart removed. Code and .venv kept (delete the directory to remove fully)."
  exit 0
fi

echo "== 1/4 prerequisites =="
command -v python3 >/dev/null || { echo "ERROR: python3 not found"; exit 1; }
if command -v opencode >/dev/null; then
  echo "opencode CLI: $(command -v opencode) ($(opencode --version 2>/dev/null || echo version unknown))"
else
  echo "WARNING: 'opencode' not on PATH — install it from https://opencode.ai"
  echo "and make sure it is authenticated, then re-run ./install.sh."
fi

echo "== 2/4 python environment =="
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -m py_compile opencode_proxy.py
echo "venv ready."

echo "== 3/4 autostart =="
if [ "$AUTOSTART" -eq 0 ]; then
  echo "Skipped (--no-autostart). Start manually with: ./run.sh start --port $PORT"
else
  if have_user_systemd; then
    mkdir -p "$UNIT_DIR"
    # Render the unit with this project's real paths (no %h assumptions).
    sed -e "s|@PROJECT_DIR@|$PWD|g" -e "s|@PORT@|$PORT|g" \
      systemd/opencode-proxy.service.template >"$UNIT_DIR/$SERVICE"
    systemctl --user daemon-reload
    systemctl --user enable "$SERVICE"
    systemctl --user restart "$SERVICE"  # restart (not just start) so a re-install picks up flag changes
    echo "systemd user service enabled + started: $SERVICE"
    echo "logs: journalctl --user -u $SERVICE -f"
  elif command -v crontab >/dev/null 2>&1; then
    cron_uninstall
    (crontab -l 2>/dev/null; echo "@reboot $PWD/run.sh start --port $PORT >>$PWD/opencode-proxy.log 2>&1") | crontab -
    echo "No user systemd session — installed cron @reboot fallback instead."
  else
    echo "WARNING: neither user systemd nor crontab available; skipping autostart."
    echo "Start manually with: ./run.sh start --port $PORT"
  fi
fi

echo "== 4/4 verify =="
if [ "$AUTOSTART" -eq 1 ] && have_user_systemd \
  && systemctl --user is-active -q "$SERVICE"; then
  echo "service is active."
elif ./run.sh status 2>/dev/null | grep -q "^running"; then
  echo "background copy already running."
else
  # --no-autostart, or the cron fallback (which only fires on reboot):
  # boot a copy now so the proxy is usable immediately.
  ./run.sh start --port "$PORT"
fi
for _ in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! curl -sf "http://127.0.0.1:$PORT/health"; then
  echo "ERROR: proxy is not responding on :$PORT. Check logs:"
  echo "  systemd: journalctl --user -u $SERVICE -n 30 --no-pager"
  echo "  run.sh:  tail -30 opencode-proxy.log"
  exit 1
fi
echo
echo "Done. Proxy is live at http://127.0.0.1:$PORT"
echo "Smoke test: ./test_proxy.sh http://127.0.0.1:$PORT"
