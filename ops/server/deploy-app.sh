#!/bin/bash
set -euo pipefail

APP_USER="${APP_USER:-claude-bot}"
APP_HOME="${APP_HOME:-/home/claude-bot}"
APP_DIR="${APP_DIR:-/srv/claude-bot/app}"
SERVICE_NAME="${SERVICE_NAME:-claude-bot.service}"
BRANCH="${BRANCH:-main}"

run_as_app() {
  sudo -u "$APP_USER" env HOME="$APP_HOME" "$@"
}

require_clean_worktree() {
  local status
  status="$(run_as_app git -C "$APP_DIR" status --porcelain --untracked-files=no)"
  if [[ -n "$status" ]]; then
    echo "Refusing deploy: worktree is dirty in $APP_DIR" >&2
    echo "$status" >&2
    exit 1
  fi
}

if [[ $EUID -ne 0 ]]; then
  echo "Run this script as root." >&2
  exit 1
fi

if [[ ! -d "$APP_DIR/.git" ]]; then
  echo "App directory is not a git checkout: $APP_DIR" >&2
  exit 1
fi

require_clean_worktree

run_as_app git -C "$APP_DIR" fetch origin "$BRANCH"
run_as_app git -C "$APP_DIR" checkout "$BRANCH"
run_as_app git -C "$APP_DIR" pull --ff-only origin "$BRANCH"

install -D -m 755 "$APP_DIR/ops/server/setup-workspace.sh" /usr/local/lib/claude-bot/setup-workspace.sh
install -D -m 644 "$APP_DIR/ops/systemd/claude-bot.service" /etc/systemd/system/claude-bot.service

run_as_app "$APP_DIR/.venv/bin/python" -m compileall -q "$APP_DIR/src"

systemctl daemon-reload
systemctl restart "$SERVICE_NAME"
sleep 5

if ! systemctl is-active --quiet "$SERVICE_NAME"; then
  echo "Service failed to start: $SERVICE_NAME" >&2
  journalctl -u "$SERVICE_NAME" -n 80 --no-pager >&2 || true
  exit 1
fi

echo "Deployed commit $(run_as_app git -C "$APP_DIR" rev-parse --short HEAD)"
systemctl status "$SERVICE_NAME" --no-pager -n 20
