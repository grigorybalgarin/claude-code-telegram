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

check_untracked_garbage() {
  # Detect common garbage from manual cp -r deployments
  local garbage_dirs=("$APP_DIR/src/src" "$APP_DIR/src/bot/bot" "$APP_DIR/config/config")
  for dir in "${garbage_dirs[@]}"; do
    if [[ -d "$dir" ]]; then
      echo "WARNING: removing garbage directory $dir (likely from manual cp -r)" >&2
      rm -rf "$dir"
    fi
  done

  # Check for any .pyc outside __pycache__ (sign of broken copy)
  local stray_pyc
  stray_pyc="$(find "$APP_DIR/src" -name '*.pyc' -not -path '*/__pycache__/*' 2>/dev/null | head -5)"
  if [[ -n "$stray_pyc" ]]; then
    echo "WARNING: stray .pyc files found outside __pycache__:" >&2
    echo "$stray_pyc" >&2
  fi
}

if [[ $EUID -ne 0 ]]; then
  echo "Run this script as root." >&2
  exit 1
fi

if [[ ! -d "$APP_DIR/.git" ]]; then
  echo "App directory is not a git checkout: $APP_DIR" >&2
  echo "Do NOT deploy with 'cp -r'. Use this script or 'git pull'." >&2
  exit 1
fi

require_clean_worktree
check_untracked_garbage

echo "Pulling $BRANCH..."
run_as_app git -C "$APP_DIR" fetch origin "$BRANCH"
run_as_app git -C "$APP_DIR" checkout "$BRANCH"
run_as_app git -C "$APP_DIR" pull --ff-only origin "$BRANCH"

install -D -m 755 "$APP_DIR/ops/server/setup-workspace.sh" /usr/local/lib/claude-bot/setup-workspace.sh
install -D -m 644 "$APP_DIR/ops/systemd/claude-bot.service" /etc/systemd/system/claude-bot.service
install -D -m 644 "$APP_DIR/ops/server/50-claude-bot-systemd.rules" /etc/polkit-1/rules.d/50-claude-bot-systemd.rules

echo "Compiling Python bytecode..."
if ! run_as_app "$APP_DIR/.venv/bin/python" -m compileall -q "$APP_DIR/src" 2>&1; then
  echo "FATAL: Python compilation failed. There are syntax errors in the code." >&2
  echo "The service was NOT restarted. Fix the errors and re-deploy." >&2
  exit 1
fi

echo "Restarting service..."
systemctl daemon-reload
systemctl restart "$SERVICE_NAME"
sleep 5

if ! systemctl is-active --quiet "$SERVICE_NAME"; then
  echo "FATAL: Service failed to start after deploy." >&2
  journalctl -u "$SERVICE_NAME" -n 80 --no-pager >&2 || true
  exit 1
fi

DEPLOYED_COMMIT="$(run_as_app git -C "$APP_DIR" rev-parse --short HEAD)"
echo ""
echo "=== Deploy successful ==="
echo "Commit: $DEPLOYED_COMMIT"
echo "Service: $(systemctl is-active "$SERVICE_NAME")"
echo "PID: $(systemctl show -p MainPID --value "$SERVICE_NAME")"
systemctl status "$SERVICE_NAME" --no-pager -n 10
