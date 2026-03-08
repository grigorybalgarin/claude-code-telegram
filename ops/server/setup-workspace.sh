#!/bin/bash
set -euo pipefail

prepare_mountpoint() {
  local target="$1"
  if mountpoint -q "$target"; then
    return 0
  fi
  install -d -o claude-bot -g claude-bot -m 755 "$target"
}

bind_workspace() {
  local source="$1"
  local target="$2"
  local label="$3"
  local required="${4:-0}"

  if mountpoint -q "$target"; then
    return 0
  fi

  if [[ ! -e "$source" ]]; then
    if [[ "$required" == "1" ]]; then
      echo "ERROR: required workspace source for $label not found: $source" >&2
      return 1
    fi

    echo "WARNING: optional workspace source for $label not found, skipping bind mount: $source" >&2
    return 0
  fi

  mount --bind "$source" "$target"
}

prepare_mountpoint /srv/claude-bot/workspaces/ClaudeBot
prepare_mountpoint /srv/claude-bot/workspaces/Gr_dev
prepare_mountpoint /srv/claude-bot/workspaces/FreelanceAggregator
prepare_mountpoint /srv/claude-bot/workspaces/MacProjects

if command -v setfacl >/dev/null 2>&1; then
  setfacl -Rm u:claude-bot:rwx /root/projects/Gr_dev
  find /root/projects/Gr_dev -type d -exec setfacl -m d:u:claude-bot:rwx {} +
else
  chown -R claude-bot:claude-bot /root/projects/Gr_dev
fi

bind_workspace /srv/claude-bot/app /srv/claude-bot/workspaces/ClaudeBot "ClaudeBot" 1
bind_workspace /root/projects/Gr_dev /srv/claude-bot/workspaces/Gr_dev "Gr_dev"
bind_workspace /root/FreelanceAggregator /srv/claude-bot/workspaces/FreelanceAggregator "FreelanceAggregator"
bind_workspace /mnt/mac-files/PycharmProjects /srv/claude-bot/workspaces/MacProjects "MacProjects"
