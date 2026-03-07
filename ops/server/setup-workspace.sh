#!/bin/bash
set -euo pipefail

prepare_mountpoint() {
  local target="$1"
  if mountpoint -q "$target"; then
    return 0
  fi
  install -d -o claude-bot -g claude-bot -m 755 "$target"
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

mountpoint -q /srv/claude-bot/workspaces/ClaudeBot || mount --bind /srv/claude-bot/app /srv/claude-bot/workspaces/ClaudeBot
mountpoint -q /srv/claude-bot/workspaces/Gr_dev || mount --bind /root/projects/Gr_dev /srv/claude-bot/workspaces/Gr_dev
mountpoint -q /srv/claude-bot/workspaces/FreelanceAggregator || mount --bind /root/FreelanceAggregator /srv/claude-bot/workspaces/FreelanceAggregator
mountpoint -q /srv/claude-bot/workspaces/MacProjects || mount --bind /mnt/mac-files/PycharmProjects /srv/claude-bot/workspaces/MacProjects
