# Systemd Setup

This guide shows how to run the Claude Code Telegram Bot under `systemd`.

## Recommended: Hardened System Service

For a personal production bot, the safer default is:

- run the bot as a dedicated Unix user such as `claude-bot`
- keep the app in `/srv/claude-bot/app`
- set `APPROVED_DIRECTORY=/srv/claude-bot/workspaces`
- expose only selected projects via bind mounts into `/srv/claude-bot/workspaces`

This keeps the bot automated, but narrows the damage radius if it makes a mistake.

### Layout

```text
/home/claude-bot
  .claude
  .ssh
  .config/gh

/srv/claude-bot
  app
  workspaces
    ClaudeBot
    Gr_dev
    FreelanceAggregator
    MacProjects
```

### Service And Workspace Scripts

Reference files live in the repo:

- `ops/systemd/claude-bot.service`
- `ops/server/setup-workspace.sh`

Install them on the server:

```bash
sudo install -D -m 755 ops/server/setup-workspace.sh /usr/local/lib/claude-bot/setup-workspace.sh
sudo install -D -m 644 ops/systemd/claude-bot.service /etc/systemd/system/claude-bot.service
sudo systemctl daemon-reload
sudo systemctl enable --now claude-bot.service
```

### Deploy Updates

Once the hardened runtime is installed, update the bot with:

```bash
sudo bash /srv/claude-bot/app/ops/server/deploy-app.sh
```

The deploy script:

- fetches and fast-forwards the app checkout
- reinstalls the managed `systemd` unit and workspace script
- runs `compileall`
- restarts `claude-bot.service`
- aborts if the app worktree is dirty

Update `.env` in `/srv/claude-bot/app`:

```bash
APPROVED_DIRECTORY=/srv/claude-bot/workspaces
MCP_CONFIG_PATH=/srv/claude-bot/app/config/mcp.json
```

Verify:

```bash
systemctl status claude-bot --no-pager
journalctl -u claude-bot -n 50 --no-pager
```

## Legacy: User Service

This mode runs the bot as a persistent systemd user service.

**⚠️ SECURITY NOTE:** Before setting up the service, ensure your `.env` file has `DEVELOPMENT_MODE=false` and `ENVIRONMENT=production` for secure operation.

## Quick Setup

### 1. Create the service file

```bash
mkdir -p ~/.config/systemd/user
nano ~/.config/systemd/user/claude-telegram-bot.service
```

Add this content:

```ini
[Unit]
Description=Claude Code Telegram Bot
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/ubuntu/Code/oss/claude-code-telegram
ExecStart=/home/ubuntu/.local/bin/poetry run claude-telegram-bot
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Environment
Environment="PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin"

[Install]
WantedBy=default.target
```

**Note:** Update `WorkingDirectory` to your project path.

### 2. Enable and start the service

```bash
# Reload systemd to recognize the new service
systemctl --user daemon-reload

# Enable auto-start on login
systemctl --user enable claude-telegram-bot.service

# Start the service now
systemctl --user start claude-telegram-bot.service
```

### 3. Verify it's running

```bash
systemctl --user status claude-telegram-bot
```

### 4. Verify secure configuration

Check that the service is running in production mode:

```bash
# Check logs for environment mode
journalctl --user -u claude-telegram-bot -n 50 | grep -i "environment\|development"

# Should show:
# "environment": "production"
# "development_mode": false (implied, not shown if false)

# Verify authentication is restricted
journalctl --user -u claude-telegram-bot -n 50 | grep -i "auth"

# Should show:
# "allowed_users": 1 (or more if multiple users configured)
# "allow_all_dev": false
```

If you see `allow_all_dev: true` or `environment: development`, **STOP THE SERVICE** and fix your `.env` file immediately.

## Common Commands

```bash
# Start service
systemctl --user start claude-telegram-bot

# Stop service
systemctl --user stop claude-telegram-bot

# Restart service
systemctl --user restart claude-telegram-bot

# View status
systemctl --user status claude-telegram-bot

# View live logs
journalctl --user -u claude-telegram-bot -f

# View recent logs (last 50 lines)
journalctl --user -u claude-telegram-bot -n 50

# Disable auto-start
systemctl --user disable claude-telegram-bot

# Enable auto-start
systemctl --user enable claude-telegram-bot
```

## Updating the Service

After editing the service file:

```bash
systemctl --user daemon-reload
systemctl --user restart claude-telegram-bot
```

## Troubleshooting

**Service won't start:**
```bash
# Check logs for errors
journalctl --user -u claude-telegram-bot -n 100

# Verify paths in service file are correct
systemctl --user cat claude-telegram-bot

# Check that Poetry is installed
poetry --version

# Test the bot manually first
cd /home/ubuntu/Code/oss/claude-code-telegram
poetry run claude-telegram-bot
```

**Service stops after logout:**

Enable lingering to keep user services running after logout:
```bash
loginctl enable-linger $USER
```

## Files

- Service file: `~/.config/systemd/user/claude-telegram-bot.service`
- Logs: View with `journalctl --user -u claude-telegram-bot`
- Project: `/home/ubuntu/Code/oss/claude-code-telegram`
