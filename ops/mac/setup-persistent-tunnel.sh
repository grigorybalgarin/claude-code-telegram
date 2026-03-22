#!/usr/bin/env bash
# Setup persistent reverse SSH tunnel from Mac to VPS.
# Run on Mac: bash setup-persistent-tunnel.sh
#
# What this does:
# 1. Configures pmset to prevent sleep (display still turns off)
# 2. Installs autossh via Homebrew
# 3. Creates a launchd agent for auto-reconnecting tunnel
# 4. Adds a scheduled wake as safety net
set -euo pipefail

VPS_USER="claude-bot"
VPS_HOST="194.87.44.239"
VPS_PORT="22"
TUNNEL_PORT="2222"
SSH_KEY="$HOME/.ssh/id_ed25519"  # adjust if different
LABEL="com.claudebot.autossh-tunnel"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"

echo "=== Mac Persistent Tunnel Setup ==="
echo ""

# --- Step 1: pmset ---
echo "--- Step 1: Power management ---"
echo "Setting: no system sleep, display off after 5 min, no standby/hibernate"
sudo pmset -a sleep 0
sudo pmset -a displaysleep 5
sudo pmset -a standby 0
sudo pmset -a autopoweroff 0
sudo pmset -a tcpkeepalive 1
sudo pmset -a powernap 1
sudo pmset -a hibernatemode 0
echo "pmset configured. Current settings:"
pmset -g | grep -E "sleep|standby|autopoweroff|tcpkeepalive|powernap|hibernatemode"
echo ""

# --- Step 2: autossh ---
echo "--- Step 2: Installing autossh ---"
if command -v autossh &>/dev/null; then
    echo "autossh already installed: $(which autossh)"
else
    if command -v brew &>/dev/null; then
        brew install autossh
        echo "autossh installed"
    else
        echo "ERROR: Homebrew not found. Install it first: https://brew.sh"
        exit 1
    fi
fi

AUTOSSH_PATH="$(which autossh)"
echo "autossh path: $AUTOSSH_PATH"
echo ""

# --- Step 3: Stop existing tunnel if running ---
echo "--- Step 3: Stopping old tunnel ---"
if launchctl list "$LABEL" &>/dev/null 2>&1; then
    launchctl unload "$PLIST" 2>/dev/null || true
    echo "Old launchd agent unloaded"
else
    echo "No existing agent found"
fi

# Kill any existing manual tunnel on port 2222
pkill -f "ssh.*-R.*${TUNNEL_PORT}:" 2>/dev/null && echo "Killed old SSH tunnel" || echo "No old tunnel running"
echo ""

# --- Step 4: Create launchd plist ---
echo "--- Step 4: Creating launchd agent ---"
mkdir -p "$HOME/Library/LaunchAgents"

cat > "$PLIST" << PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${AUTOSSH_PATH}</string>
        <string>-M</string>
        <string>0</string>
        <string>-N</string>
        <string>-R</string>
        <string>${TUNNEL_PORT}:localhost:22</string>
        <string>-p</string>
        <string>${VPS_PORT}</string>
        <string>-i</string>
        <string>${SSH_KEY}</string>
        <string>-o</string>
        <string>ServerAliveInterval=30</string>
        <string>-o</string>
        <string>ServerAliveCountMax=3</string>
        <string>-o</string>
        <string>ExitOnForwardFailure=yes</string>
        <string>-o</string>
        <string>StrictHostKeyChecking=no</string>
        <string>${VPS_USER}@${VPS_HOST}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>10</integer>
    <key>StandardOutPath</key>
    <string>/tmp/autossh-tunnel.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/autossh-tunnel.err</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>AUTOSSH_GATETIME</key>
        <string>0</string>
    </dict>
</dict>
</plist>
PLISTEOF

echo "Created: $PLIST"
echo ""

# --- Step 5: Load and start ---
echo "--- Step 5: Starting tunnel ---"
launchctl load "$PLIST"
sleep 3

if launchctl list "$LABEL" &>/dev/null 2>&1; then
    echo "Tunnel agent loaded and running"
else
    echo "WARNING: Agent may not have started. Check: launchctl list | grep claude"
fi
echo ""

# --- Step 6: Scheduled wake (safety net) ---
echo "--- Step 6: Scheduled wake (safety net) ---"
# Wake daily at 8:00 and 20:00 as safety net
sudo pmset repeat wake MTWRFSU 08:00:00
echo "Scheduled daily wake at 08:00"
echo ""

# --- Step 7: Verify ---
echo "=== Verification ==="
echo "Tunnel log: /tmp/autossh-tunnel.log"
echo "Tunnel errors: /tmp/autossh-tunnel.err"
echo ""
echo "Testing tunnel in 5 seconds..."
sleep 5

# Check if autossh process is running
if pgrep -f "autossh.*${TUNNEL_PORT}" >/dev/null; then
    echo "autossh process: RUNNING"
else
    echo "autossh process: NOT FOUND - check /tmp/autossh-tunnel.err"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "To verify from server:  ssh mac 'echo OK'"
echo "To check tunnel status: launchctl list $LABEL"
echo "To view logs:           tail -f /tmp/autossh-tunnel.log"
echo "To stop tunnel:         launchctl unload $PLIST"
echo ""
echo "IMPORTANT: For clamshell mode (lid closed), connect:"
echo "  1. Power adapter (required)"
echo "  2. HDMI dummy plug or external monitor (required)"
echo "Without these, closing the lid will still sleep the Mac."
