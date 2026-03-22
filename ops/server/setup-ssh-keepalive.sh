#!/usr/bin/env bash
# Setup SSH ClientAlive on VPS to detect dead tunnel connections faster.
# Run as root: sudo bash ops/server/setup-ssh-keepalive.sh
set -euo pipefail

SSHD_CONFIG="/etc/ssh/sshd_config"

echo "=== Configuring SSH ClientAlive on VPS ==="

# Backup
cp "$SSHD_CONFIG" "${SSHD_CONFIG}.bak.$(date +%Y%m%d%H%M%S)"

# Uncomment and set ClientAliveInterval
if grep -q "^#ClientAliveInterval" "$SSHD_CONFIG"; then
    sed -i 's/^#ClientAliveInterval.*/ClientAliveInterval 30/' "$SSHD_CONFIG"
    echo "Set ClientAliveInterval 30"
elif ! grep -q "^ClientAliveInterval" "$SSHD_CONFIG"; then
    echo "ClientAliveInterval 30" >> "$SSHD_CONFIG"
    echo "Added ClientAliveInterval 30"
else
    echo "ClientAliveInterval already set: $(grep ^ClientAliveInterval "$SSHD_CONFIG")"
fi

# Uncomment and set ClientAliveCountMax
if grep -q "^#ClientAliveCountMax" "$SSHD_CONFIG"; then
    sed -i 's/^#ClientAliveCountMax.*/ClientAliveCountMax 3/' "$SSHD_CONFIG"
    echo "Set ClientAliveCountMax 3"
elif ! grep -q "^ClientAliveCountMax" "$SSHD_CONFIG"; then
    echo "ClientAliveCountMax 3" >> "$SSHD_CONFIG"
    echo "Added ClientAliveCountMax 3"
else
    echo "ClientAliveCountMax already set: $(grep ^ClientAliveCountMax "$SSHD_CONFIG")"
fi

# Validate config
echo ""
echo "Validating sshd config..."
sshd -t && echo "Config OK" || { echo "ERROR: invalid sshd config!"; exit 1; }

# Reload sshd
echo "Reloading sshd..."
systemctl reload sshd
echo "Done. Dead connections will be detected within ~90 seconds."
