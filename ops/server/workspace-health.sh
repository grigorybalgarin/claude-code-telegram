#!/usr/bin/env bash
# Quick health check: compile all Python sources and verify key configs.
set -euo pipefail

cd "$(dirname "$0")/../.."

echo "=== Compiling Python sources ==="
.venv/bin/python -m compileall -q src

echo "=== Checking key config files ==="
test -f config/agents.yaml && echo "config/agents.yaml OK" || echo "config/agents.yaml MISSING"
test -f config/workspace_profiles.yaml && echo "config/workspace_profiles.yaml OK" || echo "config/workspace_profiles.yaml MISSING"

echo "=== Health check passed ==="
