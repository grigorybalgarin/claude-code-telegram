"""Managed service operations for agentic workspaces."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from .context import ShellActionResult
from .shell_executor import ShellExecutor


@dataclass(frozen=True)
class ServiceFollowUpResult:
    """Structured result of post-action service checks."""

    checks: List[tuple]  # List of (label, ShellActionResult)
    logs_result: Optional[ShellActionResult]
    all_passed: bool


class ServiceController:
    """Execute and inspect managed service lifecycle operations.

    Pure execution layer — no Telegram dependencies. Returns structured
    data that the orchestrator formats for display.
    """

    def __init__(self, shell: ShellExecutor):
        self.shell = shell

    @staticmethod
    def resolve_service(profile: Any, service_key: str) -> Optional[Any]:
        """Resolve a managed service from the current project profile."""
        if not profile:
            return None
        for service in getattr(profile, "services", ()):
            if service.key == service_key:
                return service
        return None

    @staticmethod
    def format_action_label(service: Any, action_key: str) -> str:
        """Build a compact button label for a managed service action."""
        short = service.display_name.strip() or service.key
        if len(short) > 12:
            short = short.split()[0]
        labels = {
            "status": f"📟 {short}",
            "health": f"🩺 {short}",
            "logs": f"📜 {short}",
            "restart": f"🔄 {short}",
            "start": f"▶️ {short}",
            "stop": f"🛑 {short}",
        }
        return labels.get(action_key, f"{short} {action_key}")

    async def run_follow_up_checks(
        self,
        service: Any,
        workspace_root: Path,
        action_key: str,
    ) -> ServiceFollowUpResult:
        """Run post-action checks for managed service lifecycle operations."""
        checks: List[tuple[str, ShellActionResult]] = []
        logs_result: Optional[ShellActionResult] = None
        all_passed = True

        if action_key in {"start", "restart"}:
            await asyncio.sleep(2.0)
            for label, command in (
                ("status", service.command_for("status")),
                ("health", service.command_for("health")),
            ):
                if not command:
                    continue
                result = await self.shell.execute(
                    workspace_root, command, timeout_seconds=45
                )
                checks.append((label, result))
                if not result.success:
                    all_passed = False
            if not all_passed and service.command_for("logs"):
                logs_result = await self.shell.execute(
                    workspace_root, service.command_for("logs"), timeout_seconds=30
                )
        elif action_key == "stop" and service.command_for("status"):
            await asyncio.sleep(1.0)
            status_result = await self.shell.execute(
                workspace_root, service.command_for("status"), timeout_seconds=30
            )
            checks.append(("status", status_result))

        return ServiceFollowUpResult(
            checks=checks, logs_result=logs_result, all_passed=all_passed
        )

    async def list_running_units(self, workspace_root: Path) -> ShellActionResult:
        """Return the currently running systemd services."""
        return await self.shell.execute(
            workspace_root=workspace_root,
            command=(
                "systemctl list-units --type=service --state=running "
                "--no-pager --plain --no-legend"
            ),
            timeout_seconds=30,
        )

    async def list_failed_units(self, workspace_root: Path) -> ShellActionResult:
        """Return currently failed systemd services."""
        return await self.shell.execute(
            workspace_root=workspace_root,
            command=(
                "systemctl list-units --type=service --state=failed "
                "--no-pager --plain --no-legend"
            ),
            timeout_seconds=30,
        )

    @staticmethod
    def parse_systemd_units(result: ShellActionResult, limit: int = 12) -> List[str]:
        """Extract systemd unit names from list-units output."""
        if not result.success or not result.stdout_text:
            return []
        units: List[str] = []
        for line in result.stdout_text.splitlines():
            compact = line.strip()
            if not compact:
                continue
            unit = compact.split()[0]
            if unit.endswith(".service"):
                units.append(unit)
            if len(units) >= limit:
                break
        return units
