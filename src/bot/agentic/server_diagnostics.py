"""Server-aware diagnostics collector.

Gathers server-level context (systemd state, recent errors, disk usage)
for smarter diagnosis of production problems. Uses profile's operations
config for project-specific diagnostics commands.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .shell_executor import ShellExecutor


@dataclass
class ServerDiagnostics:
    """Collected server-level diagnostic data."""

    service_active: Optional[bool] = None
    service_state: str = ""
    recent_errors: str = ""
    restart_count: Optional[int] = None
    disk_usage: str = ""
    git_state: str = ""
    extra: Dict[str, str] = field(default_factory=dict)

    @property
    def has_service_problem(self) -> bool:
        return self.service_active is False

    @property
    def has_disk_problem(self) -> bool:
        if not self.disk_usage:
            return False
        for line in self.disk_usage.splitlines():
            parts = line.split()
            for part in parts:
                if part.endswith("%"):
                    try:
                        usage = int(part.rstrip("%"))
                        if usage >= 95:
                            return True
                    except ValueError:
                        pass
        return False

    @property
    def is_flapping(self) -> bool:
        return self.restart_count is not None and self.restart_count >= 3

    def summary_lines(self) -> List[str]:
        """Return human-readable diagnostic lines for display."""
        lines = []
        if self.service_active is not None:
            state = "активен" if self.service_active else "не активен"
            lines.append(f"Сервис: {state}")
            if self.service_state:
                lines.append(f"Состояние: {self.service_state}")
        if self.is_flapping:
            lines.append(f"Перезапуски: {self.restart_count} (флаппинг)")
        if self.has_disk_problem:
            lines.append("Диск: мало места")
        if self.recent_errors:
            error_lines = self.recent_errors.strip().splitlines()
            if error_lines:
                lines.append(f"Последняя ошибка: {error_lines[-1][:120]}")
        return lines

    def as_prompt_context(self) -> str:
        """Format diagnostics as context for Claude prompt."""
        parts = []
        if self.service_state:
            parts.append(f"Состояние сервиса:\n{self.service_state}")
        if self.recent_errors:
            parts.append(f"Недавние ошибки:\n{self.recent_errors}")
        if self.disk_usage:
            parts.append(f"Диск:\n{self.disk_usage}")
        if self.git_state:
            parts.append(f"Git:\n{self.git_state}")
        for key, value in self.extra.items():
            parts.append(f"{key}:\n{value}")
        if not parts:
            return ""
        return "\nДиагностика сервера:\n" + "\n\n".join(parts) + "\n"


class DiagnosticsCollector:
    """Collect server diagnostics using profile's operations config."""

    def __init__(self, shell: ShellExecutor):
        self.shell = shell

    async def collect(
        self,
        profile: Any,
        workspace_root: Path,
    ) -> ServerDiagnostics:
        """Gather diagnostics from profile's operations.diagnose_commands."""
        diag = ServerDiagnostics()

        ops = getattr(profile, "operations", None)
        if not ops:
            # Fall back to service-based diagnostics
            return await self._collect_from_services(profile, workspace_root, diag)

        commands = getattr(ops, "diagnose_commands", {}) or {}

        for key, command in commands.items():
            result = await self.shell.execute(
                workspace_root=workspace_root,
                command=command,
                timeout_seconds=10,
            )
            output = result.stdout_text.strip() or result.stderr_text.strip()

            if key == "service_state":
                diag.service_state = output
                diag.service_active = result.success
            elif key == "recent_errors":
                diag.recent_errors = output
            elif key == "disk_usage":
                diag.disk_usage = output
            elif key == "git_state":
                diag.git_state = output
            else:
                if output:
                    diag.extra[key] = output

        # Detect restart count from systemd if service configured
        services = list(getattr(profile, "services", ()))
        for svc in services:
            if getattr(svc, "service_type", "") == "systemd":
                unit = None
                status_cmd = getattr(svc, "status_command", "") or ""
                if "systemctl status" in status_cmd:
                    unit = status_cmd.replace("systemctl status", "").replace("--no-pager", "").strip()
                if unit:
                    diag.restart_count = await self._get_restart_count(
                        workspace_root, unit
                    )
                break

        return diag

    async def _collect_from_services(
        self,
        profile: Any,
        workspace_root: Path,
        diag: ServerDiagnostics,
    ) -> ServerDiagnostics:
        """Basic diagnostics from service definitions (no operations config)."""
        services = list(getattr(profile, "services", ()))
        if not services:
            return diag

        svc = services[0]
        health_cmd = getattr(svc, "health_command", None)
        if health_cmd:
            result = await self.shell.execute(
                workspace_root=workspace_root,
                command=health_cmd,
                timeout_seconds=10,
            )
            diag.service_active = result.success
            diag.service_state = result.stdout_text.strip()

        logs_cmd = getattr(svc, "logs_command", None)
        if logs_cmd and not diag.service_active:
            result = await self.shell.execute(
                workspace_root=workspace_root,
                command=logs_cmd,
                timeout_seconds=10,
            )
            diag.recent_errors = result.stdout_text.strip() or result.stderr_text.strip()

        return diag

    async def _get_restart_count(
        self, workspace_root: Path, unit: str
    ) -> Optional[int]:
        """Get NRestarts from systemd for flapping detection."""
        result = await self.shell.execute(
            workspace_root=workspace_root,
            command=f"systemctl show {unit} -p NRestarts --value",
            timeout_seconds=5,
        )
        if result.success and result.stdout_text.strip().isdigit():
            return int(result.stdout_text.strip())
        return None
