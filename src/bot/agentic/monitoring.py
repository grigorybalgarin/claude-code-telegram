"""Proactive workspace monitoring with smart notifications.

Periodically runs verify pipelines, detects state transitions,
triggers auto-remediation when safe, and manages incident lifecycle.
Notifies only on state changes (OK→FAIL, FAIL→OK), not repeated failures.
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional
import uuid

import structlog

from .problem_classifier import ProblemDiagnosis, classify_problem
from .server_diagnostics import DiagnosticsCollector
from .shell_executor import ShellExecutor
from .verify_pipeline import VerifyPipeline

logger = structlog.get_logger()


class IncidentState(Enum):
    """Lifecycle state of a monitoring incident."""

    DETECTED = "detected"
    HEALING = "healing"
    HEALED = "healed"
    ESCALATED = "escalated"


@dataclass
class Incident:
    """A single monitoring incident for a workspace."""

    incident_id: str
    workspace_path: str
    state: IncidentState
    diagnosis: Optional[ProblemDiagnosis] = None
    detected_at: float = 0.0
    healed_at: Optional[float] = None
    heal_attempts: int = 0
    last_error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "workspace_path": self.workspace_path,
            "state": self.state.value,
            "problem_type": self.diagnosis.problem_type.value if self.diagnosis else None,
            "short_cause": self.diagnosis.short_cause if self.diagnosis else "",
            "detected_at": self.detected_at,
            "healed_at": self.healed_at,
            "heal_attempts": self.heal_attempts,
            "last_error": self.last_error,
        }


# Callback type for sending notifications
NotifyCallback = Callable[[str], Coroutine[Any, Any, None]]


@dataclass
class WorkspaceHealth:
    """Tracked health state for a single workspace."""

    workspace_path: str
    last_check: float = 0.0
    healthy: bool = True
    consecutive_failures: int = 0
    active_incident: Optional[Incident] = None


class WorkspaceMonitor:
    """Proactive health monitor for workspaces with operations config.

    Runs verify pipelines on a schedule, detects state transitions,
    and orchestrates auto-heal + notification.
    """

    def __init__(
        self,
        shell: ShellExecutor,
        verify: VerifyPipeline,
        diagnostics: DiagnosticsCollector,
        *,
        check_interval_seconds: float = 300.0,
        max_heal_attempts: int = 1,
    ) -> None:
        self.shell = shell
        self.verify = verify
        self.diagnostics = diagnostics
        self.check_interval = check_interval_seconds
        self.max_heal_attempts = max_heal_attempts

        self._health: Dict[str, WorkspaceHealth] = {}
        self._running = False
        self._task: Optional[asyncio.Task[None]] = None

        # Callbacks
        self._on_notify: Optional[NotifyCallback] = None
        self._on_save_operation: Optional[
            Callable[..., Coroutine[Any, Any, None]]
        ] = None

        # Profiles to monitor (set externally)
        self._profiles: List[Any] = []

    def set_profiles(self, profiles: List[Any]) -> None:
        """Set the list of workspace profiles to monitor."""
        self._profiles = profiles

    def set_notify_callback(self, callback: NotifyCallback) -> None:
        self._on_notify = callback

    def set_save_callback(
        self,
        callback: Callable[..., Coroutine[Any, Any, None]],
    ) -> None:
        self._on_save_operation = callback

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Workspace monitor started", profiles=len(self._profiles))

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Workspace monitor stopped")

    @property
    def health_states(self) -> Dict[str, WorkspaceHealth]:
        return dict(self._health)

    def get_active_incidents(self) -> List[Incident]:
        return [
            h.active_incident
            for h in self._health.values()
            if h.active_incident
            and h.active_incident.state
            in {IncidentState.DETECTED, IncidentState.HEALING}
        ]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self._check_all()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Monitor check cycle failed")
            await asyncio.sleep(self.check_interval)

    async def _check_all(self) -> None:
        for profile in self._profiles:
            if not self._running:
                break
            ops = getattr(profile, "operations", None)
            if not ops:
                continue
            await self._check_workspace(profile)

    async def _check_workspace(self, profile: Any) -> None:
        workspace_path = str(getattr(profile, "root_path", ""))
        if not workspace_path:
            return

        steps = self.verify.build_steps(profile)
        if not steps:
            return

        health = self._health.setdefault(
            workspace_path, WorkspaceHealth(workspace_path=workspace_path)
        )

        report = await self.verify.execute(profile)
        now = time.time()
        health.last_check = now
        was_healthy = health.healthy

        if report.success:
            health.healthy = True
            health.consecutive_failures = 0

            # Transition FAIL→OK
            if not was_healthy:
                await self._on_recovery(health, profile)
        else:
            health.healthy = False
            health.consecutive_failures += 1

            # Transition OK→FAIL (first failure)
            if was_healthy:
                await self._on_new_failure(health, profile, report)
            # Already failing — try heal if not exhausted
            elif (
                health.active_incident
                and health.active_incident.state == IncidentState.DETECTED
            ):
                await self._try_auto_heal(health, profile)

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    async def _on_new_failure(
        self, health: WorkspaceHealth, profile: Any, report: Any
    ) -> None:
        """Handle first failure detection for a workspace."""
        ops = getattr(profile, "operations", None)
        server_diag = await self.diagnostics.collect(profile, Path(health.workspace_path))
        diagnosis = classify_problem(
            report,
            operations_config=ops,
            server_context=server_diag.as_prompt_context(),
        )

        incident = Incident(
            incident_id=str(uuid.uuid4())[:8],
            workspace_path=health.workspace_path,
            state=IncidentState.DETECTED,
            diagnosis=diagnosis,
            detected_at=time.time(),
        )
        health.active_incident = incident

        # Persist
        await self._persist_incident(incident, "incident_detected")

        # Notify
        display_name = getattr(profile, "display_name", health.workspace_path)
        lines = [
            f"🔴 <b>{display_name}: сбой обнаружен</b>",
            "",
            f"<b>Тип:</b> {diagnosis.label}",
            f"<b>Шаг:</b> {diagnosis.failed_step_label}",
        ]
        if diagnosis.short_cause:
            lines.append(f"<b>Причина:</b> {diagnosis.short_cause}")

        if self._should_auto_heal(profile, diagnosis):
            lines.append("\nПробую автоматическое восстановление...")
        else:
            lines.append("\nТребуется ручной разбор.")

        await self._notify("\n".join(lines))

        # Try auto-heal immediately if applicable
        if self._should_auto_heal(profile, diagnosis):
            await self._try_auto_heal(health, profile)

    async def _on_recovery(self, health: WorkspaceHealth, profile: Any) -> None:
        """Handle recovery (FAIL→OK transition)."""
        incident = health.active_incident
        display_name = getattr(profile, "display_name", health.workspace_path)

        if incident:
            incident.state = IncidentState.HEALED
            incident.healed_at = time.time()
            await self._persist_incident(incident, "incident_healed")

            duration = int(incident.healed_at - incident.detected_at)
            lines = [
                f"🟢 <b>{display_name}: восстановлен</b>",
                "",
                f"<b>Был сбой:</b> {incident.diagnosis.label if incident.diagnosis else '?'}",
                f"<b>Длительность:</b> {duration}с",
            ]
            if incident.heal_attempts > 0:
                lines.append(f"<b>Авто-починок:</b> {incident.heal_attempts}")
            await self._notify("\n".join(lines))
        else:
            await self._notify(f"🟢 <b>{display_name}: проверки проходят</b>")

        health.active_incident = None

    async def _try_auto_heal(self, health: WorkspaceHealth, profile: Any) -> None:
        """Attempt automatic remediation if policy allows."""
        incident = health.active_incident
        if not incident:
            return

        if incident.heal_attempts >= self.max_heal_attempts:
            if incident.state != IncidentState.ESCALATED:
                incident.state = IncidentState.ESCALATED
                await self._persist_incident(incident, "incident_escalated")
                display_name = getattr(profile, "display_name", health.workspace_path)
                await self._notify(
                    f"⚠️ <b>{display_name}: авто-починка исчерпана</b>\n\n"
                    f"Попыток: {incident.heal_attempts}/{self.max_heal_attempts}\n"
                    "Требуется ручной разбор."
                )
            return

        if not self._should_auto_heal(profile, incident.diagnosis):
            return

        incident.state = IncidentState.HEALING
        incident.heal_attempts += 1

        # Find restart command from services
        restart_cmd = self._get_restart_command(profile)
        if not restart_cmd:
            incident.state = IncidentState.ESCALATED
            incident.last_error = "Нет команды перезапуска"
            await self._persist_incident(incident, "incident_escalated")
            return

        workspace_root = Path(health.workspace_path)
        result = await self.shell.execute(
            workspace_root=workspace_root,
            command=restart_cmd,
            timeout_seconds=30,
        )

        if not result.success:
            incident.last_error = result.stderr_text or result.error or "restart failed"
            incident.state = IncidentState.ESCALATED
            await self._persist_incident(incident, "heal_failed")
            display_name = getattr(profile, "display_name", health.workspace_path)
            await self._notify(
                f"❌ <b>{display_name}: авто-перезапуск не удался</b>\n\n"
                f"Ошибка: {incident.last_error[:120]}"
            )
            return

        # Wait and re-check
        await asyncio.sleep(5)
        health_cmd = self._get_health_command(profile)
        if health_cmd:
            check = await self.shell.execute(
                workspace_root=workspace_root,
                command=health_cmd,
                timeout_seconds=15,
            )
            if check.success:
                # Will be picked up as recovery in next cycle
                incident.state = IncidentState.DETECTED  # stay detected, next verify will confirm
                await self._persist_incident(incident, "heal_attempted")
            else:
                incident.state = IncidentState.ESCALATED
                incident.last_error = "Перезапуск выполнен, но проверка не прошла"
                await self._persist_incident(incident, "heal_failed")

    # ------------------------------------------------------------------
    # Policy helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _should_auto_heal(profile: Any, diagnosis: Optional[ProblemDiagnosis]) -> bool:
        """Check if auto-heal is allowed by profile policy."""
        ops = getattr(profile, "operations", None)
        if not ops or not getattr(ops, "self_heal_restart", False):
            return False
        # Only auto-heal service problems (not code/config/deploy issues)
        if diagnosis and diagnosis.problem_type.value in ("code", "config", "deploy"):
            return False
        return True

    @staticmethod
    def _get_restart_command(profile: Any) -> Optional[str]:
        """Extract restart command from profile services."""
        for svc in getattr(profile, "services", ()):
            cmd = getattr(svc, "restart_command", None)
            if cmd:
                return cmd
        return None

    @staticmethod
    def _get_health_command(profile: Any) -> Optional[str]:
        """Extract health check command from profile services."""
        for svc in getattr(profile, "services", ()):
            cmd = getattr(svc, "health_command", None) or getattr(
                svc, "status_command", None
            )
            if cmd:
                return cmd
        return None

    # ------------------------------------------------------------------
    # Persistence & notification
    # ------------------------------------------------------------------

    async def _persist_incident(self, incident: Incident, event_type: str) -> None:
        if self._on_save_operation:
            try:
                await self._on_save_operation(
                    workspace_path=incident.workspace_path,
                    operation_type=event_type,
                    success=incident.state == IncidentState.HEALED,
                    details=incident.to_dict(),
                    correlation_id=incident.incident_id,
                )
            except Exception:
                logger.warning("Failed to persist incident", incident_id=incident.incident_id)

    async def _notify(self, text: str) -> None:
        if self._on_notify:
            try:
                await self._on_notify(text)
            except Exception:
                logger.warning("Failed to send monitoring notification")
