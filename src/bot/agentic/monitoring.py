"""Proactive workspace monitoring with smart notifications.

Periodically runs verify pipelines, detects state transitions,
triggers auto-remediation when safe, and manages incident lifecycle.
Uses severity-aware notifications and incident deduplication.
"""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional
import uuid

import structlog

from .ops_model import (
    AutonomyGuardrails,
    Incident,
    IncidentState,
    Severity,
)
from .problem_classifier import ProblemDiagnosis, ProblemType, classify_problem
from .server_diagnostics import DiagnosticsCollector
from .shell_executor import ShellExecutor
from .verify_pipeline import VerifyPipeline

logger = structlog.get_logger()

# Re-export for backward compat with tests
__all__ = [
    "Incident",
    "IncidentState",
    "WorkspaceHealth",
    "WorkspaceMonitor",
]

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
    last_healthy_at: Optional[float] = None
    notification_cooldown_until: float = 0.0


def _classify_severity(
    diagnosis: Optional[ProblemDiagnosis],
    consecutive_failures: int,
    service_down: bool,
) -> Severity:
    """Determine incident severity from diagnosis and context."""
    if service_down:
        return Severity.CRITICAL
    if diagnosis and diagnosis.is_critical_step:
        return Severity.CRITICAL
    if diagnosis and diagnosis.problem_type in (ProblemType.SERVICE, ProblemType.ENVIRONMENT):
        return Severity.DEGRADED
    if consecutive_failures >= 3:
        return Severity.DEGRADED
    if diagnosis and diagnosis.problem_type in (ProblemType.CODE, ProblemType.DEPLOY):
        return Severity.WARNING
    return Severity.WARNING


def _make_dedup_key(workspace: str, diagnosis: Optional[ProblemDiagnosis]) -> str:
    """Generate deduplication key for incident grouping."""
    ptype = diagnosis.problem_type.value if diagnosis else "unknown"
    step = diagnosis.failed_step_label if diagnosis else ""
    return f"{workspace}:{ptype}:{step}"


class WorkspaceMonitor:
    """Proactive health monitor for workspaces with operations config.

    Runs verify pipelines on a schedule, detects state transitions,
    and orchestrates auto-heal + notification with severity awareness.
    """

    def __init__(
        self,
        shell: ShellExecutor,
        verify: VerifyPipeline,
        diagnostics: DiagnosticsCollector,
        *,
        check_interval_seconds: float = 300.0,
        max_heal_attempts: int = 2,
        guardrails: Optional[AutonomyGuardrails] = None,
    ) -> None:
        self.shell = shell
        self.verify = verify
        self.diagnostics = diagnostics
        self.check_interval = check_interval_seconds
        self.max_heal_attempts = max_heal_attempts
        self.guardrails = guardrails or AutonomyGuardrails()

        self._health: Dict[str, WorkspaceHealth] = {}
        self._running = False
        self._task: Optional[asyncio.Task[None]] = None
        self._recent_heal_timestamps: List[float] = []
        self._recent_dedup_keys: Dict[str, float] = {}  # key -> last_notified_at
        self._notification_cooldown = 300.0  # 5 min between same notifications

        # Callbacks
        self._on_notify: Optional[NotifyCallback] = None
        self._on_save_operation: Optional[
            Callable[..., Coroutine[Any, Any, None]]
        ] = None
        self._on_save_incident: Optional[
            Callable[[Incident], Coroutine[Any, Any, None]]
        ] = None
        self._load_active_incidents: Optional[
            Callable[[List[str]], Coroutine[Any, Any, List[Dict[str, Any]]]]
        ] = None

        # Profiles to monitor (set externally)
        self._profiles: List[Any] = []

    def set_profiles(self, profiles: List[Any]) -> None:
        self._profiles = profiles

    def set_notify_callback(self, callback: NotifyCallback) -> None:
        self._on_notify = callback

    def set_save_callback(
        self,
        callback: Callable[..., Coroutine[Any, Any, None]],
    ) -> None:
        self._on_save_operation = callback

    def set_incident_callback(
        self,
        callback: Callable[[Incident], Coroutine[Any, Any, None]],
    ) -> None:
        self._on_save_incident = callback

    def set_active_incidents_loader(
        self,
        callback: Callable[[List[str]], Coroutine[Any, Any, List[Dict[str, Any]]]],
    ) -> None:
        self._load_active_incidents = callback

    async def start(self) -> None:
        if self._running:
            return
        await self._restore_active_incidents()
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
            if h.active_incident and h.active_incident.is_active
        ]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        # Brief initial delay for system stabilization
        await asyncio.sleep(30)
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
            health.last_healthy_at = now

            if not was_healthy:
                await self._on_recovery(health, profile)
        else:
            health.healthy = False
            health.consecutive_failures += 1

            if was_healthy:
                await self._on_new_failure(health, profile, report)
            elif health.active_incident and health.active_incident.is_active:
                await self._try_auto_heal(health, profile)

    async def _restore_active_incidents(self) -> None:
        """Restore active incidents after bot restart."""
        if not self._load_active_incidents or not self._profiles:
            return

        workspace_paths = [
            str(getattr(profile, "root_path", ""))
            for profile in self._profiles
            if getattr(profile, "root_path", None)
        ]
        if not workspace_paths:
            return

        try:
            rows = await self._load_active_incidents(workspace_paths)
        except Exception:
            logger.warning("Failed to restore active incidents from storage")
            return

        restored = 0
        for row in rows:
            workspace_path = row.get("workspace_path")
            state_value = row.get("state")
            severity_value = row.get("severity")
            if not workspace_path or not state_value or not severity_value:
                continue
            try:
                state = IncidentState(state_value)
                severity = Severity(severity_value)
            except ValueError:
                continue
            details = row.get("details", {})
            if not isinstance(details, dict):
                details = {}
            incident = Incident(
                incident_id=str(row["incident_id"]),
                workspace_path=str(workspace_path),
                state=state,
                severity=severity,
                detected_at=float(row.get("detected_at") or 0.0),
                healed_at=float(row["healed_at"]) if row.get("healed_at") else None,
                heal_attempts=int(row.get("heal_attempts") or 0),
                dedup_key=str(row.get("dedup_key") or ""),
                suppressed_count=int(row.get("suppressed_count") or 0),
                last_error=str(details.get("last_error") or ""),
            )
            health = self._health.setdefault(
                str(workspace_path),
                WorkspaceHealth(
                    workspace_path=str(workspace_path),
                    healthy=False,
                    consecutive_failures=max(1, incident.heal_attempts or 1),
                ),
            )
            health.healthy = False
            health.active_incident = incident
            if incident.dedup_key:
                self._recent_dedup_keys[incident.dedup_key] = time.time()
            restored += 1

        if restored:
            logger.info("Restored active incidents", count=restored)

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    async def _on_new_failure(
        self, health: WorkspaceHealth, profile: Any, report: Any
    ) -> None:
        ops = getattr(profile, "operations", None)
        server_diag = await self.diagnostics.collect(profile, Path(health.workspace_path))
        diagnosis = classify_problem(
            report,
            operations_config=ops,
            server_context=server_diag.as_prompt_context(),
        )

        severity = _classify_severity(
            diagnosis, health.consecutive_failures, service_down=False
        )
        dedup_key = _make_dedup_key(health.workspace_path, diagnosis)

        # Check dedup: reopen existing instead of creating new
        if self._is_duplicate(dedup_key):
            if health.active_incident:
                health.active_incident.suppressed_count += 1
            return

        incident = Incident(
            incident_id=str(uuid.uuid4())[:8],
            workspace_path=health.workspace_path,
            state=IncidentState.DETECTED,
            severity=severity,
            diagnosis=None,  # We'll store ProblemDiagnosis info via to_dict
            detected_at=time.time(),
            max_heal_attempts=self.max_heal_attempts,
            dedup_key=dedup_key,
        )
        # Store diagnosis info in a way compatible with the old interface
        incident._problem_diagnosis = diagnosis  # type: ignore[attr-defined]
        health.active_incident = incident

        await self._persist_incident(incident, "incident_detected")
        self._recent_dedup_keys[dedup_key] = time.time()

        display_name = getattr(profile, "display_name", health.workspace_path)
        emoji = severity.emoji
        lines = [
            f"{emoji} <b>{display_name}: сбой обнаружен</b>",
            f"<b>Серьезность:</b> {severity.label_ru}",
            "",
            f"<b>Тип:</b> {diagnosis.label}",
            f"<b>Шаг:</b> {diagnosis.failed_step_label}",
        ]
        if diagnosis.short_cause:
            lines.append(f"<b>Причина:</b> {diagnosis.short_cause}")
        if diagnosis.runbook_hint:
            lines.append(f"💡 {diagnosis.runbook_hint}")

        if self._should_auto_heal(profile, diagnosis):
            lines.append("\nПробую автоматическое восстановление...")
        else:
            lines.append("\nТребуется ручной разбор.")

        await self._notify("\n".join(lines))

        if self._should_auto_heal(profile, diagnosis):
            await self._try_auto_heal(health, profile)

    async def _on_recovery(self, health: WorkspaceHealth, profile: Any) -> None:
        incident = health.active_incident
        display_name = getattr(profile, "display_name", health.workspace_path)

        if incident:
            incident.transition_to(IncidentState.HEALED, "verify passed")
            incident.healed_at = time.time()
            await self._persist_incident(incident, "incident_healed")

            duration = int(incident.healed_at - incident.detected_at)
            lines = [
                f"🟢 <b>{display_name}: восстановлен</b>",
                "",
            ]
            diag = getattr(incident, "_problem_diagnosis", None)
            if diag:
                lines.append(f"<b>Был сбой:</b> {diag.label}")
            lines.append(f"<b>Длительность:</b> {duration}с")
            if incident.heal_attempts > 0:
                lines.append(f"<b>Авто-починок:</b> {incident.heal_attempts}")
            if incident.suppressed_count > 0:
                lines.append(f"<b>Подавлено дубликатов:</b> {incident.suppressed_count}")
            await self._notify("\n".join(lines))

            # Clear dedup key on recovery
            self._recent_dedup_keys.pop(incident.dedup_key, None)
        else:
            await self._notify(f"🟢 <b>{display_name}: проверки проходят</b>")

        health.active_incident = None

    async def _try_auto_heal(self, health: WorkspaceHealth, profile: Any) -> None:
        incident = health.active_incident
        if not incident:
            return

        # Check guardrails
        if incident.heal_attempts >= incident.max_heal_attempts:
            if incident.state != IncidentState.ESCALATED:
                incident.transition_to(
                    IncidentState.ESCALATED,
                    f"max heal attempts ({incident.max_heal_attempts}) reached",
                )
                await self._persist_incident(incident, "incident_escalated")
                display_name = getattr(profile, "display_name", health.workspace_path)
                await self._notify(
                    f"⚠️ <b>{display_name}: авто-починка исчерпана</b>\n\n"
                    f"Попыток: {incident.heal_attempts}/{incident.max_heal_attempts}\n"
                    "Требуется ручной разбор."
                )
            return

        # Check cooldown
        if incident.cooldown_until > time.time():
            return

        diag = getattr(incident, "_problem_diagnosis", None)
        if not self._should_auto_heal(profile, diag):
            return

        # Check global guardrail limits
        now = time.time()
        window = self.guardrails.heal_window_seconds
        recent_heals = sum(
            1 for t in self._recent_heal_timestamps if now - t < window
        )
        if not self.guardrails.allows_heal(recent_heals):
            logger.info("Heal blocked by global guardrails")
            return

        incident.transition_to(IncidentState.HEALING, "auto-heal attempt")
        incident.heal_attempts += 1
        self._recent_heal_timestamps.append(now)

        # Multi-step heal: status check → restart → wait → health verify
        restart_cmd = self._get_restart_command(profile)
        if not restart_cmd:
            incident.transition_to(IncidentState.ESCALATED, "no restart command")
            incident.last_error = "Нет команды перезапуска"
            await self._persist_incident(incident, "incident_escalated")
            return

        workspace_root = Path(health.workspace_path)

        # Step 1: Pre-restart health check
        health_cmd = self._get_health_command(profile)
        if health_cmd:
            pre_check = await self.shell.execute(
                workspace_root=workspace_root,
                command=health_cmd,
                timeout_seconds=10,
            )
            if pre_check.success:
                # Service is actually up — transient failure, set cooldown
                incident.transition_to(
                    IncidentState.DETECTED, "service already healthy, cooldown"
                )
                incident.cooldown_until = now + self.guardrails.heal_cooldown_seconds
                await self._persist_incident(incident, "heal_attempted")
                return

        # Step 2: Restart
        result = await self.shell.execute(
            workspace_root=workspace_root,
            command=restart_cmd,
            timeout_seconds=30,
        )

        if not result.success:
            incident.last_error = result.stderr_text or result.error or "restart failed"
            incident.transition_to(IncidentState.ESCALATED, "restart command failed")
            await self._persist_incident(incident, "heal_failed")
            display_name = getattr(profile, "display_name", health.workspace_path)
            await self._notify(
                f"❌ <b>{display_name}: авто-перезапуск не удался</b>\n\n"
                f"Ошибка: {incident.last_error[:120]}"
            )
            return

        # Step 3: Wait for service to stabilize
        await asyncio.sleep(5)

        # Step 4: Post-restart health verify
        if health_cmd:
            check = await self.shell.execute(
                workspace_root=workspace_root,
                command=health_cmd,
                timeout_seconds=15,
            )
            if check.success:
                incident.transition_to(
                    IncidentState.DETECTED, "heal successful, waiting for verify"
                )
                await self._persist_incident(incident, "heal_attempted")
                # Set short cooldown before next heal attempt
                incident.cooldown_until = now + self.guardrails.heal_cooldown_seconds
            else:
                incident.transition_to(
                    IncidentState.ESCALATED,
                    "restart ok but health check failed",
                )
                incident.last_error = "Перезапуск выполнен, но проверка не прошла"
                await self._persist_incident(incident, "heal_failed")

    # ------------------------------------------------------------------
    # Policy helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _should_auto_heal(profile: Any, diagnosis: Optional[ProblemDiagnosis]) -> bool:
        ops = getattr(profile, "operations", None)
        if not ops or not getattr(ops, "self_heal_restart", False):
            return False
        if diagnosis and diagnosis.problem_type.value in ("code", "config", "deploy"):
            return False
        return True

    @staticmethod
    def _get_restart_command(profile: Any) -> Optional[str]:
        for svc in getattr(profile, "services", ()):
            cmd = getattr(svc, "restart_command", None)
            if cmd:
                return cmd
        return None

    @staticmethod
    def _get_health_command(profile: Any) -> Optional[str]:
        for svc in getattr(profile, "services", ()):
            cmd = getattr(svc, "health_command", None) or getattr(
                svc, "status_command", None
            )
            if cmd:
                return cmd
        return None

    def _is_duplicate(self, dedup_key: str) -> bool:
        """Check if this incident pattern was recently notified."""
        last = self._recent_dedup_keys.get(dedup_key)
        if last and time.time() - last < self._notification_cooldown:
            return True
        return False

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
                logger.warning(
                    "Failed to persist incident",
                    incident_id=incident.incident_id,
                )
        if self._on_save_incident:
            try:
                await self._on_save_incident(incident)
            except Exception:
                logger.warning(
                    "Failed to persist incident state",
                    incident_id=incident.incident_id,
                )

    async def _notify(self, text: str) -> None:
        if self._on_notify:
            try:
                await self._on_notify(text)
            except Exception:
                logger.warning("Failed to send monitoring notification")
