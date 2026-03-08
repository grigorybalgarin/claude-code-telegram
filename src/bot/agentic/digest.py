"""Operational digest: short summary of recent events.

Used by /diag, status, and background digest notifications.
Answers: "what happened recently and what needs attention?"
"""

import time
from typing import Any, Dict, List, Optional

from .ops_model import (
    IncidentState,
    OperationalSnapshot,
    Severity,
    SuggestedAction,
    SuggestedActionType,
)


def build_operational_snapshot(
    workspace_path: str,
    display_name: str,
    workspace_state: Dict[str, Optional[Dict]],
    active_incident: Optional[Any] = None,
    service_healthy: Optional[bool] = None,
) -> OperationalSnapshot:
    """Build a point-in-time operational snapshot from persistent state."""
    snap = OperationalSnapshot(
        workspace_path=workspace_path,
        display_name=display_name,
        service_healthy=service_healthy,
    )

    verify = workspace_state.get("verify")
    if verify:
        details = verify.get("details", {}) if isinstance(verify, dict) else {}
        if isinstance(details, dict):
            snap.last_verify_success = details.get("success", verify.get("success"))
            snap.last_verify_at = details.get("timestamp")
        else:
            snap.last_verify_success = verify.get("success")

    resolve = workspace_state.get("resolve")
    if resolve:
        details = resolve.get("details", {}) if isinstance(resolve, dict) else {}
        if isinstance(details, dict):
            snap.last_resolve_success = details.get("success", resolve.get("success"))
            snap.last_resolve_at = details.get("timestamp")
        else:
            snap.last_resolve_success = resolve.get("success")

    deploy = workspace_state.get("deploy")
    if deploy:
        details = deploy.get("details", {}) if isinstance(deploy, dict) else {}
        if isinstance(details, dict):
            snap.last_deploy_success = details.get(
                "overall_success", deploy.get("success")
            )
            snap.last_deploy_at = details.get("timestamp")
        else:
            snap.last_deploy_success = deploy.get("success")

    snap.active_incident = active_incident

    # Determine overall health
    snap.overall_health = _assess_health(snap)

    # Compute unresolved issues
    snap.unresolved_issues = _collect_issues(snap)

    # Suggest next action
    snap.suggested_action = suggest_next_action(snap)

    return snap


def suggest_next_action(
    snapshot: OperationalSnapshot,
    diagnosis: Optional[Any] = None,
    remediation_plan: Optional[Any] = None,
) -> Optional[SuggestedAction]:
    """Determine the most useful next step based on current state.

    Enhanced version: considers diagnosis type, confidence, caution level,
    and runbook hints when available.
    """
    # Active incident takes priority
    if snapshot.active_incident:
        inc = snapshot.active_incident
        state = getattr(inc, "state", None)
        if isinstance(state, IncidentState):
            if state == IncidentState.ESCALATED:
                return SuggestedAction(
                    action_type=SuggestedActionType.MANUAL_CHECK,
                    reason="Инцидент эскалирован, требуется ручной разбор",
                )
            if state in (IncidentState.DETECTED, IncidentState.INVESTIGATING):
                # Check if remediation plan says caution
                if remediation_plan:
                    from .remediation_planner import CautionLevel

                    if remediation_plan.caution_level == CautionLevel.BLOCK:
                        return SuggestedAction(
                            action_type=SuggestedActionType.MANUAL_CHECK,
                            reason="Инцидент требует ручного разбора (авто-исправление не рекомендуется)",
                        )
                    if remediation_plan.caution_level == CautionLevel.HIGH:
                        hint = remediation_plan.next_step_hint or "проверь вручную"
                        return SuggestedAction(
                            action_type=SuggestedActionType.MANUAL_CHECK,
                            reason=f"Инцидент: {hint}",
                        )

                return SuggestedAction(
                    action_type=SuggestedActionType.RESOLVE,
                    reason="Обнаружен активный инцидент",
                    button_label="Разберись",
                )

    # Service down
    if snapshot.service_healthy is False:
        return SuggestedAction(
            action_type=SuggestedActionType.SELF_HEAL,
            reason="Сервис не активен",
            auto_executable=True,
        )

    # Last verify failed — use diagnosis for smarter suggestion
    if snapshot.last_verify_success is False:
        if diagnosis:
            confidence = getattr(diagnosis, "confidence", 1.0)
            needs_caution = getattr(diagnosis, "needs_caution", False)

            if needs_caution or confidence < 0.4:
                return SuggestedAction(
                    action_type=SuggestedActionType.MANUAL_CHECK,
                    reason="Проверка не прошла — рекомендуется ручной анализ",
                )

            if remediation_plan and not remediation_plan.safe_auto_resolve:
                return SuggestedAction(
                    action_type=SuggestedActionType.MANUAL_CHECK,
                    reason="Проверка не прошла — автоматическое исправление не рекомендуется",
                )

        return SuggestedAction(
            action_type=SuggestedActionType.RESOLVE,
            reason="Последняя проверка не прошла",
            button_label="Разберись",
        )

    # Last deploy failed
    if snapshot.last_deploy_success is False:
        return SuggestedAction(
            action_type=SuggestedActionType.ROLLBACK,
            reason="Последний deploy не удался",
        )

    # Everything ok
    if snapshot.last_verify_success is True and snapshot.service_healthy is not False:
        return SuggestedAction(
            action_type=SuggestedActionType.NONE,
            reason="Все в порядке",
        )

    # No data — suggest verify
    if snapshot.last_verify_success is None:
        return SuggestedAction(
            action_type=SuggestedActionType.VERIFY,
            reason="Нет данных о последней проверке",
            button_label="Проверить",
        )

    return None


def format_operational_digest(
    snapshots: List[OperationalSnapshot],
    recent_incidents: Optional[List[Dict]] = None,
    improvement_digest: str = "",
) -> str:
    """Format a human-readable operational digest."""
    if not snapshots:
        return "Нет данных для дайджеста."

    lines = ["<b>Оперативная сводка</b>", ""]

    for snap in snapshots:
        health = snap.health_emoji()
        lines.append(f"{health} <b>{snap.display_name}</b>")

        parts = []
        if snap.service_healthy is True:
            parts.append("сервис ок")
        elif snap.service_healthy is False:
            parts.append("сервис НЕ активен")

        if snap.last_verify_success is True:
            parts.append("verify ✓")
        elif snap.last_verify_success is False:
            parts.append("verify ✗")

        if snap.last_deploy_success is True:
            parts.append("deploy ✓")
        elif snap.last_deploy_success is False:
            parts.append("deploy ✗")

        if parts:
            lines.append(f"  {', '.join(parts)}")

        if snap.active_incident:
            inc = snap.active_incident
            state = getattr(inc, "state", None)
            if isinstance(state, IncidentState):
                lines.append(f"  Инцидент: {state.value}")

        if (
            snap.suggested_action
            and snap.suggested_action.action_type != SuggestedActionType.NONE
        ):
            lines.append(f"  → {snap.suggested_action.reason}")

        if snap.unresolved_issues:
            for issue in snap.unresolved_issues[:2]:
                lines.append(f"  ⚡ {issue}")

        lines.append("")

    # Recent incidents summary
    if recent_incidents:
        healed = [
            i for i in recent_incidents if i.get("operation_type") == "incident_healed"
        ]
        escalated = [
            i
            for i in recent_incidents
            if i.get("operation_type") == "incident_escalated"
        ]
        if healed:
            lines.append(f"Восстановлено инцидентов: {len(healed)}")
        if escalated:
            lines.append(f"Эскалировано: {len(escalated)}")

    if improvement_digest:
        lines.append("")
        lines.append(improvement_digest)

    return "\n".join(lines)


def _assess_health(snap: OperationalSnapshot) -> Severity:
    """Determine overall health severity."""
    if snap.active_incident:
        inc = snap.active_incident
        sev = getattr(inc, "severity", None)
        if isinstance(sev, Severity):
            return sev
        state = getattr(inc, "state", None)
        if isinstance(state, IncidentState) and state == IncidentState.ESCALATED:
            return Severity.CRITICAL

    if snap.service_healthy is False:
        return Severity.CRITICAL

    if snap.last_verify_success is False:
        return Severity.DEGRADED

    if snap.last_deploy_success is False:
        return Severity.WARNING

    return Severity.INFO


def _collect_issues(snap: OperationalSnapshot) -> List[str]:
    """Collect unresolved issues from snapshot."""
    issues = []
    if snap.service_healthy is False:
        issues.append("Сервис не активен")
    if snap.last_verify_success is False:
        issues.append("Проверка не проходит")
    if snap.last_deploy_success is False:
        issues.append("Последний deploy не удался")
    if snap.active_incident:
        state = getattr(snap.active_incident, "state", None)
        if isinstance(state, IncidentState) and state == IncidentState.ESCALATED:
            issues.append("Эскалированный инцидент")
    return issues


def format_ago(timestamp: Optional[float]) -> str:
    """Format a timestamp as relative time in Russian."""
    if not timestamp:
        return "—"
    ago = int(time.time() - timestamp)
    if ago < 60:
        return f"{ago}с назад"
    if ago < 3600:
        return f"{ago // 60}м назад"
    if ago < 86400:
        return f"{ago // 3600}ч назад"
    return f"{ago // 86400}д назад"
