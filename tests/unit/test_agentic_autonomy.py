"""Tests for maintenance loop and monitoring persistence helpers."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.bot.agentic.autonomy import (
    AutonomyTracker,
    ImprovementBacklog,
    MaintenanceLoop,
    SelfReviewEngine,
)
from src.bot.agentic.monitoring import WorkspaceMonitor
from src.bot.agentic.ops_model import AutonomyGuardrails
from src.bot.agentic.server_diagnostics import DiagnosticsCollector
from src.bot.agentic.shell_executor import ShellExecutor
from src.bot.agentic.verify_pipeline import VerifyPipeline


@pytest.mark.asyncio
async def test_maintenance_loop_restores_pending_improvements_on_start():
    """Pending improvements should survive restart via load callback."""
    guardrails = AutonomyGuardrails()
    loop = MaintenanceLoop(
        guardrails=guardrails,
        review_engine=SelfReviewEngine(guardrails),
        backlog=ImprovementBacklog(),
        tracker=AutonomyTracker(guardrails),
    )

    async def load(limit: int):
        assert limit == 50
        return [
            {
                "improvement_id": "imp-restore",
                "improvement_type": "runbook_hint",
                "description": "Restore runbook hint backlog item",
                "category": "service",
                "confidence": 0.8,
                "priority": 4,
                "safe_to_auto_apply": 0,
                "status": "pending",
                "details": {
                    "source_incidents": ["inc-1"],
                    "requires_user_approval": True,
                    "suggested_change": "Add a runbook note",
                    "created_at": 1234.0,
                },
            }
        ]

    loop.set_improvement_load_callback(load)

    await loop.start()
    try:
        pending = loop.backlog.get_pending()
        assert len(pending) == 1
        assert pending[0].improvement_id == "imp-restore"
        assert pending[0].source_incident_ids == ["inc-1"]
    finally:
        await loop.stop()


@pytest.mark.asyncio
async def test_maintenance_loop_persists_candidates_and_runs_cleanup():
    """Self-review should persist candidates and trigger retention cleanup."""
    guardrails = AutonomyGuardrails()
    loop = MaintenanceLoop(
        guardrails=guardrails,
        review_engine=SelfReviewEngine(guardrails),
        backlog=ImprovementBacklog(),
        tracker=AutonomyTracker(guardrails),
    )

    async def get_recent_ops():
        return [
            {
                "workspace_path": "/srv/app",
                "operation_type": "verify",
                "success": False,
                "details": {"problem_type": "service"},
            },
            {
                "workspace_path": "/srv/app",
                "operation_type": "verify",
                "success": False,
                "details": {"problem_type": "service"},
            },
            {
                "workspace_path": "/srv/app",
                "operation_type": "verify",
                "success": False,
                "details": {"problem_type": "service"},
            },
        ]

    persisted_improvements = []
    saved_operations = []
    cleanup_calls = []

    async def save_improvement(candidate):
        persisted_improvements.append(candidate.improvement_id)

    async def save_operation(**kwargs):
        saved_operations.append(kwargs)

    async def cleanup(days: int):
        cleanup_calls.append(days)
        return {
            "sessions_cleaned": 0,
            "operations_cleaned": 1,
            "incidents_cleaned": 2,
            "improvements_cleaned": 3,
        }

    loop.set_ops_callback(get_recent_ops)
    loop.set_improvement_save_callback(save_improvement)
    loop.set_save_callback(save_operation)
    loop.set_cleanup_callback(cleanup)

    candidates = await loop.run_review_now()

    assert candidates
    assert persisted_improvements
    assert cleanup_calls == [30]
    assert any(
        item["operation_type"] == "maintenance_cleanup"
        for item in saved_operations
    )
    assert any(
        item["operation_type"] == "self_review"
        for item in saved_operations
    )


@pytest.mark.asyncio
async def test_workspace_monitor_restores_active_incidents_on_start(tmp_path):
    """Monitoring should rehydrate active incidents from persistent storage."""
    shell = ShellExecutor()
    monitor = WorkspaceMonitor(
        shell=shell,
        verify=VerifyPipeline(shell),
        diagnostics=DiagnosticsCollector(shell),
        check_interval_seconds=300.0,
    )
    profile = SimpleNamespace(root_path=Path(tmp_path), display_name="ClaudeBot")
    monitor.set_profiles([profile])

    async def load_incidents(workspaces):
        assert workspaces == [str(tmp_path)]
        return [
            {
                "incident_id": "inc-restored",
                "workspace_path": str(tmp_path),
                "state": "detected",
                "severity": "warning",
                "dedup_key": f"{tmp_path}:service:health",
                "detected_at": 1000.0,
                "healed_at": None,
                "heal_attempts": 1,
                "suppressed_count": 0,
                "details": {"last_error": "Connection refused"},
            }
        ]

    monitor.set_active_incidents_loader(load_incidents)

    await monitor.start()
    try:
        health = monitor.health_states[str(tmp_path)]
        assert health.active_incident is not None
        assert health.active_incident.incident_id == "inc-restored"
        assert health.active_incident.last_error == "Connection refused"
    finally:
        await monitor.stop()
