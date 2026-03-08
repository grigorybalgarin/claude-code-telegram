"""Autonomous maintenance, self-review, and self-improvement engine.

Provides controlled autonomy: the bot can monitor, heal, learn from
incidents, and propose improvements — all within strict policy guardrails.
"""

import asyncio
import json
import time
import uuid
from typing import Any, Callable, Coroutine, Dict, List, Optional

import structlog

from .ops_model import (
    AutonomyGuardrails,
    ImprovementCandidate,
    ImprovementType,
)

logger = structlog.get_logger()

# Callback types
NotifyCallback = Callable[[str], Coroutine[Any, Any, None]]
SaveCallback = Callable[..., Coroutine[Any, Any, None]]
SaveImprovementCallback = Callable[[ImprovementCandidate], Coroutine[Any, Any, None]]
LoadImprovementsCallback = Callable[[int], Coroutine[Any, Any, List[Dict[str, Any]]]]
CleanupCallback = Callable[[int], Coroutine[Any, Any, Dict[str, int]]]


class AutonomyTracker:
    """Tracks autonomous action counts for guardrail enforcement."""

    def __init__(self, guardrails: AutonomyGuardrails) -> None:
        self.guardrails = guardrails
        self._heal_timestamps: List[float] = []
        self._improvement_timestamps: List[float] = []
        self._restart_counts: Dict[str, int] = {}  # workspace -> count

    def record_heal(self, workspace: str) -> None:
        now = time.time()
        self._heal_timestamps.append(now)
        self._restart_counts[workspace] = self._restart_counts.get(workspace, 0) + 1

    def record_improvement(self) -> None:
        self._improvement_timestamps.append(time.time())

    def can_heal(self, workspace: str) -> bool:
        self._prune_old_timestamps()
        if not self.guardrails.allows_heal(len(self._heal_timestamps)):
            return False
        if self._restart_counts.get(workspace, 0) >= self.guardrails.max_consecutive_restarts:
            return False
        return True

    def can_improve(self) -> bool:
        self._prune_old_timestamps()
        today_count = sum(
            1 for t in self._improvement_timestamps
            if time.time() - t < 86400
        )
        return self.guardrails.allows_improvement(today_count)

    def reset_restart_count(self, workspace: str) -> None:
        """Reset after successful recovery."""
        self._restart_counts.pop(workspace, None)

    def _prune_old_timestamps(self) -> None:
        now = time.time()
        window = self.guardrails.heal_window_seconds
        self._heal_timestamps = [
            t for t in self._heal_timestamps if now - t < window
        ]


class SelfReviewEngine:
    """Analyzes recent operational history to find improvement opportunities."""

    def __init__(self, guardrails: AutonomyGuardrails) -> None:
        self.guardrails = guardrails

    def review_incidents(
        self,
        recent_ops: List[Dict[str, Any]],
    ) -> List[ImprovementCandidate]:
        """Analyze recent operations and generate improvement candidates."""
        candidates: List[ImprovementCandidate] = []

        # Group by workspace and operation type
        by_workspace: Dict[str, List[Dict]] = {}
        for op in recent_ops:
            ws = op.get("workspace_path", "")
            by_workspace.setdefault(ws, []).append(op)

        for workspace, ops in by_workspace.items():
            candidates.extend(self._analyze_workspace_ops(workspace, ops))

        return candidates

    def _analyze_workspace_ops(
        self,
        workspace: str,
        ops: List[Dict[str, Any]],
    ) -> List[ImprovementCandidate]:
        candidates: List[ImprovementCandidate] = []

        # Pattern: repeated verify failures of the same type
        verify_failures = [
            o for o in ops
            if o.get("operation_type") == "verify" and not o.get("success")
        ]
        if len(verify_failures) >= 3:
            details = verify_failures[-1].get("details", {})
            if isinstance(details, str):
                try:
                    details = json.loads(details)
                except (json.JSONDecodeError, TypeError):
                    details = {}
            problem_type = details.get("problem_type", "unknown")
            candidates.append(ImprovementCandidate(
                improvement_id=str(uuid.uuid4())[:8],
                improvement_type=ImprovementType.RUNBOOK_HINT,
                description=(
                    f"Повторяющиеся verify failures ({len(verify_failures)}x) "
                    f"типа '{problem_type}' в {workspace}"
                ),
                category=problem_type,
                confidence=min(0.9, 0.3 + len(verify_failures) * 0.1),
                priority=len(verify_failures),
                safe_to_auto_apply=False,
                requires_user_approval=True,
                created_at=time.time(),
            ))

        # Pattern: resolve failures (attempted but didn't fix)
        resolve_failures = [
            o for o in ops
            if o.get("operation_type") == "resolve" and not o.get("success")
        ]
        if len(resolve_failures) >= 2:
            candidates.append(ImprovementCandidate(
                improvement_id=str(uuid.uuid4())[:8],
                improvement_type=ImprovementType.REMEDIATION_POLICY,
                description=(
                    f"Resolve не справляется ({len(resolve_failures)}x) в {workspace}. "
                    "Возможно, нужен runbook hint или другой подход."
                ),
                confidence=0.6,
                priority=len(resolve_failures) + 1,
                safe_to_auto_apply=False,
                requires_user_approval=True,
                created_at=time.time(),
            ))

        # Pattern: failed self-heal attempts
        heal_failures = [
            o for o in ops
            if o.get("operation_type") in ("heal_failed", "incident_escalated")
        ]
        if len(heal_failures) >= 2:
            candidates.append(ImprovementCandidate(
                improvement_id=str(uuid.uuid4())[:8],
                improvement_type=ImprovementType.REMEDIATION_POLICY,
                description=(
                    f"Self-heal не помогает ({len(heal_failures)}x) в {workspace}. "
                    "Нужна более точная remediation policy."
                ),
                confidence=0.7,
                priority=len(heal_failures) + 2,
                safe_to_auto_apply=False,
                requires_user_approval=True,
                created_at=time.time(),
            ))

        # Pattern: deploy failures
        deploy_failures = [
            o for o in ops
            if o.get("operation_type") == "deploy" and not o.get("success")
        ]
        if deploy_failures:
            candidates.append(ImprovementCandidate(
                improvement_id=str(uuid.uuid4())[:8],
                improvement_type=ImprovementType.PROFILE_FIX,
                description=(
                    f"Deploy failures ({len(deploy_failures)}x) в {workspace}. "
                    "Проверь deploy config и safety gates."
                ),
                confidence=0.5,
                priority=len(deploy_failures),
                safe_to_auto_apply=False,
                requires_user_approval=True,
                created_at=time.time(),
            ))

        return candidates

    def learn_from_outcome(
        self,
        operation_type: str,
        success: bool,
        diagnosis_category: str,
        remediation_type: Optional[str],
    ) -> Optional[ImprovementCandidate]:
        """Learn from a single operation outcome."""
        if success:
            return None  # Only learn from failures for now

        if operation_type == "resolve" and diagnosis_category in ("code", "config"):
            return ImprovementCandidate(
                improvement_id=str(uuid.uuid4())[:8],
                improvement_type=ImprovementType.CLASSIFIER_RULE,
                description=(
                    f"Resolve не справился с {diagnosis_category} — "
                    "classifier может быть неточным или runbook hint недостаточен"
                ),
                category=diagnosis_category,
                confidence=0.4,
                priority=1,
                safe_to_auto_apply=False,
                requires_user_approval=True,
                created_at=time.time(),
            )
        return None


class ImprovementBacklog:
    """Persistent backlog of self-identified improvements."""

    def __init__(self) -> None:
        self._items: List[ImprovementCandidate] = []

    def add(self, candidate: ImprovementCandidate) -> ImprovementCandidate:
        # Deduplicate by description similarity
        for existing in self._items:
            if (
                existing.improvement_type == candidate.improvement_type
                and existing.category == candidate.category
                and existing.status == "pending"
            ):
                existing.priority = max(existing.priority, candidate.priority)
                existing.confidence = max(existing.confidence, candidate.confidence)
                if candidate.source_incident_ids:
                    existing.source_incident_ids = list({
                        *existing.source_incident_ids,
                        *candidate.source_incident_ids,
                    })
                if candidate.suggested_change and not existing.suggested_change:
                    existing.suggested_change = candidate.suggested_change
                return existing
        self._items.append(candidate)
        return candidate

    def restore(self, candidates: List[ImprovementCandidate]) -> None:
        """Restore persisted candidates without deduping away their IDs."""
        self._items = list(candidates)

    def get_pending(self, limit: int = 10) -> List[ImprovementCandidate]:
        pending = [i for i in self._items if i.status == "pending"]
        return sorted(pending, key=lambda x: -x.priority)[:limit]

    def get_auto_applicable(self) -> List[ImprovementCandidate]:
        return [
            i for i in self._items
            if i.status == "pending" and i.safe_to_auto_apply
        ]

    def mark_applied(self, improvement_id: str) -> None:
        for item in self._items:
            if item.improvement_id == improvement_id:
                item.status = "applied"
                return

    def mark_failed(self, improvement_id: str) -> None:
        for item in self._items:
            if item.improvement_id == improvement_id:
                item.status = "failed"
                return

    def to_list(self) -> List[Dict[str, Any]]:
        return [i.to_dict() for i in self._items]

    def summary_text(self) -> str:
        """Short human-readable summary."""
        pending = [i for i in self._items if i.status == "pending"]
        if not pending:
            return "Нет предложений по улучшению."
        lines = [f"<b>Backlog улучшений:</b> {len(pending)} предложений"]
        for item in sorted(pending, key=lambda x: -x.priority)[:5]:
            lines.append(
                f"  • [{item.improvement_type.value}] {item.description[:80]}"
            )
        return "\n".join(lines)


class MaintenanceLoop:
    """Scheduled maintenance: self-check, self-review, self-improvement.

    Runs as a background async loop. Does NOT modify code by default —
    only accumulates knowledge and proposes improvements.
    """

    def __init__(
        self,
        guardrails: AutonomyGuardrails,
        review_engine: SelfReviewEngine,
        backlog: ImprovementBacklog,
        tracker: AutonomyTracker,
        *,
        review_interval_seconds: float = 3600.0,  # 1 hour
    ) -> None:
        self.guardrails = guardrails
        self.review = review_engine
        self.backlog = backlog
        self.tracker = tracker
        self.review_interval = review_interval_seconds
        self._running = False
        self._task: Optional[asyncio.Task[None]] = None
        self._on_notify: Optional[NotifyCallback] = None
        self._get_recent_ops: Optional[
            Callable[..., Coroutine[Any, Any, List[Dict]]]
        ] = None
        self._save_callback: Optional[SaveCallback] = None
        self._save_improvement_callback: Optional[SaveImprovementCallback] = None
        self._load_improvements_callback: Optional[LoadImprovementsCallback] = None
        self._cleanup_callback: Optional[CleanupCallback] = None
        self.last_review_at: float = 0.0
        self.last_cleanup_at: float = 0.0
        self.cleanup_interval_seconds: float = 86400.0
        self.cleanup_retention_days: int = 30

    def set_notify_callback(self, callback: NotifyCallback) -> None:
        self._on_notify = callback

    def set_ops_callback(
        self, callback: Callable[..., Coroutine[Any, Any, List[Dict]]]
    ) -> None:
        self._get_recent_ops = callback

    def set_save_callback(self, callback: SaveCallback) -> None:
        self._save_callback = callback

    def set_improvement_save_callback(
        self,
        callback: SaveImprovementCallback,
    ) -> None:
        self._save_improvement_callback = callback

    def set_improvement_load_callback(
        self,
        callback: LoadImprovementsCallback,
    ) -> None:
        self._load_improvements_callback = callback

    def set_cleanup_callback(self, callback: CleanupCallback) -> None:
        self._cleanup_callback = callback

    async def start(self) -> None:
        if self._running:
            return
        await self._restore_backlog()
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Maintenance loop started")

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
        logger.info("Maintenance loop stopped")

    async def run_review_now(self) -> List[ImprovementCandidate]:
        """Run a self-review cycle immediately (for /diag or testing)."""
        return await self._do_review()

    async def _run_loop(self) -> None:
        # Initial delay to let the system stabilize
        await asyncio.sleep(60)
        while self._running:
            try:
                await self._do_review()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Maintenance review cycle failed")
            await asyncio.sleep(self.review_interval)

    async def _do_review(self) -> List[ImprovementCandidate]:
        """Execute one self-review cycle."""
        if not self._get_recent_ops:
            return []

        try:
            recent_ops = await self._get_recent_ops()
        except Exception:
            logger.warning("Failed to fetch recent ops for review")
            return []

        candidates = self.review.review_incidents(recent_ops)
        for candidate in candidates:
            persisted = self.backlog.add(candidate)
            await self._persist_improvement(persisted)

        self.last_review_at = time.time()
        await self._maybe_cleanup()

        if candidates:
            logger.info(
                "Self-review found improvements",
                count=len(candidates),
                types=[c.improvement_type.value for c in candidates],
            )

            # Persist review results
            if self._save_callback:
                try:
                    await self._save_callback(
                        workspace_path="__system__",
                        operation_type="self_review",
                        success=True,
                        details={
                            "candidates": len(candidates),
                            "types": [c.improvement_type.value for c in candidates],
                            "backlog_size": len(self.backlog.get_pending()),
                        },
                    )
                except Exception:
                    pass

        return candidates

    async def _restore_backlog(self) -> None:
        """Reload pending improvement backlog from persistent storage."""
        if not self._load_improvements_callback:
            return

        try:
            rows = await self._load_improvements_callback(50)
        except Exception:
            logger.warning("Failed to restore improvement backlog")
            return

        restored: List[ImprovementCandidate] = []
        for row in rows:
            try:
                improvement_type = ImprovementType(str(row["improvement_type"]))
            except (KeyError, ValueError):
                continue

            details = row.get("details", {})
            if not isinstance(details, dict):
                details = {}

            restored.append(
                ImprovementCandidate(
                    improvement_id=str(row["improvement_id"]),
                    improvement_type=improvement_type,
                    description=str(row["description"]),
                    source_incident_ids=list(details.get("source_incidents", [])),
                    category=str(row.get("category") or ""),
                    confidence=float(row.get("confidence") or 0.5),
                    priority=int(row.get("priority") or 0),
                    safe_to_auto_apply=bool(row.get("safe_to_auto_apply")),
                    requires_user_approval=bool(
                        details.get("requires_user_approval", True)
                    ),
                    suggested_change=str(details.get("suggested_change") or ""),
                    status=str(row.get("status") or "pending"),
                    created_at=float(details.get("created_at") or time.time()),
                )
            )

        if restored:
            self.backlog.restore(restored)
            logger.info("Restored improvement backlog", count=len(restored))

    async def _persist_improvement(self, candidate: ImprovementCandidate) -> None:
        if not self._save_improvement_callback:
            return
        try:
            await self._save_improvement_callback(candidate)
        except Exception:
            logger.warning(
                "Failed to persist improvement candidate",
                improvement_id=candidate.improvement_id,
            )

    async def _maybe_cleanup(self) -> None:
        """Run periodic retention cleanup for persistent operational state."""
        if not self._cleanup_callback:
            return

        now = time.time()
        if self.last_cleanup_at and (
            now - self.last_cleanup_at < self.cleanup_interval_seconds
        ):
            return

        try:
            counts = await self._cleanup_callback(self.cleanup_retention_days)
        except Exception:
            logger.warning("Maintenance cleanup failed")
            return

        self.last_cleanup_at = now
        if self._save_callback:
            try:
                await self._save_callback(
                    workspace_path="__system__",
                    operation_type="maintenance_cleanup",
                    success=True,
                    details=counts,
                )
            except Exception:
                logger.warning("Failed to persist maintenance cleanup summary")

    def get_digest(self) -> str:
        """Generate a short maintenance digest."""
        lines = []
        pending = self.backlog.get_pending(5)
        if pending:
            lines.append(f"<b>Найдено улучшений:</b> {len(self.backlog.get_pending())}")
            for item in pending:
                lines.append(f"  • {item.description[:80]}")
        else:
            lines.append("Предложений по улучшению пока нет.")

        if self.last_review_at:
            ago = int(time.time() - self.last_review_at)
            if ago < 3600:
                lines.append(f"\nПоследний self-review: {ago // 60}м назад")
            else:
                lines.append(f"\nПоследний self-review: {ago // 3600}ч назад")

        return "\n".join(lines)
