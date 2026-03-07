"""Unified operational domain model.

Central types used across verify, resolve, deploy, monitor, incidents,
remediation, and autonomy subsystems. This is the single source of truth
for operational concepts.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Severity ────────────────────────────────────────────────────────

class Severity(Enum):
    """Incident / diagnostic severity level."""

    INFO = "info"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {
            Severity.INFO: 0,
            Severity.WARNING: 1,
            Severity.DEGRADED: 2,
            Severity.CRITICAL: 3,
        }[self]

    def __ge__(self, other: "Severity") -> bool:
        return self.rank >= other.rank

    def __gt__(self, other: "Severity") -> bool:
        return self.rank > other.rank

    @property
    def emoji(self) -> str:
        return {
            Severity.INFO: "ℹ️",
            Severity.WARNING: "⚠️",
            Severity.DEGRADED: "🟠",
            Severity.CRITICAL: "🔴",
        }[self]

    @property
    def label_ru(self) -> str:
        return {
            Severity.INFO: "информация",
            Severity.WARNING: "предупреждение",
            Severity.DEGRADED: "деградация",
            Severity.CRITICAL: "критичный",
        }[self]


# ── Operation Outcome ───────────────────────────────────────────────

class OperationOutcome(Enum):
    """Final outcome of any operation."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    RECOVERED = "recovered"
    CANCELED = "canceled"
    STALE = "stale"

    @property
    def is_ok(self) -> bool:
        return self in {OperationOutcome.SUCCESS, OperationOutcome.RECOVERED}


# ── Remediation ─────────────────────────────────────────────────────

class RemediationType(Enum):
    """What kind of fix action."""

    RESTART = "restart"
    ROLLBACK = "rollback"
    CLEANUP = "cleanup"
    CODE_FIX = "code_fix"
    CONFIG_FIX = "config_fix"
    DEPENDENCY_FIX = "dependency_fix"
    MANUAL = "manual"


class RemediationPolicy(Enum):
    """How aggressively the system can auto-remediate."""

    SAFE_AUTO = "safe_auto"       # Run automatically, low risk
    CAUTIOUS_AUTO = "cautious_auto"  # Run automatically with extra checks
    SUGGEST_ONLY = "suggest_only"  # Show to user but don't execute
    NEVER_AUTO = "never_auto"     # Never auto-remediate


@dataclass(frozen=True)
class RemediationAction:
    """A concrete remediation step with policy."""

    action_type: RemediationType
    policy: RemediationPolicy
    command: Optional[str] = None
    description: str = ""
    requires_verify: bool = True
    max_attempts: int = 1
    cooldown_seconds: int = 60

    @property
    def is_auto_allowed(self) -> bool:
        return self.policy in {
            RemediationPolicy.SAFE_AUTO,
            RemediationPolicy.CAUTIOUS_AUTO,
        }


# ── Enriched Diagnosis ──────────────────────────────────────────────

class ProblemCategory(Enum):
    """High-level problem category (superset of ProblemType)."""

    CODE = "code"
    CONFIG = "config"
    DEPENDENCY = "dependency"
    SERVICE = "service"
    DEPLOY = "deploy"
    ENVIRONMENT = "environment"
    TRANSIENT = "transient"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EnrichedDiagnosis:
    """Central diagnostic object used across all operational flows.

    This is the single interpretation of "what went wrong" — used by
    verify, resolve, monitor, deploy, incidents, and remediation.
    """

    category: ProblemCategory
    severity: Severity
    confidence: float                 # 0.0–1.0
    summary_reason: str               # One-line human summary
    root_cause_hint: str = ""         # Best-guess root cause
    failed_step: str = ""             # Which step failed
    is_critical_step: bool = False
    fixability: str = "unknown"       # "auto_fixable", "manual_only", "needs_investigation"
    needs_caution: bool = False
    probable_scope: str = ""          # "single_file", "config", "service", "infrastructure"
    server_context: str = ""
    runbook_hint: str = ""
    suggested_remediation: Optional[RemediationAction] = None

    @property
    def label_ru(self) -> str:
        _labels = {
            ProblemCategory.CODE: "Ошибка в коде",
            ProblemCategory.CONFIG: "Проблема конфигурации",
            ProblemCategory.DEPENDENCY: "Проблема зависимостей",
            ProblemCategory.SERVICE: "Проблема сервиса",
            ProblemCategory.DEPLOY: "Проблема сборки/деплоя",
            ProblemCategory.ENVIRONMENT: "Проблема окружения",
            ProblemCategory.TRANSIENT: "Временный сбой",
            ProblemCategory.UNKNOWN: "Неизвестная проблема",
        }
        return _labels.get(self.category, "Неизвестная проблема")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "summary_reason": self.summary_reason,
            "root_cause_hint": self.root_cause_hint,
            "failed_step": self.failed_step,
            "is_critical_step": self.is_critical_step,
            "fixability": self.fixability,
            "needs_caution": self.needs_caution,
            "runbook_hint": self.runbook_hint,
        }


# ── Incident ────────────────────────────────────────────────────────

class IncidentState(Enum):
    """Lifecycle state of an incident."""

    DETECTED = "detected"
    INVESTIGATING = "investigating"
    HEALING = "healing"
    HEALED = "healed"
    ESCALATED = "escalated"
    SUPPRESSED = "suppressed"


@dataclass
class IncidentTransition:
    """A single state change in incident lifecycle."""

    from_state: IncidentState
    to_state: IncidentState
    timestamp: float
    reason: str = ""


@dataclass
class Incident:
    """Full-lifecycle incident object with severity and dedup."""

    incident_id: str
    workspace_path: str
    state: IncidentState
    severity: Severity = Severity.WARNING
    diagnosis: Optional[EnrichedDiagnosis] = None
    detected_at: float = 0.0
    healed_at: Optional[float] = None
    heal_attempts: int = 0
    max_heal_attempts: int = 1
    last_error: str = ""
    dedup_key: str = ""
    correlation_ids: List[str] = field(default_factory=list)
    transitions: List[IncidentTransition] = field(default_factory=list)
    cooldown_until: float = 0.0
    suppressed_count: int = 0

    def transition_to(self, new_state: IncidentState, reason: str = "") -> None:
        """Record a state transition."""
        import time
        self.transitions.append(IncidentTransition(
            from_state=self.state,
            to_state=new_state,
            timestamp=time.time(),
            reason=reason,
        ))
        self.state = new_state

    @property
    def is_active(self) -> bool:
        return self.state in {
            IncidentState.DETECTED,
            IncidentState.INVESTIGATING,
            IncidentState.HEALING,
        }

    @property
    def duration_seconds(self) -> Optional[float]:
        if not self.detected_at:
            return None
        import time
        end = self.healed_at or time.time()
        return end - self.detected_at

    def to_dict(self) -> Dict[str, Any]:
        diag = self.diagnosis
        # Support both EnrichedDiagnosis (category/summary_reason)
        # and ProblemDiagnosis (problem_type/short_cause)
        category = getattr(diag, "category", None)
        problem_type = getattr(diag, "problem_type", None)
        ptype_val = (
            category.value if category else
            problem_type.value if problem_type else None
        )
        reason = (
            getattr(diag, "summary_reason", None)
            or getattr(diag, "short_cause", None)
            or ""
        ) if diag else ""
        return {
            "incident_id": self.incident_id,
            "workspace_path": self.workspace_path,
            "state": self.state.value,
            "severity": self.severity.value,
            "problem_type": ptype_val,
            "short_cause": reason,
            "detected_at": self.detected_at,
            "healed_at": self.healed_at,
            "heal_attempts": self.heal_attempts,
            "last_error": self.last_error,
            "dedup_key": self.dedup_key,
            "suppressed_count": self.suppressed_count,
            "transitions": len(self.transitions),
        }


# ── Suggested Action ────────────────────────────────────────────────

class SuggestedActionType(Enum):
    """What the bot recommends as next step."""

    RESOLVE = "resolve"
    SELF_HEAL = "self_heal"
    DEPLOY = "deploy"
    VERIFY = "verify"
    ROLLBACK = "rollback"
    MANUAL_CHECK = "manual_check"
    WAIT = "wait"
    NONE = "none"


@dataclass(frozen=True)
class SuggestedAction:
    """What the bot recommends the user (or itself) do next."""

    action_type: SuggestedActionType
    reason: str
    auto_executable: bool = False
    button_label: str = ""

    @property
    def label_ru(self) -> str:
        _labels = {
            SuggestedActionType.RESOLVE: "Разберись",
            SuggestedActionType.SELF_HEAL: "Self-heal",
            SuggestedActionType.DEPLOY: "Deploy",
            SuggestedActionType.VERIFY: "Проверить",
            SuggestedActionType.ROLLBACK: "Откатить",
            SuggestedActionType.MANUAL_CHECK: "Ручная проверка",
            SuggestedActionType.WAIT: "Подожди",
            SuggestedActionType.NONE: "Нет действий",
        }
        return self.button_label or _labels.get(self.action_type, "")


# ── Improvement Backlog ─────────────────────────────────────────────

class ImprovementType(Enum):
    """Category of self-improvement."""

    PROFILE_FIX = "profile_fix"
    RUNBOOK_HINT = "runbook_hint"
    CLASSIFIER_RULE = "classifier_rule"
    SUMMARY_WORDING = "summary_wording"
    REMEDIATION_POLICY = "remediation_policy"
    TEST_GAP = "test_gap"
    CONFIG_CONSISTENCY = "config_consistency"
    DIAGNOSTICS_HINT = "diagnostics_hint"


@dataclass
class ImprovementCandidate:
    """A self-identified improvement the bot could make."""

    improvement_id: str
    improvement_type: ImprovementType
    description: str
    source_incident_ids: List[str] = field(default_factory=list)
    category: str = ""
    confidence: float = 0.5
    priority: int = 0  # higher = more important
    safe_to_auto_apply: bool = False
    requires_user_approval: bool = True
    suggested_change: str = ""
    status: str = "pending"  # pending, applied, rejected, failed
    created_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "improvement_id": self.improvement_id,
            "type": self.improvement_type.value,
            "description": self.description,
            "source_incidents": self.source_incident_ids,
            "confidence": self.confidence,
            "priority": self.priority,
            "safe_to_auto_apply": self.safe_to_auto_apply,
            "status": self.status,
        }


# ── Autonomy Guardrails ────────────────────────────────────────────

@dataclass
class AutonomyGuardrails:
    """Hard limits on autonomous bot behavior."""

    max_heal_attempts_per_window: int = 3
    heal_window_seconds: int = 3600        # 1 hour
    max_improvements_per_day: int = 2
    improvement_cooldown_seconds: int = 7200  # 2 hours
    heal_cooldown_seconds: int = 300       # 5 min between same heal
    max_consecutive_restarts: int = 2
    require_verify_after_heal: bool = True
    require_rollback_on_failure: bool = True
    forbidden_auto_changes: tuple = (
        "security_boundary",
        "deploy_model",
        "permissions",
        "core_flow",
    )

    def allows_heal(self, recent_attempts: int) -> bool:
        return recent_attempts < self.max_heal_attempts_per_window

    def allows_improvement(self, recent_count: int) -> bool:
        return recent_count < self.max_improvements_per_day


# ── Operational Snapshot ────────────────────────────────────────────

@dataclass
class OperationalSnapshot:
    """Point-in-time operational state for status display."""

    workspace_path: str = ""
    display_name: str = ""
    service_healthy: Optional[bool] = None
    last_verify_success: Optional[bool] = None
    last_verify_at: Optional[float] = None
    last_resolve_success: Optional[bool] = None
    last_resolve_at: Optional[float] = None
    last_deploy_success: Optional[bool] = None
    last_deploy_at: Optional[float] = None
    active_incident: Optional[Incident] = None
    unresolved_issues: List[str] = field(default_factory=list)
    suggested_action: Optional[SuggestedAction] = None
    overall_health: Severity = Severity.INFO

    def health_emoji(self) -> str:
        if self.overall_health == Severity.CRITICAL:
            return "🔴"
        if self.overall_health == Severity.DEGRADED:
            return "🟠"
        if self.overall_health == Severity.WARNING:
            return "🟡"
        return "🟢"


# ── Deploy Safety Gate ──────────────────────────────────────────────

@dataclass(frozen=True)
class DeploySafetyGate:
    """A pre-deploy check that must pass."""

    name: str
    check_type: str  # "clean_worktree", "verify_pass", "profile_complete", "service_healthy"
    hard: bool = True  # hard=abort deploy, soft=warn but continue
    description: str = ""


@dataclass
class GateResult:
    """Result of a single safety gate check."""

    gate: DeploySafetyGate
    passed: bool
    message: str = ""


@dataclass
class DeployGateReport:
    """Combined result of all pre-deploy safety checks."""

    results: List[GateResult] = field(default_factory=list)

    @property
    def can_proceed(self) -> bool:
        return all(r.passed for r in self.results if r.gate.hard)

    @property
    def warnings(self) -> List[str]:
        return [r.message for r in self.results if not r.passed and not r.gate.hard]

    @property
    def blockers(self) -> List[str]:
        return [r.message for r in self.results if not r.passed and r.gate.hard]
