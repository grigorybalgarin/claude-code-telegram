"""Runbook-driven remediation planning layer.

Transforms runbook hints from passive text annotations into active
planning input: safe actions, caution flags, resolve framing, and
suggested next steps. Does NOT execute anything — only plans.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .problem_classifier import ProblemDiagnosis, ProblemType


class CautionLevel(Enum):
    """How cautious the resolve/action should be."""

    NONE = "none"  # Safe to auto-resolve
    LOW = "low"  # Auto-resolve ok, mention caution
    MODERATE = "moderate"  # Auto-resolve ok with extra checks
    HIGH = "high"  # Suggest manual check first
    BLOCK = "block"  # Do NOT auto-resolve


@dataclass(frozen=True)
class RemediationPlan:
    """Structured plan for how to handle a diagnosed problem.

    Computed from diagnosis + runbook hints + confidence.
    Used by resolve prompt, summaries, and suggested next action.
    """

    caution_level: CautionLevel
    safe_auto_resolve: bool
    resolve_framing: str  # Guidance injected into resolve prompt
    next_step_hint: str  # What to tell the user as "next step"
    runbook_note: str = ""  # Cleaned runbook hint for display
    confidence_note: str = ""  # Note about low confidence if applicable
    extra_checks: List[str] = field(default_factory=list)


# ── Problem type → default caution mapping ─────────────────────────

_DEFAULT_CAUTION: Dict[ProblemType, CautionLevel] = {
    ProblemType.CODE: CautionLevel.NONE,
    ProblemType.CONFIG: CautionLevel.LOW,
    ProblemType.DEPENDENCY: CautionLevel.NONE,
    ProblemType.DEPLOY: CautionLevel.LOW,
    ProblemType.SERVICE: CautionLevel.HIGH,
    ProblemType.ENVIRONMENT: CautionLevel.BLOCK,
    ProblemType.UNKNOWN: CautionLevel.MODERATE,
}

# Runbook hint keywords that raise caution
_CAUTION_KEYWORDS = (
    "вручную",
    "manual",
    "осторожно",
    "careful",
    "опасно",
    "danger",
    "не автоматизируй",
    "не трогай",
    "don't",
    "backup",
    "бэкап",
)

# Runbook hint keywords that lower caution (known safe actions)
_SAFE_KEYWORDS = (
    "make install",
    "pip install",
    "npm install",
    "restart",
    "перезапусти",
    "systemctl restart",
    "запусти",
)


def build_remediation_plan(
    diagnosis: ProblemDiagnosis,
    runbook_hints: Optional[Dict[str, str]] = None,
) -> RemediationPlan:
    """Build a remediation plan from diagnosis and runbook context.

    Args:
        diagnosis: Structured diagnosis from problem_classifier.
        runbook_hints: Full runbook_hints dict from operations config.
    """
    hints = runbook_hints or {}

    # Resolve the best matching runbook hint
    runbook_note = _resolve_hint(diagnosis, hints)

    # Determine caution level
    caution = _compute_caution(diagnosis, runbook_note)

    # Determine if auto-resolve is safe
    safe_auto = (
        caution in (CautionLevel.NONE, CautionLevel.LOW) and diagnosis.safe_to_autofix
    )

    # Build resolve framing (injected into Claude prompt)
    framing = _build_framing(diagnosis, runbook_note, caution)

    # Build next step hint
    next_step = _build_next_step(diagnosis, caution, runbook_note)

    # Confidence note
    confidence_note = ""
    if diagnosis.confidence < 0.4:
        confidence_note = "Низкая уверенность в диагнозе — проверь вручную"
    elif diagnosis.confidence < 0.6:
        confidence_note = "Диагноз неточный — может потребоваться ручная проверка"

    # Extra checks
    extra_checks = _suggest_extra_checks(diagnosis, runbook_note)

    return RemediationPlan(
        caution_level=caution,
        safe_auto_resolve=safe_auto,
        resolve_framing=framing,
        next_step_hint=next_step,
        runbook_note=runbook_note,
        confidence_note=confidence_note,
        extra_checks=extra_checks,
    )


def _resolve_hint(
    diagnosis: ProblemDiagnosis,
    hints: Dict[str, str],
) -> str:
    """Find the best matching runbook hint for this diagnosis."""
    # Direct match on problem type
    hint = hints.get(diagnosis.problem_type.value, "")
    if hint:
        return hint

    # Match on failed step label
    hint = hints.get(diagnosis.failed_step_label, "")
    if hint:
        return hint

    # Already resolved by classify_problem
    if diagnosis.runbook_hint:
        return diagnosis.runbook_hint

    return ""


def _compute_caution(
    diagnosis: ProblemDiagnosis,
    runbook_note: str,
) -> CautionLevel:
    """Determine caution level from diagnosis + runbook keywords."""
    base = _DEFAULT_CAUTION.get(diagnosis.problem_type, CautionLevel.MODERATE)

    # Runbook caution keywords raise the level
    note_lower = runbook_note.lower()
    if any(kw in note_lower for kw in _CAUTION_KEYWORDS):
        base = max(base, CautionLevel.HIGH, key=lambda c: list(CautionLevel).index(c))

    # Runbook safe keywords can lower caution (but not below LOW)
    if any(kw in note_lower for kw in _SAFE_KEYWORDS):
        if base == CautionLevel.HIGH:
            base = CautionLevel.MODERATE

    # Low confidence raises caution
    if diagnosis.confidence < 0.4:
        if base.value in ("none", "low"):
            base = CautionLevel.MODERATE

    # Critical step raises caution
    if diagnosis.is_critical_step and base == CautionLevel.NONE:
        base = CautionLevel.LOW

    return base


def _build_framing(
    diagnosis: ProblemDiagnosis,
    runbook_note: str,
    caution: CautionLevel,
) -> str:
    """Build resolve prompt framing based on plan."""
    parts: List[str] = []

    if runbook_note:
        parts.append(f"Подсказка из runbook: {runbook_note}")

    if caution == CautionLevel.HIGH:
        parts.append(
            "ВНИМАНИЕ: действуй осторожно. Перед исправлением убедись, "
            "что понимаешь причину. Не делай опасных системных изменений."
        )
    elif caution == CautionLevel.BLOCK:
        parts.append(
            "СТОП: автоматическое исправление не рекомендуется. "
            "Объясни причину и предложи план, но не вноси изменения без подтверждения."
        )
    elif caution == CautionLevel.MODERATE:
        parts.append(
            "Будь аккуратен: диагноз может быть неточным. "
            "Проверь гипотезу перед правкой."
        )

    if diagnosis.is_critical_step:
        parts.append(
            "Это критичный шаг — убедись, что исправление не создает новых проблем."
        )

    if diagnosis.confidence < 0.4:
        parts.append(
            "Уверенность диагноза низкая — начни с изучения проблемы, " "а не с правок."
        )

    return "\n".join(parts)


def _build_next_step(
    diagnosis: ProblemDiagnosis,
    caution: CautionLevel,
    runbook_note: str,
) -> str:
    """Build user-facing next step hint."""
    if caution == CautionLevel.BLOCK:
        return "Требуется ручная проверка — автоматическое исправление не рекомендуется"

    if caution == CautionLevel.HIGH:
        if runbook_note:
            return f"Проверь вручную: {_truncate(runbook_note, 80)}"
        return "Проверь логи и состояние сервиса вручную"

    if diagnosis.safe_to_autofix:
        return "Нажми «Разберись» для автоматического исправления"

    if diagnosis.problem_type == ProblemType.SERVICE:
        return "Проверь логи сервиса и его состояние"

    return "Запусти проверку для диагностики"


def _suggest_extra_checks(
    diagnosis: ProblemDiagnosis,
    runbook_note: str,
) -> List[str]:
    """Suggest extra verification steps based on context."""
    checks: List[str] = []
    if diagnosis.problem_type == ProblemType.SERVICE:
        checks.append("Проверь состояние сервиса")
    if diagnosis.problem_type == ProblemType.ENVIRONMENT:
        checks.append("Проверь ресурсы сервера (диск, память)")
    if diagnosis.is_critical_step:
        checks.append("Повторно запусти полную проверку после исправления")
    return checks


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"
