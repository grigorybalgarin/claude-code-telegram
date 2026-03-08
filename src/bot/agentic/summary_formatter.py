"""Unified summary formatting for all operational outputs.

Single source of truth for how verify, resolve, service, and digest
summaries look. Enforces consistent structure:

1. Что случилось (what happened)
2. Тип проблемы (problem type)
3. Что сделано / не сделано (what was done)
4. Текущее состояние (current state)
5. Следующий шаг (next step)

Keeps success summaries short and failure summaries informative.
"""

from typing import List, Optional

from .context import ShellActionResult, VerifyReport
from .problem_classifier import ProblemDiagnosis, ProblemType
from .remediation_planner import CautionLevel, RemediationPlan

# ── Verify Summary ─────────────────────────────────────────────────


def format_verify_summary(
    report: VerifyReport,
    diagnosis: ProblemDiagnosis,
    rel_path: str,
    plan: Optional[RemediationPlan] = None,
) -> str:
    """Format a verify summary with unified structure."""
    total = len(report.results)
    passed = sum(1 for _, r in report.results if r.success)

    if report.success:
        return f"Все проверки пройдены ({passed}/{total})\nПроект: {rel_path}"

    lines: List[str] = []
    # 1. What happened
    critical = " [критичный]" if diagnosis.is_critical_step else ""
    lines.append(f"<b>{diagnosis.label}</b>")
    lines.append("")
    lines.append(
        f"<b>Что не так:</b> шаг '{diagnosis.failed_step_label}'"
        f"{critical} не прошел"
    )

    # 2. Cause
    if diagnosis.short_cause:
        lines.append(f"<b>Причина:</b> {diagnosis.short_cause}")

    # 3. Progress
    lines.append(f"<b>Прогресс:</b> {passed}/{total} шагов пройдено")

    # 4. Confidence note
    if plan and plan.confidence_note:
        lines.append(f"<b>Точность:</b> {plan.confidence_note}")

    # 5. Next step
    if plan:
        lines.append(f"\n<b>Следующий шаг:</b> {plan.next_step_hint}")
        if plan.runbook_note and plan.caution_level != CautionLevel.HIGH:
            # HIGH caution already includes hint in next_step
            lines.append(f"💡 {plan.runbook_note}")
    else:
        lines.append(_fallback_next_step(diagnosis))
        # Show runbook hint from diagnosis when no plan
        if diagnosis.runbook_hint:
            lines.append(f"\n💡 <b>Подсказка:</b> {diagnosis.runbook_hint}")

    return "\n".join(lines)


# ── Resolve Summary ────────────────────────────────────────────────


def format_resolve_summary(
    diagnosis: ProblemDiagnosis,
    success: bool,
    attempts: int,
    rollback: bool,
    error: Optional[str],
    passed: int,
    total: int,
    plan: Optional[RemediationPlan] = None,
) -> str:
    """Format a resolve summary with unified structure."""
    lines: List[str] = []

    if success:
        lines.append(f"<b>Исправлено</b> (попыток: {attempts})")
        lines.append("")
        lines.append(
            f"<b>Что было:</b> {diagnosis.label} "
            f"на шаге '{diagnosis.failed_step_label}'"
        )
        if diagnosis.short_cause:
            lines.append(f"<b>Причина:</b> {diagnosis.short_cause}")
        lines.append(f"<b>Результат:</b> все проверки пройдены ({passed}/{total})")
    elif rollback:
        lines.append(f"<b>Откат выполнен</b> (попыток: {attempts})")
        lines.append("")
        lines.append(
            f"<b>Что было:</b> {diagnosis.label} "
            f"на шаге '{diagnosis.failed_step_label}'"
        )
        lines.append(
            "<b>Что сделано:</b> автоматическое исправление не помогло, "
            "изменения откачены"
        )
        lines.append(
            f"<b>Осталось:</b> проблема '{diagnosis.failed_step_label}' "
            "все еще требует внимания"
        )
        if plan and plan.next_step_hint:
            lines.append(f"\n<b>Следующий шаг:</b> {plan.next_step_hint}")
    elif error:
        lines.append("<b>Не удалось разобраться</b>")
        lines.append("")
        lines.append(f"<b>Ошибка:</b> {error}")
        if diagnosis.failed_step_label:
            lines.append(f"<b>Шаг:</b> {diagnosis.failed_step_label}")
    else:
        lines.append(f"<b>Не добил</b> (попыток: {attempts})")
        lines.append("")
        lines.append(
            f"<b>Что было:</b> {diagnosis.label} "
            f"на шаге '{diagnosis.failed_step_label}'"
        )
        lines.append(
            "<b>Что сделано:</b> Claude попытался исправить, "
            "но проверка все еще не проходит"
        )
        lines.append(f"<b>Прогресс:</b> {passed}/{total} шагов проходят")
        lines.append("<b>Осталось:</b> требуется ручной разбор")

    # Confidence note for non-success cases
    if not success and plan and plan.confidence_note:
        lines.append(f"\n<b>Точность:</b> {plan.confidence_note}")

    return "\n".join(lines)


# ── Service Action Summary ─────────────────────────────────────────


def format_service_summary(
    service_name: str,
    action: str,
    success: bool,
    main_result: ShellActionResult,
    checks_ok: bool,
    checks: Optional[list] = None,
) -> str:
    """Format a service action summary with unified structure."""
    if success:
        lines = [f"<b>{service_name}: {action} выполнено</b>"]
        if checks:
            check_passed = sum(1 for _, r in checks if r.success)
            lines.append(f"Проверки: {check_passed}/{len(checks)} ок")
        return "\n".join(lines)

    lines = [f"<b>{service_name}: {action} не удалось</b>", ""]
    if main_result.success and not checks_ok:
        lines.append(
            "<b>Что не так:</b> команда выполнена, " "но пост-проверка не прошла"
        )
    elif main_result.timed_out:
        lines.append("<b>Что не так:</b> превышено время ожидания")
    elif main_result.error:
        lines.append(f"<b>Что не так:</b> {main_result.error}")
    else:
        lines.append(f"<b>Что не так:</b> код выхода {main_result.returncode}")

    return "\n".join(lines)


# ── Diagnosis Summary (for incidents/digest) ───────────────────────


def format_diagnosis_summary(
    diagnosis: ProblemDiagnosis,
    plan: Optional[RemediationPlan] = None,
) -> str:
    """Short diagnosis summary for incidents, digest, and status."""
    lines = [f"<b>{diagnosis.label}</b>"]

    if diagnosis.failed_step_label:
        critical = " [критичный]" if diagnosis.is_critical_step else ""
        lines.append(f"Шаг: {diagnosis.failed_step_label}{critical}")

    if diagnosis.short_cause:
        lines.append(f"Причина: {diagnosis.short_cause}")

    if plan:
        if plan.confidence_note:
            lines.append(f"⚠ {plan.confidence_note}")
        if plan.caution_level in (CautionLevel.HIGH, CautionLevel.BLOCK):
            lines.append("⚠ Требуется осторожность")

    return "\n".join(lines)


def _fallback_next_step(diagnosis: ProblemDiagnosis) -> str:
    """Fallback next step when no remediation plan is available."""
    if diagnosis.safe_to_autofix:
        return (
            "\n<b>Следующий шаг:</b> нажми «Разберись» "
            "для автоматического исправления"
        )
    if diagnosis.problem_type == ProblemType.ENVIRONMENT:
        return (
            "\n<b>Внимание:</b> проблема в серверном окружении, "
            "автоматическое исправление может быть рискованным"
        )
    if diagnosis.problem_type == ProblemType.SERVICE:
        return "\n<b>Следующий шаг:</b> проверь логи сервиса и его состояние"
    return "\n<b>Следующий шаг:</b> запусти проверку для диагностики"
