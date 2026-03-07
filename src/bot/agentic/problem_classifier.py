"""Classify workspace problems by type for smarter summaries.

Used by verify and resolve flows to produce human-readable summaries
that answer: what's wrong, what was done, what's ok now, what remains.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .context import ShellActionResult, VerifyReport, VerifyStep


class ProblemType(Enum):
    """High-level problem category."""

    CODE = "code"
    CONFIG = "config"
    DEPENDENCY = "dependency"
    SERVICE = "service"
    DEPLOY = "deploy"
    ENVIRONMENT = "environment"
    UNKNOWN = "unknown"


# Patterns mapped to problem types (checked against combined output)
_PATTERNS = {
    ProblemType.DEPENDENCY: [
        "ModuleNotFoundError",
        "ImportError",
        "No module named",
        "Cannot find module",
        "MODULE_NOT_FOUND",
        "Could not resolve",
        "package not found",
        "not installed",
        "pip install",
        "npm install",
        "missing dependency",
        "peer dep",
        "ENOENT",
        "node_modules",
    ],
    ProblemType.CONFIG: [
        "YAML",
        "yaml",
        "toml",
        "Invalid configuration",
        "config error",
        "missing key",
        "missing field",
        "environment variable",
        "env var",
        ".env",
        "settings",
        "permission denied",
        "PermissionError",
        "EACCES",
    ],
    ProblemType.SERVICE: [
        "Connection refused",
        "ECONNREFUSED",
        "port already in use",
        "EADDRINUSE",
        "service failed",
        "systemctl",
        "systemd",
        "health check",
        "unhealthy",
        "not running",
        "dead",
        "inactive",
        "failed to start",
        "timeout",
        "TimeoutError",
    ],
    ProblemType.DEPLOY: [
        "deploy",
        "build failed",
        "compile error",
        "compilation",
        "webpack",
        "rollup",
        "vite",
        "tsc",
        "type error",
    ],
    ProblemType.ENVIRONMENT: [
        "disk space",
        "out of memory",
        "OOM",
        "killed",
        "signal 9",
        "SIGKILL",
        "no space left",
        "filesystem",
        "socket",
        "network unreachable",
        "DNS",
        "cannot allocate",
    ],
    ProblemType.CODE: [
        "SyntaxError",
        "TypeError",
        "ValueError",
        "AttributeError",
        "NameError",
        "KeyError",
        "IndexError",
        "AssertionError",
        "RuntimeError",
        "ZeroDivisionError",
        "test failed",
        "FAILED",
        "assert",
        "traceback",
        "Traceback",
        "lint",
        "flake8",
        "mypy",
        "eslint",
        "Error:",
        "error:",
    ],
}

# Display labels for problem types
_LABELS = {
    ProblemType.CODE: "Ошибка в коде",
    ProblemType.CONFIG: "Проблема конфигурации",
    ProblemType.DEPENDENCY: "Проблема зависимостей",
    ProblemType.SERVICE: "Проблема сервиса",
    ProblemType.DEPLOY: "Проблема сборки/деплоя",
    ProblemType.ENVIRONMENT: "Проблема окружения сервера",
    ProblemType.UNKNOWN: "Неизвестная проблема",
}


@dataclass(frozen=True)
class ProblemDiagnosis:
    """Structured diagnosis of a workspace problem."""

    problem_type: ProblemType
    label: str
    failed_step_label: str
    short_cause: str
    safe_to_autofix: bool

    @property
    def is_code_fixable(self) -> bool:
        return self.problem_type in {
            ProblemType.CODE,
            ProblemType.CONFIG,
            ProblemType.DEPLOY,
        }


def classify_problem(report: VerifyReport) -> ProblemDiagnosis:
    """Classify the type of problem from a failing verify report."""
    if report.success or not report.failed_step:
        return ProblemDiagnosis(
            problem_type=ProblemType.UNKNOWN,
            label=_LABELS[ProblemType.UNKNOWN],
            failed_step_label="",
            short_cause="",
            safe_to_autofix=False,
        )

    failed_step = report.failed_step
    last_result = report.results[-1][1] if report.results else None
    output = _collect_output(last_result, report.logs_result)

    problem_type = _detect_type(output, failed_step)
    short_cause = _extract_short_cause(output, problem_type)
    safe_to_autofix = problem_type in {
        ProblemType.CODE,
        ProblemType.CONFIG,
        ProblemType.DEPENDENCY,
        ProblemType.DEPLOY,
    }

    return ProblemDiagnosis(
        problem_type=problem_type,
        label=_LABELS[problem_type],
        failed_step_label=failed_step.label,
        short_cause=short_cause,
        safe_to_autofix=safe_to_autofix,
    )


def format_verify_summary(
    report: VerifyReport,
    diagnosis: ProblemDiagnosis,
    rel_path: str,
) -> str:
    """Format a human-readable verify summary answering the 4 questions."""
    total = len(report.results)
    passed = sum(1 for _, r in report.results if r.success)

    if report.success:
        return (
            f"Все проверки пройдены ({passed}/{total})\n"
            f"Проект: {rel_path}"
        )

    lines = [
        f"<b>{diagnosis.label}</b>",
        "",
        f"<b>Что не так:</b> шаг '{diagnosis.failed_step_label}' не прошел",
    ]
    if diagnosis.short_cause:
        lines.append(f"<b>Причина:</b> {diagnosis.short_cause}")
    lines.append(f"<b>Прогресс:</b> {passed}/{total} шагов пройдено")

    if diagnosis.safe_to_autofix:
        lines.append(
            "\n<b>Можно попробовать:</b> нажми «Разберись» для автоматического исправления"
        )
    elif diagnosis.problem_type == ProblemType.ENVIRONMENT:
        lines.append(
            "\n<b>Внимание:</b> проблема в серверном окружении, "
            "автоматическое исправление может быть рискованным"
        )
    elif diagnosis.problem_type == ProblemType.SERVICE:
        lines.append(
            "\n<b>Совет:</b> проверь логи сервиса и его состояние"
        )

    return "\n".join(lines)


def format_resolve_summary(
    diagnosis: ProblemDiagnosis,
    success: bool,
    attempts: int,
    rollback: bool,
    error: Optional[str],
    passed: int,
    total: int,
) -> str:
    """Format a human-readable resolve summary answering the 4 questions."""
    lines: List[str] = []

    if success:
        lines.append(f"<b>Исправлено</b> (попыток: {attempts})")
        lines.append("")
        lines.append(f"<b>Что было:</b> {diagnosis.label} "
                      f"на шаге '{diagnosis.failed_step_label}'")
        if diagnosis.short_cause:
            lines.append(f"<b>Причина:</b> {diagnosis.short_cause}")
        lines.append(f"<b>Результат:</b> все проверки пройдены ({passed}/{total})")
    elif rollback:
        lines.append(f"<b>Откат выполнен</b> (попыток: {attempts})")
        lines.append("")
        lines.append(f"<b>Что было:</b> {diagnosis.label} "
                      f"на шаге '{diagnosis.failed_step_label}'")
        lines.append("<b>Что сделано:</b> автоматическое исправление не помогло, "
                      "изменения откачены")
        lines.append(f"<b>Осталось:</b> проблема '{diagnosis.failed_step_label}' "
                      "все еще требует внимания")
    elif error:
        lines.append("<b>Не удалось разобраться</b>")
        lines.append("")
        lines.append(f"<b>Ошибка:</b> {error}")
        if diagnosis.failed_step_label:
            lines.append(f"<b>Шаг:</b> {diagnosis.failed_step_label}")
    else:
        lines.append(f"<b>Не добил</b> (попыток: {attempts})")
        lines.append("")
        lines.append(f"<b>Что было:</b> {diagnosis.label} "
                      f"на шаге '{diagnosis.failed_step_label}'")
        lines.append("<b>Что сделано:</b> Claude попытался исправить, "
                      "но проверка все еще не проходит")
        lines.append(f"<b>Прогресс:</b> {passed}/{total} шагов проходят")
        lines.append("<b>Осталось:</b> требуется ручной разбор")

    return "\n".join(lines)


def format_service_summary(
    service_name: str,
    action: str,
    success: bool,
    main_result: ShellActionResult,
    checks_ok: bool,
    checks: Optional[list] = None,
) -> str:
    """Format a human-readable service action summary."""
    if success:
        lines = [
            f"<b>{service_name}: {action} выполнено</b>",
        ]
        if checks:
            passed = sum(1 for _, r in checks if r.success)
            lines.append(f"Проверки: {passed}/{len(checks)} ок")
        return "\n".join(lines)

    lines = [
        f"<b>{service_name}: {action} не удалось</b>",
        "",
    ]
    if main_result.success and not checks_ok:
        lines.append("<b>Что не так:</b> команда выполнена, "
                      "но пост-проверка не прошла")
    elif main_result.timed_out:
        lines.append("<b>Что не так:</b> превышено время ожидания")
    elif main_result.error:
        lines.append(f"<b>Что не так:</b> {main_result.error}")
    else:
        lines.append(f"<b>Что не так:</b> код выхода {main_result.returncode}")

    return "\n".join(lines)


def _collect_output(
    result: Optional[ShellActionResult],
    logs_result: Optional[ShellActionResult],
) -> str:
    """Combine all output sources into a single string for pattern matching."""
    parts = []
    if result:
        if result.stderr_text:
            parts.append(result.stderr_text)
        if result.stdout_text:
            parts.append(result.stdout_text)
        if result.error:
            parts.append(result.error)
    if logs_result:
        if logs_result.stdout_text:
            parts.append(logs_result.stdout_text)
        if logs_result.stderr_text:
            parts.append(logs_result.stderr_text)
    return "\n".join(parts)


def _detect_type(output: str, failed_step: VerifyStep) -> ProblemType:
    """Detect problem type from output patterns and step label."""
    scores: dict[ProblemType, int] = {t: 0 for t in ProblemType}

    # Step label hints
    label = failed_step.label.lower()
    if any(w in label for w in ("health", "статус", "проверка")):
        scores[ProblemType.SERVICE] += 2
    if any(w in label for w in ("тест", "test")):
        scores[ProblemType.CODE] += 2
    if any(w in label for w in ("линт", "lint", "typecheck", "mypy")):
        scores[ProblemType.CODE] += 3
    if any(w in label for w in ("сборка", "build")):
        scores[ProblemType.DEPLOY] += 2

    # Pattern matching on output
    for problem_type, patterns in _PATTERNS.items():
        for pattern in patterns:
            if pattern in output:
                scores[problem_type] += 1

    # Pick highest score
    best = max(scores, key=lambda t: scores[t])
    if scores[best] == 0:
        return ProblemType.UNKNOWN
    return best


def _extract_short_cause(output: str, problem_type: ProblemType) -> str:
    """Extract a short human-readable cause from the output."""
    if not output:
        return ""

    lines = output.strip().splitlines()

    # For code errors, find the most informative error line
    if problem_type == ProblemType.CODE:
        for line in reversed(lines):
            stripped = line.strip()
            if any(stripped.startswith(prefix) for prefix in (
                "Error:", "error:", "SyntaxError", "TypeError",
                "ValueError", "AttributeError", "NameError",
                "FAILED", "E ", "AssertionError",
            )):
                return _truncate(stripped, 120)

    # For dependency errors
    if problem_type == ProblemType.DEPENDENCY:
        for line in lines:
            stripped = line.strip()
            if any(kw in stripped for kw in (
                "ModuleNotFoundError", "No module named",
                "Cannot find module", "not installed",
            )):
                return _truncate(stripped, 120)

    # For service errors
    if problem_type == ProblemType.SERVICE:
        for line in lines:
            stripped = line.strip()
            if any(kw in stripped for kw in (
                "Connection refused", "port already",
                "not running", "failed", "inactive",
            )):
                return _truncate(stripped, 120)

    # Fallback: last non-empty line
    for line in reversed(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith(("$", "#", "+")):
            return _truncate(stripped, 120)

    return ""


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
