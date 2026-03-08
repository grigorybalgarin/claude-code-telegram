"""Autonomous resolve: diagnose, fix, and re-verify a workspace."""

from pathlib import Path
from typing import Callable, Coroutine, Optional

import structlog

from .context import (
    AgenticWorkspaceContext,
    ResolveResult,
    VerifyReport,
)
from .problem_classifier import ProblemDiagnosis, ProblemType, classify_problem
from .remediation_planner import RemediationPlan, build_remediation_plan
from .shell_executor import ShellExecutor
from .verify_pipeline import VerifyPipeline

logger = structlog.get_logger()

MAX_RESOLVE_ATTEMPTS = 2

# Type-specific guidance for the resolve prompt
_TYPE_GUIDANCE = {
    ProblemType.CODE: (
        "   Это проблема в коде. Внеси минимальные точечные правки — "
        "не рефактори лишнего.\n"
    ),
    ProblemType.CONFIG: (
        "   Это проблема конфигурации. Проверь конфиги, env-переменные, "
        "пути — не трогай бизнес-логику.\n"
    ),
    ProblemType.DEPENDENCY: (
        "   Это проблема зависимостей. Установи или обнови нужный пакет, "
        "но не меняй версии без необходимости.\n"
    ),
    ProblemType.SERVICE: (
        "   Это проблема сервиса. Проверь логи, статус, порты. "
        "Не делай опасных инфраструктурных изменений.\n"
    ),
    ProblemType.DEPLOY: (
        "   Это проблема сборки/деплоя. Проверь build-конфиг, "
        "типы, импорты.\n"
    ),
    ProblemType.ENVIRONMENT: (
        "   Это проблема серверного окружения. Объясни причину и "
        "предложи безопасное исправление, не делай опасных изменений молча.\n"
    ),
    ProblemType.UNKNOWN: "",
}


class ResolveRunner:
    """Diagnose a failing workspace, ask Claude to fix it, and re-verify.

    Pure execution layer -- no Telegram dependencies. Returns a structured
    ResolveResult that the orchestrator uses for display and audit.
    """

    def __init__(self, verify: VerifyPipeline):
        self.verify = verify
        self._shell = ShellExecutor()

    async def run(
        self,
        ctx: AgenticWorkspaceContext,
        user_id: int,
        session_id: str | None,
        initial_report: VerifyReport,
        on_progress: Optional[Callable[[str], Coroutine]] = None,
        server_context: str = "",
    ) -> ResolveResult:
        """Execute the full resolve cycle with up to MAX_RESOLVE_ATTEMPTS.

        1. Build a rich prompt from the failing verification step
        2. Create a git checkpoint if available
        3. Call Claude to fix the issue
        4. Re-run verification
        5. If still failing, give Claude the new error and retry
        6. Rollback if verification still fails after all attempts

        Args:
            ctx: Workspace context with all deps.
            user_id: Telegram user id.
            session_id: Current Claude session id for resume.
            initial_report: The verification report that triggered resolve.
            on_progress: Optional async callback(status_text) for UI updates.

        Returns ResolveResult with all structured data.
        """
        if not ctx.project_automation or not ctx.profile:
            return ResolveResult(
                initial_failure=None,
                claude_response=None,
                final_report=None,
                rollback_report=None,
                success=False,
                error="Project automation unavailable",
            )

        if initial_report.success:
            return ResolveResult(
                initial_failure=None,
                claude_response=None,
                final_report=initial_report,
                rollback_report=None,
                success=True,
            )

        failed_step = initial_report.failed_step
        checkpoint = None
        rollback_report = None
        claude_response = None
        attempt = 0

        try:
            if ctx.change_guard and ctx.profile.has_git_repo:
                checkpoint = await ctx.change_guard.create_checkpoint(
                    ctx.profile.root_path
                )

            current_report = initial_report
            current_session_id = session_id
            ops_config = getattr(ctx.profile, "operations", None)
            diagnosis = classify_problem(
                initial_report,
                operations_config=ops_config,
                server_context=server_context,
            )

            for attempt in range(1, MAX_RESOLVE_ATTEMPTS + 1):
                is_retry = attempt > 1

                if on_progress:
                    if is_retry:
                        await on_progress(
                            f"Попытка {attempt}/{MAX_RESOLVE_ATTEMPTS}: "
                            f"повторно исправляю '{current_report.failed_step.label}'"
                        )
                    else:
                        await on_progress(
                            f"Исправляю '{current_report.failed_step.label}'"
                        )

                if is_retry:
                    diagnosis = classify_problem(
                        current_report,
                        operations_config=ops_config,
                        server_context=server_context,
                    )

                prompt = self._build_prompt(
                    ctx, current_report, diagnosis=diagnosis, is_retry=is_retry
                )

                claude_response = await ctx.claude_integration.run_command(
                    prompt=prompt,
                    working_directory=ctx.profile.root_path,
                    user_id=user_id,
                    session_id=current_session_id,
                    force_new=False,
                )
                current_session_id = claude_response.session_id

                if ctx.storage:
                    try:
                        await ctx.storage.save_claude_interaction(
                            user_id=user_id,
                            session_id=claude_response.session_id,
                            prompt=f"[button] resolve attempt {attempt}",
                            response=claude_response,
                            ip_address=None,
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to log resolve interaction", error=str(e)
                        )

                if on_progress:
                    await on_progress("Проверяю результат исправления")

                final_report = await self.verify.execute(ctx.profile)

                if final_report.success:
                    if checkpoint and ctx.change_guard:
                        await ctx.change_guard.cleanup_checkpoint(checkpoint)
                    return ResolveResult(
                        initial_failure=failed_step,
                        claude_response=claude_response,
                        final_report=final_report,
                        rollback_report=None,
                        success=True,
                        checkpoint_created=checkpoint is not None,
                        attempts=attempt,
                    )

                current_report = final_report

            # All attempts exhausted — rollback
            if checkpoint and ctx.change_guard:
                rollback_report = await ctx.change_guard.rollback(
                    checkpoint,
                    reason=(
                        f"resolve verification failed after {attempt} attempts: "
                        f"{current_report.failed_step.command}"
                    ),
                )

            return ResolveResult(
                initial_failure=failed_step,
                claude_response=claude_response,
                final_report=current_report,
                rollback_report=rollback_report,
                success=False,
                checkpoint_created=checkpoint is not None,
                attempts=attempt,
            )

        except Exception as e:
            if checkpoint and ctx.change_guard:
                rollback_report = await ctx.change_guard.rollback(
                    checkpoint,
                    reason=f"resolve error: {type(e).__name__}",
                )
            return ResolveResult(
                initial_failure=failed_step,
                claude_response=claude_response,
                final_report=None,
                rollback_report=rollback_report,
                success=False,
                checkpoint_created=checkpoint is not None,
                error=str(e),
                attempts=max(attempt, 1),
            )

    def _build_prompt(
        self,
        ctx: AgenticWorkspaceContext,
        report: VerifyReport,
        diagnosis: ProblemDiagnosis | None = None,
        is_retry: bool = False,
    ) -> str:
        """Build a rich diagnostic prompt from the verification report."""
        failed_step = report.failed_step
        failing_result = report.results[-1][1]

        # Collect both stdout and stderr
        output_parts = []
        if failing_result.stderr_text:
            output_parts.append(f"stderr:\n{failing_result.stderr_text}")
        if failing_result.stdout_text:
            output_parts.append(f"stdout:\n{failing_result.stdout_text}")
        if failing_result.error:
            output_parts.append(f"error: {failing_result.error}")
        failure_output = "\n\n".join(output_parts) or "нет подробного вывода"

        # Context from passing steps
        passing_context = ""
        passing_steps = [
            (step, result)
            for step, result in report.results
            if result.success
        ]
        if passing_steps:
            passed_labels = ", ".join(step.label for step, _ in passing_steps)
            passing_context = f"Уже проходят: {passed_labels}.\n"

        # Diagnosis context
        diagnosis_context = ""
        if diagnosis and diagnosis.problem_type != ProblemType.UNKNOWN:
            diagnosis_context = f"Предварительная классификация: {diagnosis.label}.\n"
            if diagnosis.short_cause:
                diagnosis_context += f"Возможная причина: {diagnosis.short_cause}\n"

        # Type-specific guidance
        type_guidance = _TYPE_GUIDANCE.get(
            diagnosis.problem_type if diagnosis else ProblemType.UNKNOWN, ""
        )

        # Build remediation plan for policy-aware framing
        ops_config = getattr(ctx.profile, "operations", None)
        runbook_hints = getattr(ops_config, "runbook_hints", None) if ops_config else None
        plan: Optional[RemediationPlan] = None
        remediation_context = ""
        if diagnosis:
            plan = build_remediation_plan(diagnosis, runbook_hints)
            if plan.resolve_framing:
                remediation_context = plan.resolve_framing + "\n"

        # Logs from the failing step if available
        logs_context = ""
        if report.logs_result:
            log_text = (
                report.logs_result.stdout_text
                or report.logs_result.stderr_text
                or ""
            )
            if log_text:
                logs_context = f"\nЛоги сервиса:\n{log_text}\n"

        # Server diagnostics context
        server_diag_context = ""
        if diagnosis and diagnosis.server_context:
            server_diag_context = diagnosis.server_context

        # Git diff for recent changes context
        git_context = ""
        if ctx.profile.has_git_repo:
            git_context = self._get_git_context(ctx.profile.root_path)

        if is_retry:
            user_request = (
                "Предыдущая попытка исправления не помогла.\n"
                f"{passing_context}"
                f"{diagnosis_context}"
                f"{remediation_context}"
                f"Шаг проверки '{failed_step.label}' все еще не проходит.\n"
                f"Команда: {failed_step.command}\n"
                f"Вывод ошибки:\n{failure_output}\n"
                f"{logs_context}"
                f"{server_diag_context}"
                f"{git_context}\n"
                "Действуй автономно:\n"
                "1. Внимательно прочитай новый вывод ошибки — возможно, ты внес новый баг.\n"
                "2. Проверь свои предыдущие правки и исправь их.\n"
                "3. Прогони проверку снова.\n"
                "4. Заверши коротким отчетом."
            )
        else:
            user_request = (
                "Разберись с проблемой в проекте и исправь ее.\n"
                f"{passing_context}"
                f"{diagnosis_context}"
                f"{remediation_context}"
                f"Сейчас не проходит шаг проверки '{failed_step.label}'.\n"
                f"Команда шага: {failed_step.command}\n"
                f"Вывод ошибки:\n{failure_output}\n"
                f"{logs_context}"
                f"{server_diag_context}"
                f"{git_context}\n"
                "Действуй автономно:\n"
                "1. Сначала составь краткий план исправления (2-3 шага).\n"
                "2. Определи точную причину сбоя.\n"
                f"{type_guidance}"
                "3. После правок самостоятельно прогони релевантные проверки снова.\n"
                "4. Заверши коротким отчетом: причина, что исправлено, итог проверок, что осталось."
            )

        return ctx.project_automation.build_general_autopilot_prompt(
            user_request, ctx.profile
        )

    def _get_git_context(self, root_path: Path) -> str:
        """Get recent git changes for diagnostic context (sync, best-effort)."""
        import subprocess

        parts = []
        try:
            diff = subprocess.run(
                ["git", "diff", "--stat", "HEAD~3..HEAD"],
                cwd=root_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if diff.returncode == 0 and diff.stdout.strip():
                parts.append(
                    f"Недавние изменения (git diff --stat HEAD~3..HEAD):\n"
                    f"{diff.stdout.strip()}"
                )
        except Exception:
            pass

        try:
            status = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=root_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if status.returncode == 0 and status.stdout.strip():
                parts.append(
                    f"Незакоммиченные изменения (git status):\n"
                    f"{status.stdout.strip()}"
                )
        except Exception:
            pass

        if parts:
            return "\n" + "\n\n".join(parts) + "\n"
        return ""
