"""Autonomous resolve: diagnose, fix, and re-verify a workspace."""

import structlog

from .context import AgenticWorkspaceContext, ResolveResult, VerifyReport
from .verify_pipeline import VerifyPipeline

logger = structlog.get_logger()


class ResolveRunner:
    """Diagnose a failing workspace, ask Claude to fix it, and re-verify.

    Pure execution layer — no Telegram dependencies. Returns a structured
    ResolveResult that the orchestrator uses for display and audit.
    """

    def __init__(self, verify: VerifyPipeline):
        self.verify = verify

    async def run(
        self,
        ctx: AgenticWorkspaceContext,
        user_id: int,
        session_id: str | None,
        initial_report: VerifyReport,
    ) -> ResolveResult:
        """Execute the full resolve cycle.

        1. Build a prompt from the failing verification step
        2. Create a git checkpoint if available
        3. Call Claude to fix the issue
        4. Re-run verification
        5. Rollback if verification still fails

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
        failing_result = initial_report.results[-1][1]
        failure_output = (
            failing_result.stderr_text
            or failing_result.stdout_text
            or failing_result.error
            or "нет подробного вывода"
        )

        prompt = ctx.project_automation.build_general_autopilot_prompt(
            (
                "Разберись с проблемой в проекте и исправь ее.\n"
                f"Сейчас не проходит шаг проверки '{failed_step.label}'.\n"
                f"Команда шага: {failed_step.command}\n"
                f"Вывод ошибки:\n{failure_output}\n\n"
                "Действуй автономно:\n"
                "1. Определи точную причину сбоя.\n"
                "2. Если проблема в коде или конфигурации проекта, внеси минимальные правки.\n"
                "3. Если проблема только в окружении сервера, не делай опасных инфраструктурных "
                "изменений молча; объясни точную причину и предложи безопасное исправление.\n"
                "4. После правок самостоятельно прогони релевантные проверки снова.\n"
                "5. Заверши коротким отчетом: причина, что исправлено, итог проверок, что осталось."
            ),
            ctx.profile,
        )

        checkpoint = None
        rollback_report = None
        claude_response = None

        try:
            if ctx.change_guard and ctx.profile.has_git_repo:
                checkpoint = await ctx.change_guard.create_checkpoint(
                    ctx.profile.root_path
                )

            claude_response = await ctx.claude_integration.run_command(
                prompt=prompt,
                working_directory=ctx.profile.root_path,
                user_id=user_id,
                session_id=session_id,
                force_new=False,
            )

            if ctx.storage:
                try:
                    await ctx.storage.save_claude_interaction(
                        user_id=user_id,
                        session_id=claude_response.session_id,
                        prompt="[button] resolve",
                        response=claude_response,
                        ip_address=None,
                    )
                except Exception as e:
                    logger.warning("Failed to log resolve interaction", error=str(e))

            final_report = await self.verify.execute(ctx.profile)

            if checkpoint and ctx.change_guard:
                if final_report.success:
                    await ctx.change_guard.cleanup_checkpoint(checkpoint)
                else:
                    rollback_report = await ctx.change_guard.rollback(
                        checkpoint,
                        reason=f"resolve verification failed: {final_report.failed_step.command}",
                    )

            return ResolveResult(
                initial_failure=failed_step,
                claude_response=claude_response,
                final_report=final_report,
                rollback_report=rollback_report,
                success=final_report.success,
                checkpoint_created=checkpoint is not None,
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
            )
