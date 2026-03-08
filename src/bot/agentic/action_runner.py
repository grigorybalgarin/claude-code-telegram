"""Telegram action handlers for agentic workspace operations.

Extracted from MessageOrchestrator to reduce file size. Contains all
``_run_agentic_*`` action methods that handle inline/reply keyboard
callbacks for playbooks, shell commands, services, verify, resolve,
and background jobs.

Depends on Telegram types (query, context) — not a pure layer.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from telegram.ext import ContextTypes

from ...config.settings import Settings
from ..features.change_guard import ChangeGuardReport
from ..utils.formatting import ResponseFormatter
from ..utils.html_format import escape_html
from .context import AgenticWorkspaceContext, ShellActionResult, VerifyStep
from .deploy_pipeline import DeployPipeline, DeployProfile
from .panel_builder import PanelBuilder
from .problem_classifier import (
    classify_problem,
    format_resolve_summary,
    format_service_summary,
    format_verify_summary,
)
from .resolve_runner import ResolveRunner
from .server_diagnostics import DiagnosticsCollector
from .service_controller import ServiceController
from .shell_executor import ShellExecutor
from .verify_pipeline import VerifyPipeline

logger = structlog.get_logger()


class ActionRunner:
    """Execute agentic workspace actions triggered by Telegram buttons.

    Thin execution layer — receives resolved workspace context and
    Telegram query/context, runs the action, sends results back.
    """

    def __init__(
        self,
        settings: Settings,
        shell: ShellExecutor,
        verify: VerifyPipeline,
        services: ServiceController,
        resolver: ResolveRunner,
        panel: PanelBuilder,
    ):
        self.settings = settings
        self.shell = shell
        self.verify = verify
        self.services = services
        self.resolver = resolver
        self.panel = panel
        self.deploy = DeployPipeline(shell)
        self.diagnostics = DiagnosticsCollector(shell)

    # ------------------------------------------------------------------
    # Context resolution helpers (shared with orchestrator)
    # ------------------------------------------------------------------

    def _get_boundary_root(self, context: ContextTypes.DEFAULT_TYPE) -> Path:
        return Path(
            context.bot_data.get("boundary_root", self.settings.approved_directory)
        ).resolve()

    def _get_workspace_profile(
        self, context: ContextTypes.DEFAULT_TYPE
    ) -> tuple[Path, Path, Path, Optional[Any], Optional[Any]]:
        """Resolve current directory, workspace root, and detected profile."""
        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        boundary_root = self._get_boundary_root(context)
        features = context.bot_data.get("features")
        project_automation = (
            getattr(features, "get_project_automation", lambda: None)()
            if features
            else None
        )
        if not project_automation:
            return current_dir, current_dir, boundary_root, None, None

        current_dir = Path(current_dir).resolve()
        if current_dir == boundary_root:
            summaries = project_automation.list_workspace_summaries(boundary_root)
            preferred = next(
                (s for s in summaries if s.relative_path != "/"),
                summaries[0] if summaries else None,
            )
            if preferred is not None:
                current_dir = preferred.root_path
                context.user_data["current_directory"] = current_dir

        profile = project_automation.build_profile(current_dir, boundary_root)
        return (
            current_dir,
            profile.root_path,
            boundary_root,
            project_automation,
            profile,
        )

    def _get_operator_runtime(
        self, context: ContextTypes.DEFAULT_TYPE
    ) -> Optional[Any]:
        """Resolve the persistent background operator runtime."""
        features = context.bot_data.get("features")
        return (
            getattr(features, "get_workspace_operator", lambda: None)()
            if features
            else None
        )

    def _build_workspace_context(
        self, context: ContextTypes.DEFAULT_TYPE
    ) -> AgenticWorkspaceContext:
        """Build a rich workspace context from Telegram context."""
        current_dir, current_workspace, boundary_root, project_automation, profile = (
            self._get_workspace_profile(context)
        )
        features = context.bot_data.get("features")
        change_guard = (
            getattr(features, "get_project_change_guard", lambda: None)()
            if features
            else None
        )
        return AgenticWorkspaceContext(
            current_directory=current_dir,
            current_workspace=current_workspace,
            boundary_root=boundary_root,
            project_automation=project_automation,
            profile=profile,
            operator_runtime=self._get_operator_runtime(context),
            claude_integration=context.bot_data.get("claude_integration"),
            storage=context.bot_data.get("storage"),
            audit_logger=context.bot_data.get("audit_logger"),
            change_guard=change_guard,
        )

    # ------------------------------------------------------------------
    # Persistent operational state
    # ------------------------------------------------------------------

    @staticmethod
    async def _save_operation(
        context: Any,
        workspace_path: str,
        operation_type: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Persist an operational event to DB (best-effort)."""
        storage = context.bot_data.get("storage")
        if not storage or not hasattr(storage, "operations"):
            return
        try:
            await storage.operations.save(
                workspace_path=workspace_path,
                operation_type=operation_type,
                success=success,
                details=details,
                correlation_id=correlation_id,
            )
        except Exception as e:
            logger.warning("Failed to persist operation", error=str(e))

    # ------------------------------------------------------------------
    # Action methods
    # ------------------------------------------------------------------

    async def run_playbook(
        self,
        query: Any,
        context: ContextTypes.DEFAULT_TYPE,
        playbook_slug: str,
    ) -> None:
        """Run a deterministic playbook from an agentic control button."""
        user_id = query.from_user.id
        current_dir, current_workspace, boundary_root, project_automation, profile = (
            self._get_workspace_profile(context)
        )
        claude_integration = context.bot_data.get("claude_integration")
        if not claude_integration or not project_automation or not profile:
            await query.edit_message_text("Автоматизация проекта недоступна.")
            return

        playbook = project_automation.get_playbook(playbook_slug, profile)
        if playbook is None:
            await query.answer(
                "Сценарий недоступен для этого проекта.", show_alert=True
            )
            return

        prompt = project_automation.build_playbook_prompt(playbook_slug, profile)
        rel_path = self.panel.format_relative_path(profile.root_path, boundary_root)
        status_msg = await query.message.reply_text(
            "▶️ <b>Запуск сценария</b>\n\n"
            f"Сценарий: <code>{escape_html(playbook.slug)}</code>\n"
            f"Проект: <code>{escape_html(rel_path)}</code>",
            parse_mode="HTML",
        )

        features = context.bot_data.get("features")
        change_guard = (
            getattr(features, "get_project_change_guard", lambda: None)()
            if features
            else None
        )
        storage = context.bot_data.get("storage")
        audit_logger = context.bot_data.get("audit_logger")

        session_id = context.user_data.get("claude_session_id")
        if current_workspace != profile.root_path:
            session_id = None
        context.user_data["current_directory"] = profile.root_path

        mutating = playbook_slug in {"setup", "test", "quality"}
        checkpoint = None
        guard_report = None
        success = False

        try:
            if mutating and change_guard and profile.has_git_repo:
                checkpoint = await change_guard.create_checkpoint(profile.root_path)

            claude_response = await claude_integration.run_command(
                prompt=prompt,
                working_directory=profile.root_path,
                user_id=user_id,
                session_id=session_id,
                force_new=False,
            )
            success = True
            context.user_data["claude_session_id"] = claude_response.session_id

            if storage:
                try:
                    await storage.save_claude_interaction(
                        user_id=user_id,
                        session_id=claude_response.session_id,
                        prompt=f"[button] run {playbook_slug}",
                        response=claude_response,
                        ip_address=None,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to log button playbook interaction", error=str(e)
                    )

            if mutating and change_guard:
                verification_results = await change_guard.run_verification_commands(
                    profile.root_path,
                    project_automation.get_verification_commands(profile),
                )
                guard_report = ChangeGuardReport(
                    checkpoint_created=checkpoint is not None,
                    checkpoint_id=checkpoint.checkpoint_id if checkpoint else None,
                    verification_results=verification_results,
                )
                failed_result = next(
                    (r for r in verification_results if not r.success), None
                )
                if failed_result and checkpoint:
                    success = False
                    rollback_report = await change_guard.rollback(
                        checkpoint,
                        reason=f"verification failed: {failed_result.command}",
                    )
                    guard_report.rollback_triggered = rollback_report.rollback_triggered
                    guard_report.rollback_succeeded = rollback_report.rollback_succeeded
                    guard_report.failure_reason = rollback_report.failure_reason
                    guard_report.rollback_error = rollback_report.rollback_error
                elif checkpoint:
                    await change_guard.cleanup_checkpoint(checkpoint)
            elif checkpoint and change_guard:
                guard_report = ChangeGuardReport(
                    checkpoint_created=True,
                    checkpoint_id=checkpoint.checkpoint_id,
                )
                await change_guard.cleanup_checkpoint(checkpoint)

            await status_msg.delete()

            formatter = ResponseFormatter(self.settings)
            for message in formatter.format_claude_response(claude_response.content):
                if message.text and message.text.strip():
                    await query.message.reply_text(
                        message.text,
                        parse_mode=message.parse_mode,
                    )

            if guard_report and change_guard:
                await query.message.reply_text(
                    change_guard.format_report_html(guard_report),
                    parse_mode="HTML",
                )

            if audit_logger:
                verification_results = []
                if guard_report:
                    verification_results = [
                        {
                            "command": r.command,
                            "success": r.success,
                            "returncode": r.returncode,
                        }
                        for r in guard_report.verification_results
                    ]
                await audit_logger.log_automation_run(
                    user_id=user_id,
                    request=f"button:{playbook_slug}",
                    workspace_root=str(profile.root_path),
                    matched_playbook=playbook_slug,
                    read_only=playbook_slug in {"doctor", "review"},
                    success=success,
                    mode="agentic_button",
                    checkpoint_created=bool(
                        checkpoint or (guard_report and guard_report.checkpoint_created)
                    ),
                    verification_results=verification_results,
                    rollback_triggered=bool(
                        guard_report and guard_report.rollback_triggered
                    ),
                    rollback_succeeded=bool(
                        guard_report and guard_report.rollback_succeeded
                    ),
                    workspace_changed=current_workspace != profile.root_path,
                )
        except Exception as e:
            if checkpoint and change_guard:
                guard_report = await change_guard.rollback(
                    checkpoint,
                    reason=f"button playbook error: {type(e).__name__}",
                )
            try:
                await status_msg.edit_text(
                    "❌ <b>Ошибка сценария</b>\n\n"
                    f"Сценарий: <code>{escape_html(playbook_slug)}</code>\n"
                    f"Ошибка: <code>{escape_html(str(e))}</code>",
                    parse_mode="HTML",
                )
            except Exception:
                pass
            if audit_logger:
                await audit_logger.log_automation_run(
                    user_id=user_id,
                    request=f"button:{playbook_slug}",
                    workspace_root=str(profile.root_path),
                    matched_playbook=playbook_slug,
                    read_only=playbook_slug in {"doctor", "review"},
                    success=False,
                    mode="agentic_button",
                    rollback_triggered=bool(
                        guard_report and guard_report.rollback_triggered
                    ),
                    rollback_succeeded=bool(
                        guard_report and guard_report.rollback_succeeded
                    ),
                    workspace_changed=current_workspace != profile.root_path,
                )

    async def run_shell(
        self,
        query: Any,
        context: ContextTypes.DEFAULT_TYPE,
        workspace_root: Path,
        boundary_root: Path,
        title: str,
        command: str,
        audit_command: str,
        timeout_seconds: int = 120,
    ) -> None:
        """Run a deterministic shell action and report the result in Telegram."""
        user_id = query.from_user.id
        rel_path = self.panel.format_relative_path(workspace_root, boundary_root)
        status_msg = await query.message.reply_text(
            "▶️ <b>Выполнение команды</b>\n\n"
            f"Действие: <code>{escape_html(title)}</code>\n"
            f"Проект: <code>{escape_html(rel_path)}</code>\n"
            f"Команда: <code>{escape_html(command)}</code>",
            parse_mode="HTML",
        )

        result = await self.shell.execute(
            workspace_root=workspace_root,
            command=command,
            timeout_seconds=timeout_seconds,
        )
        await status_msg.edit_text(
            "\n".join(
                self.shell.format_result_lines(
                    title=title,
                    workspace_root=workspace_root,
                    boundary_root=boundary_root,
                    result=result,
                )
            ),
            parse_mode="HTML",
        )

        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=user_id,
                command=audit_command,
                args=[command, str(workspace_root)],
                success=result.success,
            )

    async def run_command(
        self,
        query: Any,
        context: ContextTypes.DEFAULT_TYPE,
        command_key: str,
    ) -> None:
        """Run a direct workspace command for safe operator actions."""
        _current_dir, _current_workspace, boundary_root, project_automation, profile = (
            self._get_workspace_profile(context)
        )
        if not project_automation or not profile:
            await query.edit_message_text("Автоматизация проекта недоступна.")
            return

        command = profile.commands.get(command_key)
        if not command:
            await query.answer("Команда недоступна для этого проекта.", show_alert=True)
            return

        title = {
            "health": "Проверка",
            "build": "Сборка",
        }.get(command_key, command_key.title())
        await self.run_shell(
            query=query,
            context=context,
            workspace_root=profile.root_path,
            boundary_root=boundary_root,
            title=title,
            command=command,
            audit_command=f"workspace_{command_key}",
        )

    async def run_service(
        self,
        query: Any,
        context: ContextTypes.DEFAULT_TYPE,
        service_key: str,
        action_key: str,
    ) -> None:
        """Run a managed service action from the explicit service catalog."""
        (
            _current_dir,
            _current_workspace,
            boundary_root,
            _project_automation,
            profile,
        ) = self._get_workspace_profile(context)
        service = self.services.resolve_service(profile, service_key)
        if not profile or not service:
            await query.answer("Сервис не настроен для этого проекта.", show_alert=True)
            return

        command = service.command_for(action_key)
        if not command:
            await query.answer("Это действие недоступно для сервиса.", show_alert=True)
            return

        user_id = query.from_user.id
        rel_path = self.panel.format_relative_path(profile.root_path, boundary_root)
        status_msg = await query.message.reply_text(
            "▶️ <b>Выполнение действия сервиса</b>\n\n"
            f"Сервис: <code>{escape_html(service.display_name)}</code>\n"
            f"Действие: <code>{escape_html(action_key)}</code>\n"
            f"Проект: <code>{escape_html(rel_path)}</code>\n"
            f"Команда: <code>{escape_html(command)}</code>",
            parse_mode="HTML",
        )

        main_result = await self.shell.execute(
            workspace_root=profile.root_path,
            command=command,
            timeout_seconds=120,
        )

        checks: List[tuple[str, ShellActionResult]] = []
        logs_result: Optional[ShellActionResult] = None
        checks_ok = True
        if action_key in {"start", "restart", "stop"} and main_result.success:
            await status_msg.edit_text(
                "⏳ <b>Ожидание проверки сервиса</b>\n\n"
                f"Сервис: <code>{escape_html(service.display_name)}</code>\n"
                f"Действие: <code>{escape_html(action_key)}</code>",
                parse_mode="HTML",
            )
            follow_up = await self.services.run_follow_up_checks(
                service=service,
                workspace_root=profile.root_path,
                action_key=action_key,
            )
            checks = follow_up.checks
            logs_result = follow_up.logs_result
            checks_ok = follow_up.all_passed
        elif (
            not main_result.success
            and action_key in {"start", "restart"}
            and service.command_for("logs")
        ):
            logs_result = await self.shell.execute(
                profile.root_path,
                service.command_for("logs"),
                timeout_seconds=30,
            )

        final_success = main_result.success and checks_ok

        # Smart summary first
        summary = format_service_summary(
            service_name=service.display_name,
            action=action_key,
            success=final_success,
            main_result=main_result,
            checks_ok=checks_ok,
            checks=checks,
        )

        # Detailed output below summary
        lines = [summary]

        if checks:
            lines.extend(["", "<b>Проверки</b>"])
            for label, result in checks:
                check_state = "ok" if result.success else "ошибка"
                if result.timed_out:
                    check_state = "таймаут"
                lines.append(f"  {label}: {check_state}")

        # Show output only on failure (don't dump stdout on success)
        if not final_success:
            if main_result.stderr_text:
                lines.extend(
                    [
                        "",
                        "<b>Вывод ошибки</b>",
                        f"<pre>{escape_html(main_result.stderr_text)}</pre>",
                    ]
                )
            elif main_result.stdout_text:
                lines.extend(
                    [
                        "",
                        "<b>Вывод</b>",
                        f"<pre>{escape_html(main_result.stdout_text)}</pre>",
                    ]
                )

        if logs_result and (
            logs_result.stdout_text or logs_result.stderr_text or logs_result.error
        ):
            log_body = logs_result.stdout_text or logs_result.stderr_text
            if logs_result.error:
                log_body = logs_result.error
            lines.extend(
                ["", "<b>Логи сервиса</b>", f"<pre>{escape_html(log_body)}</pre>"]
            )

        await status_msg.edit_text("\n".join(lines), parse_mode="HTML")

        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=user_id,
                command=f"service_{service.key}_{action_key}",
                args=[command, str(profile.root_path)],
                success=final_success,
            )

    async def run_verify(
        self,
        query: Any,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Run the deterministic verification suite for the current workspace."""
        _current_dir, _current_workspace, boundary_root, project_automation, profile = (
            self._get_workspace_profile(context)
        )
        if not project_automation or not profile:
            await query.edit_message_text("Автоматизация проекта недоступна.")
            return

        steps = self.verify.build_steps(profile)
        if not steps:
            project_name = profile.display_name if profile else "этого проекта"
            await query.answer(
                f"Для {project_name} полная проверка пока не настроена.",
                show_alert=True,
            )
            return

        rel_path = self.panel.format_relative_path(profile.root_path, boundary_root)
        status_msg = await query.message.reply_text(
            "▶️ <b>Запуск проверки</b>\n\n"
            f"Проект: <code>{escape_html(rel_path)}</code>\n"
            f"Шаги: <code>{escape_html(', '.join(s.label for s in steps))}</code>",
            parse_mode="HTML",
        )

        async def _on_step(index: int, total: int, step: VerifyStep) -> None:
            await status_msg.edit_text(
                "⏳ <b>Выполняю проверку</b>\n\n"
                f"Проект: <code>{escape_html(rel_path)}</code>\n"
                f"Шаг <code>{index}/{total}</code>: "
                f"<code>{escape_html(step.label)}</code>\n"
                f"Команда: <code>{escape_html(step.command)}</code>",
                parse_mode="HTML",
            )

        report = await self.verify.execute(profile, on_step=_on_step)

        # Collect server diagnostics for failing reports
        server_context = ""
        if not report.success:
            server_diag = await self.diagnostics.collect(profile, profile.root_path)
            server_context = server_diag.as_prompt_context()

        ops_config = getattr(profile, "operations", None)
        diagnosis = classify_problem(
            report,
            operations_config=ops_config,
            server_context=server_context,
        )

        # Smart summary + detailed report below
        summary = format_verify_summary(report, diagnosis, rel_path)
        detail = self.verify.format_report(profile, boundary_root, report)
        await status_msg.edit_text(
            f"{summary}\n\n{detail}",
            parse_mode="HTML",
        )

        # Persist last verify result for status display
        import time as _time
        import uuid as _uuid

        correlation_id = str(_uuid.uuid4())[:8]
        verify_data = {
            "success": report.success,
            "failed_step": report.failed_step.label if report.failed_step else None,
            "steps_total": len(report.results),
            "steps_passed": sum(1 for _, r in report.results if r.success),
            "problem_type": diagnosis.problem_type.value,
            "short_cause": diagnosis.short_cause,
            "timestamp": _time.time(),
            "correlation_id": correlation_id,
        }
        context.user_data["last_verify"] = verify_data
        await self._save_operation(
            context,
            str(profile.root_path),
            "verify",
            report.success,
            verify_data,
            correlation_id=correlation_id,
        )

        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=query.from_user.id,
                command="workspace_verify",
                args=[step.command for step in steps],
                success=report.success,
            )

    async def run_resolve(
        self,
        query: Any,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Autonomously investigate, fix, and re-verify the current workspace."""
        user_id = query.from_user.id
        ws_ctx = self._build_workspace_context(context)
        current_workspace = ws_ctx.current_workspace
        profile = ws_ctx.profile
        boundary_root = ws_ctx.boundary_root

        if (
            not ws_ctx.claude_integration
            or not ws_ctx.project_automation
            or not profile
        ):
            await query.edit_message_text("Автоматизация проекта недоступна.")
            return

        steps = self.verify.build_steps(profile)
        if not steps:
            await query.answer(
                f"Для {profile.display_name} нет шагов, с которых можно начать разбор.",
                show_alert=True,
            )
            return

        rel_path = ws_ctx.format_relative_path(profile.root_path)
        status_msg = await query.message.reply_text(
            "🛠 <b>Разбираюсь</b>\n\n"
            f"Проект: <code>{escape_html(rel_path)}</code>\n"
            "Сначала найду проблему, потом попробую исправить ее автоматически.",
            parse_mode="HTML",
        )

        async def _on_resolve_step(index: int, total: int, step: VerifyStep) -> None:
            await status_msg.edit_text(
                "⏳ <b>Проверяю</b>\n\n"
                f"Проект: <code>{escape_html(rel_path)}</code>\n"
                f"Шаг <code>{index}/{total}</code>: "
                f"<code>{escape_html(step.label)}</code>\n"
                f"Команда: <code>{escape_html(step.command)}</code>",
                parse_mode="HTML",
            )

        initial_report = await self.verify.execute(profile, on_step=_on_resolve_step)
        if initial_report.success:
            await status_msg.edit_text(
                "✅ <b>Проблем не найдено</b>\n\n"
                f"Проект: <code>{escape_html(rel_path)}</code>\n"
                "Все детерминированные проверки уже проходят.",
                parse_mode="HTML",
            )
            return

        session_id = context.user_data.get("claude_session_id")
        if current_workspace != profile.root_path:
            session_id = None
        context.user_data["current_directory"] = profile.root_path

        # Collect server diagnostics for context
        server_diag = await self.diagnostics.collect(profile, profile.root_path)
        diag_lines = server_diag.summary_lines()
        diag_hint = ""
        if diag_lines:
            diag_hint = "\n" + "\n".join(f"  {line}" for line in diag_lines)

        await status_msg.edit_text(
            "🛠 <b>Разбираюсь</b>\n\n"
            f"Найден сбой: <code>{escape_html(initial_report.failed_step.label)}</code>\n"
            f"{diag_hint}\n"
            "Пробую исправить автоматически.",
            parse_mode="HTML",
        )

        async def _on_resolve_progress(status_text: str) -> None:
            try:
                await status_msg.edit_text(
                    f"🛠 <b>Разбираюсь</b>\n\n{escape_html(status_text)}",
                    parse_mode="HTML",
                )
            except Exception:
                pass

        result = await self.resolver.run(
            ws_ctx,
            user_id,
            session_id,
            initial_report,
            on_progress=_on_resolve_progress,
            server_context=server_diag.as_prompt_context(),
        )

        if result.claude_response:
            context.user_data["claude_session_id"] = result.claude_response.session_id

        if result.error and not result.claude_response:
            try:
                await status_msg.edit_text(
                    "❌ <b>Не удалось разобраться автоматически</b>\n\n"
                    f"Проект: <code>{escape_html(rel_path)}</code>\n"
                    f"Ошибка: <code>{escape_html(result.error)}</code>",
                    parse_mode="HTML",
                )
            except Exception:
                pass
        else:
            await status_msg.delete()

            if result.claude_response:
                formatter = ResponseFormatter(self.settings)
                for message in formatter.format_claude_response(
                    result.claude_response.content
                ):
                    if message.text and message.text.strip():
                        await query.message.reply_text(
                            message.text,
                            parse_mode=message.parse_mode,
                        )

            if result.final_report:
                diagnosis = classify_problem(initial_report)
                final_passed = sum(
                    1 for _, r in result.final_report.results if r.success
                )
                final_total = len(result.final_report.results)
                resolve_summary = format_resolve_summary(
                    diagnosis=diagnosis,
                    success=result.success,
                    attempts=result.attempts,
                    rollback=bool(
                        result.rollback_report
                        and getattr(result.rollback_report, "rollback_triggered", False)
                    ),
                    error=result.error,
                    passed=final_passed,
                    total=final_total,
                )
                verify_report_text = self.verify.format_report(
                    profile, boundary_root, result.final_report
                )
                await query.message.reply_text(
                    f"{resolve_summary}\n\n{verify_report_text}",
                    parse_mode="HTML",
                )

            if result.rollback_report and ws_ctx.change_guard:
                await query.message.reply_text(
                    ws_ctx.change_guard.format_report_html(result.rollback_report),
                    parse_mode="HTML",
                )

        # Persist last resolve result for status display
        import time as _time
        import uuid as _uuid

        resolve_diagnosis = classify_problem(initial_report)
        correlation_id = str(_uuid.uuid4())[:8]
        resolve_data = {
            "success": result.success,
            "attempts": result.attempts,
            "error": result.error,
            "rollback": bool(
                result.rollback_report
                and getattr(result.rollback_report, "rollback_triggered", False)
            ),
            "problem_type": resolve_diagnosis.problem_type.value,
            "timestamp": _time.time(),
            "correlation_id": correlation_id,
        }
        context.user_data["last_resolve"] = resolve_data
        await self._save_operation(
            context,
            str(profile.root_path),
            "resolve",
            result.success,
            resolve_data,
            correlation_id=correlation_id,
        )

        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_automation_run(
                user_id=user_id,
                request="button:resolve",
                workspace_root=str(profile.root_path),
                matched_playbook=None,
                read_only=False,
                success=result.success,
                mode="agentic_button",
                checkpoint_created=result.checkpoint_created,
                rollback_triggered=bool(
                    result.rollback_report and result.rollback_report.rollback_triggered
                ),
                rollback_succeeded=bool(
                    result.rollback_report and result.rollback_report.rollback_succeeded
                ),
                workspace_changed=current_workspace != profile.root_path,
            )

    async def run_background(
        self,
        query: Any,
        context: ContextTypes.DEFAULT_TYPE,
        action_key: str,
    ) -> None:
        """Launch a long-running workspace command in the background."""
        user_id = query.from_user.id
        _current_dir, current_workspace, boundary_root, project_automation, profile = (
            self._get_workspace_profile(context)
        )
        operator_runtime = self._get_operator_runtime(context)
        if not project_automation or not profile or not operator_runtime:
            await query.edit_message_text("Фоновые задачи недоступны.")
            return

        command = profile.commands.get(action_key)
        if not command:
            await query.answer("Это действие недоступно для проекта.", show_alert=True)
            return

        verification_command = self.verify.select_background_verification(profile)
        verification_mode = None
        verification_delay_seconds = 0.0
        verification_retries = 1
        verification_interval_seconds = 0.0
        if verification_command:
            if action_key in {"start", "dev"}:
                verification_mode = "while_running"
                verification_delay_seconds = 3.0
                verification_retries = 4
                verification_interval_seconds = 3.0
            elif action_key == "deploy":
                verification_mode = "after_exit"
                verification_delay_seconds = 0.0
                verification_retries = 4
                verification_interval_seconds = 3.0

        title = {
            "start": "Запуск",
            "dev": "Разработка",
            "deploy": "Деплой",
        }.get(action_key, action_key.title())

        try:
            job = await operator_runtime.launch_job(
                workspace_root=profile.root_path,
                action_key=action_key,
                command=command,
                title=title,
                verification_command=verification_command,
                verification_mode=verification_mode,
                verification_delay_seconds=verification_delay_seconds,
                verification_retries=verification_retries,
                verification_interval_seconds=verification_interval_seconds,
            )
        except RuntimeError as exc:
            await query.answer(str(exc), show_alert=True)
            ws_ctx = self._build_workspace_context(context)
            text, reply_markup = await self.panel.build_jobs_text(ws_ctx)
            await query.edit_message_text(
                text, parse_mode="HTML", reply_markup=reply_markup
            )
            return

        rel_path = self.panel.format_relative_path(profile.root_path, boundary_root)
        header = (
            "▶️ <b>Фоновая задача запущена</b>\n\n"
            f"Действие: <code>{escape_html(title)}</code>\n"
            f"Проект: <code>{escape_html(rel_path)}</code>\n"
            f"Задача: <code>{escape_html(job.job_id)}</code>\n"
            f"Команда: <code>{escape_html(command)}</code>"
        )
        if verification_command and verification_mode:
            header += (
                "\n"
                f"Проверка после запуска: <code>{escape_html(verification_command)}</code>"
            )
        ws_ctx = self._build_workspace_context(context)
        text, reply_markup = await self.panel.build_jobs_text(ws_ctx, header=header)
        await query.edit_message_text(
            text, parse_mode="HTML", reply_markup=reply_markup
        )

        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=user_id,
                command=f"workspace_job_{action_key}",
                args=[command, str(current_workspace)],
                success=True,
            )

    async def stop_background(
        self,
        query: Any,
        context: ContextTypes.DEFAULT_TYPE,
        job_id: str,
    ) -> None:
        """Stop a persisted background workspace job."""
        operator_runtime = self._get_operator_runtime(context)
        if not operator_runtime:
            await query.edit_message_text("Фоновые задачи недоступны.")
            return

        try:
            job = await operator_runtime.stop_job(job_id)
        except RuntimeError as exc:
            await query.answer(str(exc), show_alert=True)
            return

        header = (
            "🛑 <b>Остановка запрошена</b>\n\n"
            f"Задача: <code>{escape_html(job.job_id)}</code>\n"
            f"Действие: <code>{escape_html(job.action_key)}</code>\n"
            f"Статус: <code>{escape_html(job.status)}</code>"
        )
        ws_ctx = self._build_workspace_context(context)
        text, reply_markup = await self.panel.build_jobs_text(ws_ctx, header=header)
        await query.edit_message_text(
            text, parse_mode="HTML", reply_markup=reply_markup
        )

        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=query.from_user.id,
                command="workspace_job_stop",
                args=[job_id],
                success=True,
            )

    async def run_deploy(
        self,
        query: Any,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Run staged deploy pipeline for the current workspace."""
        user_id = query.from_user.id
        _current_dir, _current_workspace, boundary_root, project_automation, profile = (
            self._get_workspace_profile(context)
        )
        if not project_automation or not profile:
            await query.edit_message_text("Автоматизация проекта недоступна.")
            return

        deploy_profile = DeployProfile.from_workspace_profile(profile, boundary_root)
        if not deploy_profile.restart_command and not deploy_profile.update_command:
            await query.answer("Deploy не настроен для этого проекта.", show_alert=True)
            return

        rel_path = self.panel.format_relative_path(profile.root_path, boundary_root)

        # Pre-deploy safety gates
        gate_report = await self.deploy.check_safety_gates(deploy_profile)
        if not gate_report.can_proceed:
            lines = ["⛔ <b>Deploy заблокирован</b>\n"]
            for gr in gate_report.results:
                icon = "✅" if gr.passed else ("⛔" if gr.gate.hard else "⚠️")
                lines.append(f"  {icon} {gr.gate.description}")
                if gr.message:
                    lines.append(f"      {gr.message}")
            await query.message.reply_text("\n".join(lines), parse_mode="HTML")
            return

        status_msg = await query.message.reply_text(
            "🚀 <b>Запуск deploy</b>\n\n"
            f"Проект: <code>{escape_html(rel_path)}</code>",
            parse_mode="HTML",
        )

        async def _on_stage(stage: Any, label: str) -> None:
            try:
                await status_msg.edit_text(
                    f"🚀 <b>Deploy</b>\n\n"
                    f"Проект: <code>{escape_html(rel_path)}</code>\n"
                    f"Стадия: <code>{escape_html(label)}</code>",
                    parse_mode="HTML",
                )
            except Exception:
                pass

        result = await self.deploy.execute(deploy_profile, on_stage=_on_stage)

        await status_msg.edit_text(
            result.format_summary(),
            parse_mode="HTML",
        )

        # Persist to DB
        await self._save_operation(
            context,
            str(profile.root_path),
            "deploy",
            result.overall_success,
            result.to_dict(),
            correlation_id=result.correlation_id,
        )

        # Update user_data for status display
        import time as _time

        context.user_data["last_deploy"] = {
            "success": result.overall_success,
            "failed_stage": result.failed_stage.value if result.failed_stage else None,
            "rollback": result.rollback_performed,
            "commit": result.post_deploy_commit,
            "timestamp": _time.time(),
        }

        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=user_id,
                command="workspace_deploy",
                args=[str(profile.root_path)],
                success=result.overall_success,
            )

    # ------------------------------------------------------------------
    # Self-heal: safe automated recovery for known scenarios
    # ------------------------------------------------------------------

    async def try_self_heal(
        self,
        query: Any,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> bool:
        """Attempt safe self-heal for the current workspace.

        Only acts if the profile has self_heal_restart enabled and the
        primary service is down. Returns True if healing was attempted.
        """
        _current_dir, _current_workspace, boundary_root, project_automation, profile = (
            self._get_workspace_profile(context)
        )
        if not profile:
            return False

        ops = getattr(profile, "operations", None)
        if not ops or not getattr(ops, "self_heal_restart", False):
            return False

        # Check if primary service is down
        primary_service = self.verify.select_primary_service(profile)
        if not primary_service:
            return False

        restart_cmd = getattr(primary_service, "restart_command", None)
        if not restart_cmd:
            return False

        health_cmd = getattr(primary_service, "health_command", None) or getattr(
            primary_service, "status_command", None
        )
        if not health_cmd:
            return False

        # Check current state
        health_result = await self.shell.execute(
            workspace_root=profile.root_path,
            command=health_cmd,
            timeout_seconds=10,
        )
        if health_result.success:
            return False  # Service is up, no heal needed

        import time as _time
        import uuid as _uuid

        correlation_id = str(_uuid.uuid4())[:8]
        rel_path = self.panel.format_relative_path(profile.root_path, boundary_root)

        status_msg = await query.message.reply_text(
            "🔄 <b>Self-heal</b>\n\n"
            f"Сервис <code>{escape_html(primary_service.display_name)}</code> не активен.\n"
            "Перезапускаю...",
            parse_mode="HTML",
        )

        # Restart
        restart_result = await self.shell.execute(
            workspace_root=profile.root_path,
            command=restart_cmd,
            timeout_seconds=30,
        )

        if not restart_result.success:
            await status_msg.edit_text(
                "❌ <b>Self-heal: перезапуск не удался</b>\n\n"
                f"Сервис: {escape_html(primary_service.display_name)}\n"
                f"Ошибка: <code>{escape_html(restart_result.stderr_text or restart_result.error or '?')}</code>",
                parse_mode="HTML",
            )
            await self._save_operation(
                context,
                str(profile.root_path),
                "self_heal",
                False,
                {"action": "restart", "error": restart_result.error},
                correlation_id=correlation_id,
            )
            return True

        # Verify after restart if configured
        verify_ok = None
        if getattr(ops, "self_heal_verify_after_restart", False):
            import asyncio

            await asyncio.sleep(3)
            health_result = await self.shell.execute(
                workspace_root=profile.root_path,
                command=health_cmd,
                timeout_seconds=15,
            )
            verify_ok = health_result.success

        if verify_ok is False:
            await status_msg.edit_text(
                "⚠️ <b>Self-heal: перезапущен, но проверка не прошла</b>\n\n"
                f"Сервис: {escape_html(primary_service.display_name)}\n"
                "Требуется ручной разбор.",
                parse_mode="HTML",
            )
        else:
            await status_msg.edit_text(
                "✅ <b>Self-heal: сервис восстановлен</b>\n\n"
                f"Сервис: {escape_html(primary_service.display_name)}\n"
                f"Проект: <code>{escape_html(rel_path)}</code>",
                parse_mode="HTML",
            )

        await self._save_operation(
            context,
            str(profile.root_path),
            "self_heal",
            verify_ok is not False,
            {
                "action": "restart",
                "verify_ok": verify_ok,
                "timestamp": _time.time(),
            },
            correlation_id=correlation_id,
        )
        return True
