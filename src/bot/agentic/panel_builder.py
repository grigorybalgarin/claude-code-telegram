"""Panel text and keyboard construction for agentic Telegram UI.

Builds HTML text blocks and InlineKeyboardMarkup objects for the control
panel views. Pure presentation layer — no message sending, no state mutation.
Receives pre-resolved workspace context and returns structured output that
the orchestrator sends to Telegram.
"""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
)

from ..utils.html_format import escape_html
from .context import AgenticWorkspaceContext
from .service_controller import ServiceController
from .shell_executor import ShellExecutor
from .verify_pipeline import VerifyPipeline


class PanelBuilder:
    """Constructs all agentic-mode panel views and keyboards."""

    def __init__(
        self,
        verify: VerifyPipeline,
        shell: ShellExecutor,
        services: ServiceController,
    ):
        self.verify = verify
        self.shell = shell
        self.services = services

    # ------------------------------------------------------------------
    # Keyboard builders
    # ------------------------------------------------------------------

    def build_start_keyboard(self) -> InlineKeyboardMarkup:
        """Return the default control buttons shown in agentic mode."""
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "\U0001f4c2 \u0421\u0442\u0430\u0442\u0443\u0441",
                        callback_data="act:status",
                    ),
                    InlineKeyboardButton(
                        "\u2705 \u041f\u0440\u043e\u0432\u0435\u0440\u0438\u0442\u044c",
                        callback_data="act:verify",
                    ),
                    InlineKeyboardButton(
                        "\U0001f6e0 \u0420\u0430\u0437\u0431\u0435\u0440\u0438\u0441\u044c",
                        callback_data="act:resolve",
                    ),
                ],
            ]
        )

    def build_control_panel_markup(self) -> InlineKeyboardMarkup:
        """Build the persistent control panel keyboard for agentic mode."""
        rows = [
            [
                InlineKeyboardButton(
                    "\U0001f4c2 \u0421\u0442\u0430\u0442\u0443\u0441",
                    callback_data="act:status",
                ),
                InlineKeyboardButton(
                    "\u2705 \u041f\u0440\u043e\u0432\u0435\u0440\u0438\u0442\u044c",
                    callback_data="act:verify",
                ),
                InlineKeyboardButton(
                    "\U0001f6e0 \u0420\u0430\u0437\u0431\u0435\u0440\u0438\u0441\u044c",
                    callback_data="act:resolve",
                ),
            ]
        ]
        return InlineKeyboardMarkup(rows)

    def build_reply_keyboard(self) -> ReplyKeyboardMarkup:
        """Return a persistent bottom keyboard for the most common actions."""
        rows: List[List[str]] = [
            [
                "\U0001f4c2 \u0421\u0442\u0430\u0442\u0443\u0441",
                "\u2705 \u041f\u0440\u043e\u0432\u0435\u0440\u0438\u0442\u044c",
                "\U0001f6e0 \u0420\u0430\u0437\u0431\u0435\u0440\u0438\u0441\u044c",
            ],
        ]
        return ReplyKeyboardMarkup(
            rows,
            resize_keyboard=True,
            is_persistent=True,
            input_field_placeholder="\u041d\u0430\u043f\u0438\u0448\u0438 \u0437\u0430\u0434\u0430\u0447\u0443 \u0438\u043b\u0438 \u043d\u0430\u0436\u043c\u0438 \u043a\u043d\u043e\u043f\u043a\u0443 \u043d\u0438\u0436\u0435",
        )

    @staticmethod
    def map_reply_action(message_text: str) -> Optional[str]:
        """Map reply-keyboard button text to an internal action."""
        normalized = message_text.strip()
        return {
            "\U0001f4c2 \u0421\u0442\u0430\u0442\u0443\u0441": "status",
            "\u2705 \u041f\u0440\u043e\u0432\u0435\u0440\u0438\u0442\u044c": "verify",
            "\U0001f6e0 \u0420\u0430\u0437\u0431\u0435\u0440\u0438\u0441\u044c": "resolve",
            "\U0001f4c2 Status": "status",
            "\u2705 Verify": "verify",
            "Resolve": "resolve",
        }.get(normalized)

    # ------------------------------------------------------------------
    # Path formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_relative_path(path: Path, boundary_root: Path) -> str:
        """Format a workspace-relative path for Telegram output."""
        try:
            relative = path.relative_to(boundary_root)
            return "/" if str(relative) == "." else relative.as_posix()
        except ValueError:
            return str(path)

    # ------------------------------------------------------------------
    # Job formatters
    # ------------------------------------------------------------------

    def format_job_status(self, job: Any, boundary_root: Path) -> str:
        """Build a compact one-line status for a persisted operator job."""
        workspace = self.format_relative_path(job.workspace_root, boundary_root)
        status = f"{job.status} {job.action_key}"
        verification_label = self.format_job_verification(job)
        if verification_label:
            status += f" \u00b7 {verification_label}"
        return f"{workspace} \u00b7 {status} \u00b7 {job.job_id[:8]}"

    @staticmethod
    def format_job_verification(job: Any) -> Optional[str]:
        """Return a compact verification label for background jobs."""
        if not getattr(job, "verification_command", None):
            return None

        status = getattr(job, "verification_status", None) or "pending"
        label = {
            "pending": "\u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 \u043e\u0436\u0438\u0434\u0430\u0435\u0442\u0441\u044f",
            "running": "\u0438\u0434\u0435\u0442 \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0430",
            "passed": "\u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 \u043f\u0440\u043e\u0439\u0434\u0435\u043d\u0430",
            "failed": "\u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 \u043d\u0435 \u043f\u0440\u043e\u0439\u0434\u0435\u043d\u0430",
        }.get(status, f"\u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 {status}")

        attempts = getattr(job, "verification_attempts", 0) or 0
        if attempts and status in {"running", "passed", "failed"}:
            label += f" ({attempts}x)"
        return label

    # ------------------------------------------------------------------
    # Text builders
    # ------------------------------------------------------------------

    async def build_status_text(
        self,
        ctx: AgenticWorkspaceContext,
        user_id: int,
        session_id: Optional[str],
        active_task_elapsed: Optional[int] = None,
        queue_size: int = 0,
        last_verify: Optional[Dict[str, Any]] = None,
        last_resolve: Optional[Dict[str, Any]] = None,
        last_deploy: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a short human-readable status summary."""
        session_status = "активна" if session_id else "нет"

        if not session_id and ctx.claude_integration:
            existing = await ctx.claude_integration._find_resumable_session(
                user_id, ctx.current_workspace
            )
            if existing:
                session_status = f"можно восстановить {existing.session_id[:8]}..."

        task_parts = []
        if active_task_elapsed is not None:
            task_parts.append(f"задача выполняется {active_task_elapsed}с")
            if queue_size:
                task_parts.append(f"в очереди {queue_size}")

        profile = ctx.profile
        project_label = (
            profile.display_name
            if profile and profile.display_name
            else self.format_relative_path(ctx.current_workspace, ctx.boundary_root)
        )
        project_path = self.format_relative_path(
            ctx.current_workspace, ctx.boundary_root
        )
        verify_steps = self.verify.build_steps(profile) if profile else []
        verify_status = "доступна" if verify_steps else "не настроена"
        primary_service = self.verify.select_primary_service(profile)
        active_incident = await self._get_active_incident(ctx)
        pending_improvements = await self._get_pending_improvements(ctx)
        system_summary = await self._get_system_summary(ctx)
        snapshot, diagnosis, _plan = self._build_operational_snapshot(
            ctx,
            last_verify=last_verify,
            last_resolve=last_resolve,
            last_deploy=last_deploy,
            active_incident=active_incident,
        )

        lines = [
            "<b>Статус</b>",
            "",
            f"\U0001f4e6 Активный проект: <code>{escape_html(project_label)}</code>",
            f"\U0001f4c2 Путь: <code>{escape_html(project_path)}</code>",
            f"\U0001f916 Сессия: <code>{escape_html(session_status)}</code>",
            f"\u2705 Полная проверка: <code>{escape_html(verify_status)}</code>",
        ]
        if primary_service:
            lines.append(
                f"\U0001f9e9 Основной сервис: <code>{escape_html(primary_service.display_name)}</code>"
            )
        if task_parts:
            task_joined = escape_html(" · ".join(task_parts))
            lines.append(f"\u2699\ufe0f Выполнение: <code>{task_joined}</code>")
        if ctx.operator_runtime:
            latest_job = ctx.operator_runtime.get_latest_job(ctx.current_workspace)
            if latest_job:
                lines.append(
                    f"\U0001f9f5 Задача: <code>{escape_html(self.format_job_status(latest_job, ctx.boundary_root))}</code>"
                )
                if latest_job.status == "stale":
                    lines.append(
                        "⚠️ <b>Внимание:</b> последняя фоновая задача зависла после рестарта или завершилась вне контроля бота."
                    )

        # Last verify/resolve/deploy results
        if last_verify or last_resolve or last_deploy:
            lines.append("")
        if last_verify:
            import time as _time

            ago = int(_time.time() - last_verify.get("timestamp", 0))
            ago_text = self._format_ago(ago)
            if last_verify.get("success"):
                v_label = f"✅ всё ок ({last_verify['steps_passed']}/{last_verify['steps_total']})"
            else:
                failed = escape_html(last_verify.get("failed_step", "?"))
                v_label = f"❌ сбой на '{failed}' ({last_verify['steps_passed']}/{last_verify['steps_total']})"
            lines.append(f"Проверка: <code>{v_label}</code> · {ago_text}")
        if last_resolve:
            import time as _time

            ago = int(_time.time() - last_resolve.get("timestamp", 0))
            ago_text = self._format_ago(ago)
            if last_resolve.get("success"):
                r_label = "✅ исправлено"
            elif last_resolve.get("rollback"):
                r_label = "⚠️ откат"
            elif last_resolve.get("error"):
                r_label = "❌ ошибка"
            else:
                r_label = "⚠️ не добил"
            attempts = last_resolve.get("attempts", 1)
            if attempts > 1:
                r_label += f" ({attempts}x)"
            lines.append(f"Разбор: <code>{r_label}</code> · {ago_text}")
        if last_deploy:
            import time as _time

            ago = int(_time.time() - last_deploy.get("timestamp", 0))
            ago_text = self._format_ago(ago)
            if last_deploy.get("success"):
                commit = last_deploy.get("commit", "")
                d_label = "✅ успешно"
                if commit:
                    d_label += f" ({commit[:8]})"
            else:
                failed_stage = escape_html(last_deploy.get("failed_stage", "?"))
                d_label = f"❌ сбой на '{failed_stage}'"
                if last_deploy.get("rollback"):
                    d_label += " · откат"
            lines.append(f"Деплой: <code>{d_label}</code> · {ago_text}")

        if active_incident:
            lines.extend(
                [
                    "",
                    "<b>Активный инцидент</b>",
                    self._format_incident_line(active_incident),
                ]
            )

        if snapshot and (snapshot.unresolved_issues or diagnosis):
            lines.extend(
                [
                    "",
                    f"{snapshot.health_emoji()} <b>Оперативная оценка:</b> "
                    f"{escape_html(self._format_health_label(snapshot.overall_health))}",
                ]
            )
            if diagnosis and diagnosis.short_cause:
                lines.append(f"• {escape_html(diagnosis.short_cause)}")
            for issue in snapshot.unresolved_issues[:2]:
                lines.append(f"• {escape_html(issue)}")

        if pending_improvements:
            count = len(pending_improvements)
            top = pending_improvements[0]
            description = escape_html(str(top.get("description") or "есть предложения"))
            lines.extend(
                [
                    "",
                    f"🔧 <b>Backlog улучшений:</b> <code>{count}</code>",
                    f"• {description}",
                ]
            )

        if system_summary:
            lines.extend(["", "<b>Системный контур</b>", *system_summary])

        # Suggested next action based on operational state
        suggested = None
        if snapshot and snapshot.suggested_action:
            reason = getattr(snapshot.suggested_action, "reason", "")
            if reason:
                suggested = reason
        if not suggested and active_incident:
            suggested = "Есть активный инцидент — открой разбор"
        if not suggested and ctx.operator_runtime:
            latest_job = ctx.operator_runtime.get_latest_job(ctx.current_workspace)
            if latest_job and latest_job.status == "stale":
                suggested = "Проверь зависшую задачу и при необходимости запусти проверку заново"
        if suggested:
            lines.append("")
            lines.append(f"💡 <b>Рекомендация:</b> {suggested}")

        if (
            ctx.project_automation
            and profile
            and ctx.current_workspace != ctx.boundary_root
        ):
            lines.extend(
                [
                    "",
                    "Можешь просто написать задачу обычным текстом. Бот выполнит ее в этом проекте.",
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def _suggest_from_ops_state(
        last_verify: Optional[Dict[str, Any]],
        last_resolve: Optional[Dict[str, Any]],
        last_deploy: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """Derive a suggested next action from last operation results."""
        # Failed verify → suggest resolve
        if last_verify and not last_verify.get("success"):
            failed = last_verify.get("failed_step", "")
            return f"Запусти разбор — сбой на '{failed}'"
        # Failed resolve → suggest manual check
        if last_resolve and not last_resolve.get("success"):
            if last_resolve.get("rollback"):
                return "Был откат — проверь вручную"
            return "Разбор не помог — нужен ручной анализ"
        # Failed deploy → suggest verify
        if last_deploy and not last_deploy.get("success"):
            return "Деплой не удался — запусти проверку"
        return None

    @staticmethod
    def _format_ago(seconds: int) -> str:
        """Format seconds ago as a human-readable Russian string."""
        if seconds < 60:
            return "только что"
        if seconds < 3600:
            mins = seconds // 60
            return f"{mins} мин назад"
        hours = seconds // 3600
        return f"{hours} ч назад"

    async def build_panel_text(
        self,
        ctx: AgenticWorkspaceContext,
        user_id: int,
        session_id: Optional[str],
        verbose_level: int = 1,
    ) -> str:
        """Build the main control panel text for agentic mode."""
        profile = ctx.profile
        lines = [
            "<b>\u041f\u0430\u043d\u0435\u043b\u044c \u0443\u043f\u0440\u0430\u0432\u043b\u0435\u043d\u0438\u044f</b>",
            "",
        ]
        if profile:
            lines.append(
                f"\U0001f4e6 \u041f\u0440\u043e\u0435\u043a\u0442: <code>{escape_html(self.format_relative_path(ctx.current_workspace, ctx.boundary_root))}</code>"
            )
            lines.append(
                f"\U0001f9f1 \u0421\u0442\u0435\u043a: <code>{escape_html(', '.join(profile.stacks))}</code>"
            )
        lines.append(
            "\U0001f6e1\ufe0f \u0410\u0432\u0442\u043e\u043f\u0438\u043b\u043e\u0442: <code>\u0432\u043a\u043b\u044e\u0447\u0435\u043d</code>"
        )
        verbose_label = {
            0: "\u043a\u043e\u0440\u043e\u0442\u043a\u043e",
            1: "\u043d\u043e\u0440\u043c\u0430\u043b\u044c\u043d\u043e",
            2: "\u043f\u043e\u0434\u0440\u043e\u0431\u043d\u043e",
        }[verbose_level]
        lines.append(
            f"\U0001f50a \u0420\u0435\u0436\u0438\u043c \u043e\u0442\u0432\u0435\u0442\u0430: <code>{escape_html(verbose_label)}</code>"
        )

        session_text = "\u043d\u0435\u0442"
        if session_id:
            session_text = (
                f"\u0430\u043a\u0442\u0438\u0432\u043d\u0430 {session_id[:8]}..."
            )
        elif ctx.claude_integration:
            existing = await ctx.claude_integration._find_resumable_session(
                user_id, ctx.current_workspace
            )
            if existing:
                session_text = f"\u043c\u043e\u0436\u043d\u043e \u0432\u043e\u0441\u0441\u0442\u0430\u043d\u043e\u0432\u0438\u0442\u044c {existing.session_id[:8]}..."
        lines.append(
            f"\U0001f916 \u0421\u0435\u0441\u0441\u0438\u044f: <code>{escape_html(session_text)}</code>"
        )

        if profile and ctx.project_automation:
            playbooks = ", ".join(
                playbook.slug
                for playbook in ctx.project_automation.list_playbooks(profile)
            )
            playbooks_display = escape_html(playbooks or "\u043d\u0435\u0442")
            lines.append(
                f"\U0001f9ed \u0421\u0446\u0435\u043d\u0430\u0440\u0438\u0438: <code>{playbooks_display}</code>"
            )
            operator_commands = ", ".join(
                key
                for key, _command in ctx.project_automation.list_operator_commands(
                    profile
                )
            )
            if operator_commands:
                lines.append(
                    f"\U0001f9f0 \u041e\u043f\u0435\u0440\u0430\u0446\u0438\u0438: <code>{escape_html(operator_commands)}</code>"
                )
            if profile.services:
                service_names = ", ".join(
                    service.display_name for service in profile.services
                )
                lines.append(
                    f"\U0001f9e9 \u0421\u0435\u0440\u0432\u0438\u0441\u044b: <code>{escape_html(service_names)}</code>"
                )
            if profile.operator_notes:
                note_preview = profile.operator_notes[:160]
                if len(profile.operator_notes) > 160:
                    note_preview += "..."
                lines.extend(
                    [
                        "",
                        f"\U0001f4dd \u0417\u0430\u043c\u0435\u0442\u043a\u0438: {escape_html(note_preview)}",
                    ]
                )
        if ctx.operator_runtime:
            latest_job = ctx.operator_runtime.get_latest_job(ctx.current_workspace)
            if latest_job:
                lines.extend(
                    [
                        "",
                        f"\U0001f9f5 \u041f\u043e\u0441\u043b\u0435\u0434\u043d\u044f\u044f \u0437\u0430\u0434\u0430\u0447\u0430: <code>{escape_html(self.format_job_status(latest_job, ctx.boundary_root))}</code>",
                    ]
                )

        lines.extend(
            [
                "",
                "\u0418\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0439 \u043a\u043d\u043e\u043f\u043a\u0438 \u043d\u0438\u0436\u0435, \u0447\u0442\u043e\u0431\u044b \u043f\u043e\u0441\u043c\u043e\u0442\u0440\u0435\u0442\u044c \u043f\u0440\u043e\u0435\u043a\u0442, \u043f\u0435\u0440\u0435\u043a\u043b\u044e\u0447\u0438\u0442\u044c workspace, \u0437\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u044c \u0441\u0446\u0435\u043d\u0430\u0440\u0438\u0439 \u0438\u043b\u0438 \u0444\u043e\u043d\u043e\u0432\u0443\u044e \u043e\u043f\u0435\u0440\u0430\u0446\u0438\u044e.",
            ]
        )
        return "\n".join(lines)

    async def build_recent_text(
        self,
        ctx: AgenticWorkspaceContext,
        user_id: int,
    ) -> str:
        """Build a compact recent activity view for the control panel."""
        if not ctx.storage:
            return "\u0425\u0440\u0430\u043d\u0438\u043b\u0438\u0449\u0435 \u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u043d\u043e."

        audit_entries = await ctx.storage.audit.get_user_audit_log(user_id, limit=10)
        messages = await ctx.storage.messages.get_user_messages(user_id, limit=4)
        command_entries = [
            entry for entry in audit_entries if entry.event_type == "command"
        ][:4]
        automation_entries = [
            entry for entry in audit_entries if entry.event_type == "automation_run"
        ][:4]
        workspace_state = await self._get_workspace_state(ctx)
        active_incident = await self._get_active_incident(ctx)
        snapshot, diagnosis, _plan = self._build_operational_snapshot(
            ctx,
            last_verify=workspace_state.get("verify"),
            last_resolve=workspace_state.get("resolve"),
            last_deploy=workspace_state.get("deploy"),
            active_incident=active_incident,
        )

        lines = [
            "<b>\u041d\u0435\u0434\u0430\u0432\u043d\u044f\u044f \u0430\u043a\u0442\u0438\u0432\u043d\u043e\u0441\u0442\u044c</b>"
        ]
        if snapshot:
            lines.extend(["", "<b>Сейчас</b>"])
            status_parts: List[str] = [snapshot.health_emoji()]
            if snapshot.last_verify_success is True:
                status_parts.append("verify ✓")
            elif snapshot.last_verify_success is False:
                status_parts.append("verify ✗")
            if snapshot.last_deploy_success is True:
                status_parts.append("deploy ✓")
            elif snapshot.last_deploy_success is False:
                status_parts.append("deploy ✗")
            if snapshot.active_incident:
                status_parts.append("есть инцидент")
            lines.append(" ".join(status_parts))
            if diagnosis and diagnosis.short_cause:
                lines.append(f"• {escape_html(diagnosis.short_cause)}")
            if snapshot.suggested_action and getattr(
                snapshot.suggested_action, "reason", ""
            ):
                lines.append(f"→ {escape_html(snapshot.suggested_action.reason)}")

        if automation_entries:
            lines.extend(
                ["", "<b>\u0410\u0432\u0442\u043e\u043f\u0438\u043b\u043e\u0442</b>"]
            )
            for entry in automation_entries:
                details = (entry.event_data or {}).get("details", {})
                playbook = escape_html(str(details.get("playbook", "general")))
                workspace = escape_html(
                    self.format_relative_path(
                        Path(str(details.get("workspace_root", ctx.current_directory))),
                        ctx.boundary_root,
                    )
                )
                result = "\u2705" if entry.success else "\u26a0\ufe0f"
                lines.append(
                    f"{result} <code>{playbook}</code> \u00b7 <code>{workspace}</code>"
                )

        if command_entries:
            lines.extend(["", "<b>\u041a\u043e\u043c\u0430\u043d\u0434\u044b</b>"])
            for entry in command_entries:
                details = (entry.event_data or {}).get("details", {})
                command_name = escape_html(str(details.get("command", "command")))
                lines.append(f"\u2022 <code>{command_name}</code>")

        if messages:
            lines.extend(["", "<b>\u0417\u0430\u043f\u0440\u043e\u0441\u044b</b>"])
            for message in messages:
                preview = escape_html(" ".join(message.prompt.split())[:72])
                lines.append(f"\u2022 {preview}")

        workspace_ops = await self._get_workspace_ops(ctx)
        if workspace_ops:
            lines.extend(
                [
                    "",
                    "<b>\u041e\u043f\u0435\u0440\u0430\u0446\u0438\u0438 \u043f\u0440\u043e\u0435\u043a\u0442\u0430</b>",
                ]
            )
            for op in workspace_ops:
                lines.append(self._format_operation_line(op))

        incidents = await self._get_active_incidents(ctx)
        if incidents:
            lines.extend(
                ["", "<b>\u0418\u043d\u0446\u0438\u0434\u0435\u043d\u0442\u044b</b>"]
            )
            for incident in incidents[:3]:
                lines.append(self._format_incident_line(incident))

        pending_improvements = await self._get_pending_improvements(ctx)
        if pending_improvements:
            lines.extend(
                ["", "<b>\u0423\u043b\u0443\u0447\u0448\u0435\u043d\u0438\u044f</b>"]
            )
            for item in pending_improvements[:3]:
                description = escape_html(str(item.get("description") or ""))
                lines.append(f"\u2022 {description}")

        system_summary = await self._get_system_summary(ctx)
        if system_summary:
            lines.extend(
                [
                    "",
                    "<b>\u0421\u0438\u0441\u0442\u0435\u043c\u0430</b>",
                    *system_summary,
                ]
            )

        if len(lines) == 1:
            lines.extend(
                [
                    "",
                    "\u041f\u043e\u043a\u0430 \u043d\u0435\u0434\u0430\u0432\u043d\u0435\u0439 \u0430\u043a\u0442\u0438\u0432\u043d\u043e\u0441\u0442\u0438 \u043d\u0435\u0442.",
                ]
            )

        lines.append("")
        lines.append(
            f"\u0422\u0435\u043a\u0443\u0449\u0438\u0439 \u043f\u0440\u043e\u0435\u043a\u0442: <code>{escape_html(self.format_relative_path(ctx.current_workspace, ctx.boundary_root))}</code>"
        )
        return "\n".join(lines)

    async def _get_workspace_ops(
        self, ctx: AgenticWorkspaceContext
    ) -> List[Dict[str, Any]]:
        if not ctx.storage or not hasattr(ctx.storage, "operations"):
            return []
        try:
            return await ctx.storage.operations.get_recent(
                str(ctx.current_workspace), limit=4
            )
        except Exception:
            return []

    async def _get_workspace_state(
        self, ctx: AgenticWorkspaceContext
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        if not ctx.storage or not hasattr(ctx.storage, "operations"):
            return {}
        try:
            raw_state = await ctx.storage.operations.get_workspace_state(
                str(ctx.current_workspace)
            )
        except Exception:
            return {}

        return {
            key: self._normalize_operation_state(row) for key, row in raw_state.items()
        }

    async def _get_active_incidents(
        self, ctx: AgenticWorkspaceContext
    ) -> List[Dict[str, Any]]:
        if not ctx.storage or not hasattr(ctx.storage, "incidents"):
            return []
        try:
            return await ctx.storage.incidents.list_active(
                [str(ctx.current_workspace)], limit=3
            )
        except Exception:
            return []

    async def _get_active_incident(
        self, ctx: AgenticWorkspaceContext
    ) -> Optional[Dict[str, Any]]:
        incidents = await self._get_active_incidents(ctx)
        return incidents[0] if incidents else None

    async def _get_pending_improvements(
        self, ctx: AgenticWorkspaceContext
    ) -> List[Dict[str, Any]]:
        if not ctx.storage or not hasattr(ctx.storage, "improvements"):
            return []
        try:
            return await ctx.storage.improvements.list_pending(limit=5)
        except Exception:
            return []

    async def _get_system_summary(self, ctx: AgenticWorkspaceContext) -> List[str]:
        if not ctx.storage or not hasattr(ctx.storage, "operations"):
            return []
        try:
            ops = await ctx.storage.operations.get_recent("__system__", limit=6)
        except Exception:
            return []

        lines: List[str] = []
        self_review = next(
            (op for op in ops if op.get("operation_type") == "self_review"),
            None,
        )
        cleanup = next(
            (op for op in ops if op.get("operation_type") == "maintenance_cleanup"),
            None,
        )

        if self_review:
            details = (
                self_review.get("details", {}) if isinstance(self_review, dict) else {}
            )
            candidates = (
                details.get("candidates", 0) if isinstance(details, dict) else 0
            )
            ago = self._format_ago(self._seconds_ago(self_review.get("created_at")))
            lines.append(
                f"🧠 self-review: <code>{candidates}</code> кандидатов · {ago}"
            )
        if cleanup:
            details = cleanup.get("details", {}) if isinstance(cleanup, dict) else {}
            if isinstance(details, dict):
                cleaned = sum(
                    int(details.get(key, 0))
                    for key in (
                        "sessions_cleaned",
                        "operations_cleaned",
                        "incidents_cleaned",
                        "improvements_cleaned",
                    )
                )
            else:
                cleaned = 0
            ago = self._format_ago(self._seconds_ago(cleanup.get("created_at")))
            lines.append(f"🧹 cleanup: <code>{cleaned}</code> записей · {ago}")
        return lines

    def _format_incident_line(self, incident: Dict[str, Any]) -> str:
        severity = str(incident.get("severity") or "warning")
        state = str(incident.get("state") or "detected")
        details = incident.get("details", {})
        if not isinstance(details, dict):
            details = {}
        cause = (
            details.get("short_cause")
            or details.get("last_error")
            or "требует внимания"
        )
        severity_emoji = {
            "critical": "🔴",
            "degraded": "🟠",
            "warning": "⚠️",
            "info": "ℹ️",
        }.get(severity, "⚠️")
        return (
            f"{severity_emoji} <code>{escape_html(state)}</code> · "
            f"{escape_html(str(cause)[:120])}"
        )

    def _format_operation_line(self, operation: Dict[str, Any]) -> str:
        op_type = str(operation.get("operation_type") or "operation")
        success = bool(operation.get("success"))
        details = operation.get("details", {})
        if not isinstance(details, dict):
            details = {}
        icon = "✅" if success else "⚠️"
        label = {
            "verify": "проверка",
            "resolve": "разбор",
            "deploy": "деплой",
            "operator_job_stale": "stale job",
            "incident_detected": "инцидент",
            "incident_healed": "восстановление",
            "incident_escalated": "эскалация",
            "heal_attempted": "автовосстановление",
            "heal_failed": "ошибка починки",
        }.get(op_type, op_type)
        summary = (
            details.get("failed_step")
            or details.get("failed_stage")
            or details.get("problem_type")
            or details.get("action_key")
            or ""
        )
        ago = self._format_ago(self._seconds_ago(operation.get("created_at")))
        if summary:
            return f"{icon} <code>{escape_html(label)}</code> · {escape_html(str(summary))} · {ago}"
        return f"{icon} <code>{escape_html(label)}</code> · {ago}"

    def _build_operational_snapshot(
        self,
        ctx: AgenticWorkspaceContext,
        *,
        last_verify: Optional[Dict[str, Any]],
        last_resolve: Optional[Dict[str, Any]],
        last_deploy: Optional[Dict[str, Any]],
        active_incident: Optional[Dict[str, Any]],
    ) -> tuple[Optional[Any], Optional[Any], Optional[Any]]:
        """Build a presentation-friendly operational snapshot and recommendation."""
        from .digest import build_operational_snapshot, suggest_next_action
        from .remediation_planner import build_remediation_plan

        workspace_state = {
            "verify": last_verify,
            "resolve": last_resolve,
            "deploy": last_deploy,
        }
        incident_obj = self._coerce_incident_for_snapshot(active_incident)
        profile = ctx.profile
        display_name = (
            profile.display_name
            if profile and profile.display_name
            else self.format_relative_path(ctx.current_workspace, ctx.boundary_root)
        )
        snapshot = build_operational_snapshot(
            workspace_path=str(ctx.current_workspace),
            display_name=display_name,
            workspace_state=workspace_state,
            active_incident=incident_obj,
            service_healthy=None,
        )

        diagnosis = self._build_diagnosis_from_state(
            ctx,
            last_verify=last_verify,
            last_resolve=last_resolve,
            last_deploy=last_deploy,
            active_incident=active_incident,
        )

        plan = None
        if diagnosis and profile:
            ops_config = getattr(profile, "operations", None)
            runbook_hints = (
                getattr(ops_config, "runbook_hints", None) if ops_config else None
            )
            plan = build_remediation_plan(
                diagnosis,
                runbook_hints,
            )
            snapshot.suggested_action = suggest_next_action(
                snapshot,
                diagnosis=diagnosis,
                remediation_plan=plan,
            )
        return snapshot, diagnosis, plan

    def _build_diagnosis_from_state(
        self,
        ctx: AgenticWorkspaceContext,
        *,
        last_verify: Optional[Dict[str, Any]],
        last_resolve: Optional[Dict[str, Any]],
        last_deploy: Optional[Dict[str, Any]],
        active_incident: Optional[Dict[str, Any]],
    ) -> Optional[Any]:
        """Reconstruct a lightweight diagnosis from persisted operation details."""
        from .problem_classifier import ProblemDiagnosis, ProblemType

        source: Optional[Dict[str, Any]] = None
        fallback_type = ProblemType.UNKNOWN
        failed_label = ""

        if last_verify and not last_verify.get("success"):
            source = last_verify
            failed_label = str(source.get("failed_step") or "")
        elif last_resolve and not last_resolve.get("success"):
            source = last_resolve
            failed_label = str(
                source.get("failed_step") or source.get("error") or "resolve"
            )
        elif last_deploy and not last_deploy.get("success"):
            source = last_deploy
            fallback_type = ProblemType.DEPLOY
            failed_label = str(source.get("failed_stage") or "deploy")
        elif active_incident:
            details = active_incident.get("details", {})
            if isinstance(details, dict):
                source = details
                failed_label = str(
                    details.get("failed_step")
                    or details.get("failed_stage")
                    or active_incident.get("state")
                    or "incident"
                )

        if not source:
            return None

        ptype = self._parse_problem_type(source.get("problem_type"), fallback_type)
        label = self._problem_label(ptype)
        short_cause = str(
            source.get("short_cause")
            or source.get("last_error")
            or source.get("error")
            or ""
        )
        confidence_raw = source.get("confidence")
        try:
            confidence = float(confidence_raw) if confidence_raw is not None else 0.6
        except (TypeError, ValueError):
            confidence = 0.6

        profile = ctx.profile
        ops = getattr(profile, "operations", None) if profile else None
        critical_steps = set(getattr(ops, "critical_steps", ()) or ())
        runbook_hints = getattr(ops, "runbook_hints", {}) or {}
        runbook_hint = str(
            runbook_hints.get(ptype.value) or runbook_hints.get(failed_label) or ""
        )

        return ProblemDiagnosis(
            problem_type=ptype,
            label=label,
            failed_step_label=failed_label,
            short_cause=short_cause,
            safe_to_autofix=ptype
            in {
                ProblemType.CODE,
                ProblemType.CONFIG,
                ProblemType.DEPENDENCY,
                ProblemType.DEPLOY,
            },
            confidence=confidence,
            is_critical_step=failed_label in critical_steps,
            runbook_hint=runbook_hint,
        )

    @staticmethod
    def _coerce_incident_for_snapshot(
        active_incident: Optional[Dict[str, Any]],
    ) -> Optional[Any]:
        if not active_incident:
            return None

        from .ops_model import IncidentState, Severity

        state_raw = str(active_incident.get("state") or "").lower()
        severity_raw = str(active_incident.get("severity") or "").lower()

        try:
            state = IncidentState(state_raw)
        except ValueError:
            state = None
        try:
            severity = Severity(severity_raw)
        except ValueError:
            severity = None

        return SimpleNamespace(state=state, severity=severity)

    @staticmethod
    def _parse_problem_type(value: Any, fallback: Any) -> Any:
        from .problem_classifier import ProblemType

        if value:
            try:
                return ProblemType(str(value))
            except ValueError:
                pass
        return fallback

    @staticmethod
    def _problem_label(problem_type: Any) -> str:
        labels = {
            "code": "Ошибка в коде",
            "config": "Проблема конфигурации",
            "dependency": "Проблема зависимостей",
            "service": "Проблема сервиса",
            "deploy": "Проблема сборки/деплоя",
            "environment": "Проблема окружения сервера",
            "unknown": "Неизвестная проблема",
        }
        key = getattr(problem_type, "value", "unknown")
        return labels.get(str(key), "Неизвестная проблема")

    @staticmethod
    def _normalize_operation_state(
        operation: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not operation or not isinstance(operation, dict):
            return None

        details = operation.get("details")
        if isinstance(details, dict):
            normalized = dict(details)
            if "success" not in normalized and "success" in operation:
                normalized["success"] = operation.get("success")
            if (
                "timestamp" not in normalized
                and operation.get("created_at") is not None
            ):
                normalized["timestamp"] = PanelBuilder._timestamp_from_value(
                    operation.get("created_at")
                )
            return normalized

        return operation

    @staticmethod
    def _format_health_label(severity: Any) -> str:
        value = getattr(severity, "value", "")
        return {
            "critical": "критичная проблема",
            "degraded": "есть деградация",
            "warning": "есть предупреждения",
            "info": "явных проблем не видно",
        }.get(str(value), "состояние неизвестно")

    @staticmethod
    def _seconds_ago(timestamp: Any) -> int:
        import time as _time

        ts = PanelBuilder._timestamp_from_value(timestamp)
        if ts <= 0:
            return 0
        return max(0, int(_time.time() - ts))

    @staticmethod
    def _timestamp_from_value(timestamp: Any) -> float:
        if isinstance(timestamp, datetime):
            return timestamp.timestamp()
        if isinstance(timestamp, (int, float)):
            return float(timestamp)
        if isinstance(timestamp, str):
            normalized = timestamp.strip().replace("Z", "+00:00")
            try:
                return datetime.fromisoformat(normalized).timestamp()
            except ValueError:
                return 0.0
        return 0.0

    async def build_jobs_text(
        self,
        ctx: AgenticWorkspaceContext,
        header: Optional[str] = None,
    ) -> tuple[str, InlineKeyboardMarkup]:
        """Render recent background workspace jobs and management buttons."""
        profile = ctx.profile
        if not ctx.operator_runtime:
            return (
                "\u0424\u043e\u043d\u043e\u0432\u044b\u0435 \u0437\u0430\u0434\u0430\u0447\u0438 \u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u043d\u044b.",
                self.build_start_keyboard(),
            )

        jobs = ctx.operator_runtime.list_jobs(limit=8)
        current_jobs = ctx.operator_runtime.list_jobs(
            workspace_root=ctx.current_workspace, limit=4
        )

        lines = [
            "<b>\u0424\u043e\u043d\u043e\u0432\u044b\u0435 \u0437\u0430\u0434\u0430\u0447\u0438</b>"
        ]
        if header:
            lines.extend(["", header])

        if current_jobs:
            lines.extend(
                [
                    "",
                    "<b>\u0422\u0435\u043a\u0443\u0449\u0438\u0439 \u043f\u0440\u043e\u0435\u043a\u0442</b>",
                ]
            )
            for job in current_jobs:
                lines.append(
                    f"\u2022 <code>{escape_html(self.format_job_status(job, ctx.boundary_root))}</code>"
                )
        if jobs:
            other_jobs = [
                job for job in jobs if job.workspace_root != ctx.current_workspace
            ][:4]
            if other_jobs:
                lines.extend(
                    ["", "<b>\u041d\u0435\u0434\u0430\u0432\u043d\u0438\u0435</b>"]
                )
                for job in other_jobs:
                    lines.append(
                        f"\u2022 <code>{escape_html(self.format_job_status(job, ctx.boundary_root))}</code>"
                    )
        if len(lines) == 1:
            lines.extend(
                [
                    "",
                    "\u0424\u043e\u043d\u043e\u0432\u044b\u0445 \u0437\u0430\u0434\u0430\u0447 \u043f\u043e\u043a\u0430 \u043d\u0435\u0442.",
                ]
            )

        latest_job = current_jobs[0] if current_jobs else None
        if latest_job:
            lines.extend(
                [
                    "",
                    "<b>\u041f\u043e\u0441\u043b\u0435\u0434\u043d\u044f\u044f \u0437\u0430\u0434\u0430\u0447\u0430</b>",
                    f"\u0414\u0435\u0439\u0441\u0442\u0432\u0438\u0435: <code>{escape_html(latest_job.action_key)}</code>",
                    f"\u0421\u0442\u0430\u0442\u0443\u0441: <code>{escape_html(latest_job.status)}</code>",
                ]
            )
            if latest_job.exit_code is not None:
                lines.append(
                    f"\u041a\u043e\u0434 \u0432\u044b\u0445\u043e\u0434\u0430: <code>{latest_job.exit_code}</code>"
                )
            if latest_job.verification_command:
                verify_status = self.format_job_verification(latest_job)
                if verify_status:
                    lines.append(
                        f"\u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430: <code>{escape_html(verify_status)}</code>"
                    )
                if latest_job.verification_exit_code is not None:
                    lines.append(
                        f"\u041a\u043e\u0434 \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0438: <code>{latest_job.verification_exit_code}</code>"
                    )
                if latest_job.verification_error:
                    lines.append(
                        f"\u041e\u0448\u0438\u0431\u043a\u0430 \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0438: <code>{escape_html(latest_job.verification_error)}</code>"
                    )
            log_tail = ctx.operator_runtime.read_log_tail(latest_job, limit=500)
            if log_tail:
                lines.extend(
                    [
                        "",
                        "<b>\u041f\u043e\u0441\u043b\u0435\u0434\u043d\u0438\u0435 \u043b\u043e\u0433\u0438</b>",
                        f"<pre>{escape_html(log_tail)}</pre>",
                    ]
                )

        keyboard_rows: List[list] = []
        if latest_job and latest_job.is_active:
            keyboard_rows.append(
                [
                    InlineKeyboardButton(
                        f"\U0001f6d1 \u041e\u0441\u0442\u0430\u043d\u043e\u0432\u0438\u0442\u044c {latest_job.action_key}",
                        callback_data=f"act:stop:{latest_job.job_id}",
                    )
                ]
            )

        if profile:
            action_row = []
            for key, label in (
                ("start", "\u25b6\ufe0f \u0417\u0430\u043f\u0443\u0441\u043a"),
                (
                    "dev",
                    "\U0001f6e0\ufe0f \u0420\u0430\u0437\u0440\u0430\u0431\u043e\u0442\u043a\u0430",
                ),
                ("deploy", "\U0001f680 \u0414\u0435\u043f\u043b\u043e\u0439"),
            ):
                if key in profile.commands:
                    action_row.append(
                        InlineKeyboardButton(label, callback_data=f"act:{key}")
                    )
            if action_row:
                keyboard_rows.append(action_row[:3])

        keyboard_rows.append(
            [
                InlineKeyboardButton(
                    "\U0001f504 \u041e\u0431\u043d\u043e\u0432\u0438\u0442\u044c",
                    callback_data="act:jobs",
                ),
                InlineKeyboardButton(
                    "\U0001f39b\ufe0f \u041f\u0430\u043d\u0435\u043b\u044c",
                    callback_data="act:panel",
                ),
                InlineKeyboardButton(
                    "\U0001f4c2 \u0421\u0442\u0430\u0442\u0443\u0441",
                    callback_data="act:status",
                ),
            ]
        )
        keyboard_rows.append(
            [
                InlineKeyboardButton(
                    "\U0001f4c1 \u041f\u0440\u043e\u0435\u043a\u0442\u044b",
                    callback_data="act:projects",
                )
            ]
        )
        return "\n".join(lines), InlineKeyboardMarkup(keyboard_rows)

    async def build_services_text(
        self,
        ctx: AgenticWorkspaceContext,
        header: Optional[str] = None,
    ) -> tuple[str, InlineKeyboardMarkup]:
        """Render managed service definitions for the current workspace."""
        profile = ctx.profile
        if not profile or not profile.services:
            return (
                "\u0414\u043b\u044f \u044d\u0442\u043e\u0433\u043e \u043f\u0440\u043e\u0435\u043a\u0442\u0430 \u0443\u043f\u0440\u0430\u0432\u043b\u044f\u0435\u043c\u044b\u0435 \u0441\u0435\u0440\u0432\u0438\u0441\u044b \u043d\u0435 \u043d\u0430\u0441\u0442\u0440\u043e\u0435\u043d\u044b.",
                self.build_control_panel_markup(),
            )

        lines = [
            "<b>\u0423\u043f\u0440\u0430\u0432\u043b\u044f\u0435\u043c\u044b\u0435 \u0441\u0435\u0440\u0432\u0438\u0441\u044b</b>",
            "",
            f"\u041f\u0440\u043e\u0435\u043a\u0442: <code>{escape_html(self.format_relative_path(ctx.current_workspace, ctx.boundary_root))}</code>",
            "\u0414\u0435\u0439\u0441\u0442\u0432\u0438\u044f \u0437\u0430\u043f\u0443\u0441\u043a\u0430 \u0438 \u0440\u0435\u0441\u0442\u0430\u0440\u0442\u0430 \u0430\u0432\u0442\u043e\u043c\u0430\u0442\u0438\u0447\u0435\u0441\u043a\u0438 \u0432\u044b\u043f\u043e\u043b\u043d\u044f\u044e\u0442 \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0443 \u0438 \u043f\u0440\u0438\u043a\u043b\u0430\u0434\u044b\u0432\u0430\u044e\u0442 \u043b\u043e\u0433\u0438 \u043f\u0440\u0438 \u043e\u0448\u0438\u0431\u043a\u0435.",
        ]
        if header:
            lines.extend(["", header])

        for service in profile.services:
            actions = ", ".join(service.available_actions) or "none"
            lines.extend(
                [
                    "",
                    f"\u2022 <b>{escape_html(service.display_name)}</b> \u00b7 <code>{escape_html(service.service_type)}</code>",
                    f"  \u0434\u0435\u0439\u0441\u0442\u0432\u0438\u044f: <code>{escape_html(actions)}</code>",
                ]
            )

        keyboard_rows: List[list] = []
        for service in profile.services:
            inspect_row = []
            lifecycle_row = []
            for action_key in ("status", "health", "logs"):
                if service.command_for(action_key):
                    inspect_row.append(
                        InlineKeyboardButton(
                            self.services.format_action_label(service, action_key),
                            callback_data=f"act:svc:{service.key}:{action_key}",
                        )
                    )
            for action_key in ("restart", "start", "stop"):
                if service.command_for(action_key):
                    lifecycle_row.append(
                        InlineKeyboardButton(
                            self.services.format_action_label(service, action_key),
                            callback_data=f"act:svc:{service.key}:{action_key}",
                        )
                    )
            if inspect_row:
                keyboard_rows.append(inspect_row[:3])
            if lifecycle_row:
                keyboard_rows.append(lifecycle_row[:3])

        keyboard_rows.append(
            [
                InlineKeyboardButton(
                    "\U0001f504 \u041e\u0431\u043d\u043e\u0432\u0438\u0442\u044c",
                    callback_data="act:services",
                ),
                InlineKeyboardButton(
                    "\U0001f39b\ufe0f \u041f\u0430\u043d\u0435\u043b\u044c",
                    callback_data="act:panel",
                ),
                InlineKeyboardButton(
                    "\U0001f4c2 \u0421\u0442\u0430\u0442\u0443\u0441",
                    callback_data="act:status",
                ),
            ]
        )
        keyboard_rows.append(
            [
                InlineKeyboardButton(
                    "\U0001f4c1 \u041f\u0440\u043e\u0435\u043a\u0442\u044b",
                    callback_data="act:projects",
                )
            ]
        )
        return "\n".join(lines), InlineKeyboardMarkup(keyboard_rows)

    async def build_running_services_text(
        self,
        ctx: AgenticWorkspaceContext,
        header: Optional[str] = None,
    ) -> tuple[str, InlineKeyboardMarkup]:
        """Render a live view of managed and system-level running services."""
        profile = ctx.profile
        lines = [
            "<b>\u0417\u0430\u043f\u0443\u0449\u0435\u043d\u043d\u044b\u0435 \u0441\u0435\u0440\u0432\u0438\u0441\u044b</b>",
            "",
            f"\u041f\u0440\u043e\u0435\u043a\u0442: <code>{escape_html(self.format_relative_path(ctx.current_workspace, ctx.boundary_root))}</code>",
        ]
        if header:
            lines.extend(["", header])

        if profile and profile.services:
            lines.extend(
                [
                    "",
                    "<b>\u0423\u043f\u0440\u0430\u0432\u043b\u044f\u0435\u043c\u044b\u0435 \u0441\u0435\u0440\u0432\u0438\u0441\u044b</b>",
                ]
            )
            for service in profile.services:
                command = service.health_command or service.status_command
                if not command:
                    lines.append(
                        f"\u2022 <code>{escape_html(service.display_name)}</code>: <code>\u043d\u0435\u0442 live-\u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0438</code>"
                    )
                    continue
                result = await self.shell.execute(
                    workspace_root=profile.root_path,
                    command=command,
                    timeout_seconds=45,
                )
                state = (
                    "ok" if result.success else "\u043e\u0448\u0438\u0431\u043a\u0430"
                )
                if result.timed_out:
                    state = "\u0442\u0430\u0439\u043c\u0430\u0443\u0442"
                lines.append(
                    f"\u2022 <code>{escape_html(service.display_name)}</code>: <code>{escape_html(state)}</code>"
                )
                summary = self.shell.summarize(result)
                if (
                    summary
                    and summary
                    != "\u043d\u0435\u0442 \u0432\u044b\u0432\u043e\u0434\u0430"
                ):
                    lines.append(f"  <code>{escape_html(summary)}</code>")
        else:
            lines.extend(
                [
                    "",
                    "\u0414\u043b\u044f \u044d\u0442\u043e\u0433\u043e \u043f\u0440\u043e\u0435\u043a\u0442\u0430 \u0443\u043f\u0440\u0430\u0432\u043b\u044f\u0435\u043c\u044b\u0435 \u0441\u0435\u0440\u0432\u0438\u0441\u044b \u043d\u0435 \u043d\u0430\u0441\u0442\u0440\u043e\u0435\u043d\u044b.",
                ]
            )

        running_units_result = await self.services.list_running_units(
            ctx.current_workspace
        )
        running_units = self.services.parse_systemd_units(running_units_result)
        lines.extend(
            [
                "",
                "<b>\u0421\u0438\u0441\u0442\u0435\u043c\u043d\u044b\u0435 \u0441\u0435\u0440\u0432\u0438\u0441\u044b \u0441\u0435\u0440\u0432\u0435\u0440\u0430</b>",
            ]
        )
        if running_units:
            for unit in running_units:
                lines.append(f"\u2022 <code>{escape_html(unit)}</code>")
        else:
            summary = self.shell.summarize(running_units_result)
            label = "\u0441\u043f\u0438\u0441\u043e\u043a systemd \u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u0435\u043d"
            if running_units_result.success:
                label = "\u0437\u0430\u043f\u0443\u0449\u0435\u043d\u043d\u044b\u0435 \u0441\u0435\u0440\u0432\u0438\u0441\u044b \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d\u044b"
            lines.append(f"<code>{escape_html(label)}</code>")
            if summary and summary not in {
                "\u043d\u0435\u0442 \u0432\u044b\u0432\u043e\u0434\u0430",
                label,
            }:
                lines.append(f"<code>{escape_html(summary)}</code>")

        failed_units_result = await self.services.list_failed_units(
            ctx.current_workspace
        )
        failed_units = self.services.parse_systemd_units(failed_units_result, limit=6)
        if failed_units:
            lines.extend(
                [
                    "",
                    "<b>\u0421\u0435\u0440\u0432\u0438\u0441\u044b \u0441 \u043e\u0448\u0438\u0431\u043a\u0430\u043c\u0438</b>",
                ]
            )
            for unit in failed_units:
                lines.append(f"\u2022 <code>{escape_html(unit)}</code>")

        keyboard_rows = [
            [
                InlineKeyboardButton(
                    "\U0001f504 \u041e\u0431\u043d\u043e\u0432\u0438\u0442\u044c",
                    callback_data="act:running",
                ),
                InlineKeyboardButton(
                    "\U0001f39b\ufe0f \u041f\u0430\u043d\u0435\u043b\u044c",
                    callback_data="act:panel",
                ),
                InlineKeyboardButton(
                    "\U0001f4c2 \u0421\u0442\u0430\u0442\u0443\u0441",
                    callback_data="act:status",
                ),
            ],
            [
                InlineKeyboardButton(
                    "\U0001f9e9 \u0421\u0435\u0440\u0432\u0438\u0441\u044b",
                    callback_data="act:services",
                ),
                InlineKeyboardButton(
                    "\U0001f4c1 \u041f\u0440\u043e\u0435\u043a\u0442\u044b",
                    callback_data="act:projects",
                ),
            ],
        ]
        return "\n".join(lines), InlineKeyboardMarkup(keyboard_rows)

    async def build_workspace_catalog(
        self,
        ctx: AgenticWorkspaceContext,
    ) -> tuple[str, InlineKeyboardMarkup]:
        """Build the workspace catalog view used by /repo and control buttons."""
        if ctx.project_automation:
            summaries = ctx.project_automation.list_workspace_summaries(
                ctx.boundary_root
            )
            if summaries:
                lines: List[str] = [
                    "<b>\u041f\u0440\u043e\u0435\u043a\u0442\u044b</b>",
                    "",
                ]
                for summary in summaries:
                    lines.extend(
                        ctx.project_automation.describe_workspace_summary_lines(
                            summary, current_workspace=ctx.current_workspace
                        )
                    )
                lines.extend(
                    [
                        "",
                        "\u0410\u0432\u0442\u043e\u043f\u0438\u043b\u043e\u0442 \u0443\u043c\u0435\u0435\u0442 \u0432\u044b\u0431\u0438\u0440\u0430\u0442\u044c \u043f\u0440\u043e\u0435\u043a\u0442 \u043f\u043e \u0438\u043c\u0435\u043d\u0438, \u0430\u043b\u0438\u0430\u0441\u0443 \u0438\u043b\u0438 \u043e\u0442\u043d\u043e\u0441\u0438\u0442\u0435\u043b\u044c\u043d\u043e\u043c\u0443 \u043f\u0443\u0442\u0438.",
                    ]
                )

                keyboard_rows: List[list] = []
                for i in range(0, len(summaries), 2):
                    row = []
                    for j in range(2):
                        if i + j < len(summaries):
                            summary = summaries[i + j]
                            row.append(
                                InlineKeyboardButton(
                                    summary.button_label,
                                    callback_data=f"cd:{summary.relative_path}",
                                )
                            )
                    keyboard_rows.append(row)

                keyboard_rows.append(
                    [
                        InlineKeyboardButton(
                            "\U0001f504 \u041e\u0431\u043d\u043e\u0432\u0438\u0442\u044c",
                            callback_data="act:projects",
                        ),
                        InlineKeyboardButton(
                            "\U0001f39b\ufe0f \u041f\u0430\u043d\u0435\u043b\u044c",
                            callback_data="act:panel",
                        ),
                        InlineKeyboardButton(
                            "\U0001f4c2 \u0421\u0442\u0430\u0442\u0443\u0441",
                            callback_data="act:status",
                        ),
                    ]
                )
                keyboard_rows.append(
                    [
                        InlineKeyboardButton(
                            "\U0001f558 \u041d\u0435\u0434\u0430\u0432\u043d\u0435\u0435",
                            callback_data="act:recent",
                        )
                    ]
                )
                return "\n".join(lines), InlineKeyboardMarkup(keyboard_rows)

        lines = ["<b>\u041f\u0440\u043e\u0435\u043a\u0442\u044b</b>", ""]
        try:
            entries = sorted(
                [
                    d
                    for d in ctx.boundary_root.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ],
                key=lambda d: d.name.casefold(),
            )
        except OSError as e:
            return (
                f"\u041e\u0448\u0438\u0431\u043a\u0430 \u0447\u0442\u0435\u043d\u0438\u044f \u043f\u0440\u043e\u0435\u043a\u0442\u0430: {escape_html(str(e))}",
                self.build_start_keyboard(),
            )

        kb_rows: List[list] = []
        for entry in entries:
            marker = " \u25c0" if entry == ctx.current_workspace else ""
            lines.append(f"\u2022 <code>{escape_html(entry.name)}</code>{marker}")
        for i in range(0, len(entries), 2):
            row = []
            for j in range(2):
                if i + j < len(entries):
                    row.append(
                        InlineKeyboardButton(
                            entries[i + j].name,
                            callback_data=f"cd:{entries[i + j].name}",
                        )
                    )
            kb_rows.append(row)
        kb_rows.append(
            [
                InlineKeyboardButton(
                    "\U0001f504 \u041e\u0431\u043d\u043e\u0432\u0438\u0442\u044c",
                    callback_data="act:projects",
                ),
                InlineKeyboardButton(
                    "\U0001f39b\ufe0f \u041f\u0430\u043d\u0435\u043b\u044c",
                    callback_data="act:panel",
                ),
            ]
        )
        return "\n".join(lines), InlineKeyboardMarkup(kb_rows)
