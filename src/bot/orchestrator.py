"""Message orchestrator — single entry point for all Telegram updates.

Routes messages based on agentic vs classic mode. In agentic mode, provides
an autopilot-first conversational interface with inline control buttons. In
classic mode, delegates to existing full-featured handlers.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import structlog
from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ..claude.sdk_integration import StreamUpdate
from ..config.settings import Settings
from ..projects import PrivateTopicsUnavailableError
from .features.change_guard import ChangeGuardReport
from ..utils.redaction import redact_sensitive_text
from .utils.draft_streamer import DraftStreamer, generate_draft_id
from .utils.html_format import escape_html
from .utils.image_extractor import (
    ImageAttachment,
    should_send_as_photo,
    validate_image_path,
)

logger = structlog.get_logger()


def _redact_secrets(text: str) -> str:
    """Replace likely secrets/credentials with redacted placeholders."""
    return redact_sensitive_text(text)


# Tool name -> friendly emoji mapping for verbose output
_TOOL_ICONS: Dict[str, str] = {
    "Read": "\U0001f4d6",
    "Write": "\u270f\ufe0f",
    "Edit": "\u270f\ufe0f",
    "MultiEdit": "\u270f\ufe0f",
    "Bash": "\U0001f4bb",
    "Glob": "\U0001f50d",
    "Grep": "\U0001f50d",
    "LS": "\U0001f4c2",
    "Task": "\U0001f9e0",
    "TaskOutput": "\U0001f9e0",
    "WebFetch": "\U0001f310",
    "WebSearch": "\U0001f310",
    "NotebookRead": "\U0001f4d3",
    "NotebookEdit": "\U0001f4d3",
    "TodoRead": "\u2611\ufe0f",
    "TodoWrite": "\u2611\ufe0f",
}


def _tool_icon(name: str) -> str:
    """Return emoji for a tool, with a default wrench."""
    return _TOOL_ICONS.get(name, "\U0001f527")


@dataclass
class _ActiveTask:
    """Tracks a running Claude task for a user."""

    task: asyncio.Task  # type: ignore[type-arg]
    original_prompt: str
    started_at: float
    cancel_requested: bool = False


# Patterns that indicate user wants to stop/cancel
_STOP_PATTERNS = re.compile(
    r"^(стоп|stop|cancel|отмена|хватит|стой|enough)$",
    re.IGNORECASE,
)


@dataclass
class _PendingMessage:
    """A queued message waiting for the active task to finish."""

    update: Update
    context: Any


@dataclass(frozen=True)
class _MessageActionProxy:
    """Adapter that lets reply-keyboard presses reuse quick-action handlers."""

    update: Update

    @property
    def from_user(self) -> Any:
        """Expose the user like telegram CallbackQuery does."""
        return self.update.effective_user

    @property
    def message(self) -> Any:
        """Expose the original message to shared action runners."""
        return self.update.message

    async def answer(
        self,
        text: Optional[str] = None,
        show_alert: bool = False,
    ) -> None:
        """Fallback answer implementation for non-callback interactions."""
        del show_alert
        if text:
            await self.update.message.reply_text(text)

    async def edit_message_text(
        self,
        text: str,
        parse_mode: Optional[str] = None,
        reply_markup: Optional[Any] = None,
    ) -> None:
        """Send a fresh message when there is no callback message to edit."""
        await self.update.message.reply_text(
            text,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
        )


@dataclass(frozen=True)
class _ShellActionResult:
    """Structured result of a deterministic shell action."""

    command: str
    returncode: int
    success: bool
    timed_out: bool
    stdout_text: str
    stderr_text: str
    error: Optional[str] = None


@dataclass(frozen=True)
class _VerifyStep:
    """One deterministic verification step for the current workspace."""

    label: str
    command: str
    logs_command: Optional[str] = None


class MessageOrchestrator:
    """Routes messages based on mode. Single entry point for all Telegram updates."""

    def __init__(self, settings: Settings, deps: Dict[str, Any]):
        self.settings = settings
        self.deps = deps
        # Track active Claude tasks per user for interrupt/cancel
        self._active_tasks: Dict[int, _ActiveTask] = {}
        # Queue of messages received while a task is running
        self._pending_messages: Dict[int, List[_PendingMessage]] = {}
        # Per-user lock to protect _active_tasks and _pending_messages
        self._user_locks: Dict[int, asyncio.Lock] = {}

    def _inject_deps(self, handler: Callable) -> Callable:  # type: ignore[type-arg]
        """Wrap handler to inject dependencies into context.bot_data."""

        async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            for key, value in self.deps.items():
                context.bot_data[key] = value
            context.bot_data["settings"] = self.settings
            context.user_data.pop("_thread_context", None)

            is_sync_bypass = handler.__name__ == "sync_threads"
            is_start_bypass = handler.__name__ in {"start_command", "agentic_start"}
            message_thread_id = self._extract_message_thread_id(update)
            should_enforce = self.settings.enable_project_threads

            if should_enforce:
                if self.settings.project_threads_mode == "private":
                    should_enforce = not is_sync_bypass and not (
                        is_start_bypass and message_thread_id is None
                    )
                else:
                    should_enforce = not is_sync_bypass

            if should_enforce:
                allowed = await self._apply_thread_routing_context(update, context)
                if not allowed:
                    return

            try:
                await handler(update, context)
            finally:
                if should_enforce:
                    self._persist_thread_state(context)

        return wrapped

    async def _apply_thread_routing_context(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """Enforce strict project-thread routing and load thread-local state."""
        manager = context.bot_data.get("project_threads_manager")
        if manager is None:
            await self._reject_for_thread_mode(
                update,
                "❌ <b>Project Thread Mode Misconfigured</b>\n\n"
                "Thread manager is not initialized.",
            )
            return False

        chat = update.effective_chat
        message = update.effective_message
        if not chat or not message:
            return False

        if self.settings.project_threads_mode == "group":
            if chat.id != self.settings.project_threads_chat_id:
                await self._reject_for_thread_mode(
                    update,
                    manager.guidance_message(mode=self.settings.project_threads_mode),
                )
                return False
        else:
            if getattr(chat, "type", "") != "private":
                await self._reject_for_thread_mode(
                    update,
                    manager.guidance_message(mode=self.settings.project_threads_mode),
                )
                return False

        message_thread_id = self._extract_message_thread_id(update)
        if not message_thread_id:
            await self._reject_for_thread_mode(
                update,
                manager.guidance_message(mode=self.settings.project_threads_mode),
            )
            return False

        project = await manager.resolve_project(chat.id, message_thread_id)
        if not project:
            await self._reject_for_thread_mode(
                update,
                manager.guidance_message(mode=self.settings.project_threads_mode),
            )
            return False

        state_key = f"{chat.id}:{message_thread_id}"
        thread_states = context.user_data.setdefault("thread_state", {})
        state = thread_states.get(state_key, {})

        project_root = project.absolute_path
        current_dir_raw = state.get("current_directory")
        current_dir = (
            Path(current_dir_raw).resolve() if current_dir_raw else project_root
        )
        if not self._is_within(current_dir, project_root) or not current_dir.is_dir():
            current_dir = project_root

        context.user_data["current_directory"] = current_dir
        context.user_data["claude_session_id"] = state.get("claude_session_id")
        context.user_data["_thread_context"] = {
            "chat_id": chat.id,
            "message_thread_id": message_thread_id,
            "state_key": state_key,
            "project_slug": project.slug,
            "project_root": str(project_root),
            "project_name": project.name,
        }
        return True

    def _persist_thread_state(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Persist compatibility keys back into per-thread state."""
        thread_context = context.user_data.get("_thread_context")
        if not thread_context:
            return

        project_root = Path(thread_context["project_root"])
        current_dir = context.user_data.get("current_directory", project_root)
        if not isinstance(current_dir, Path):
            current_dir = Path(str(current_dir))
        current_dir = current_dir.resolve()
        if not self._is_within(current_dir, project_root) or not current_dir.is_dir():
            current_dir = project_root

        thread_states = context.user_data.setdefault("thread_state", {})
        thread_states[thread_context["state_key"]] = {
            "current_directory": str(current_dir),
            "claude_session_id": context.user_data.get("claude_session_id"),
            "project_slug": thread_context["project_slug"],
        }

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        """Return True if path is within root."""
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _extract_message_thread_id(update: Update) -> Optional[int]:
        """Extract topic/thread id from update message for forum/direct topics."""
        message = update.effective_message
        if not message:
            return None
        message_thread_id = getattr(message, "message_thread_id", None)
        if isinstance(message_thread_id, int) and message_thread_id > 0:
            return message_thread_id
        dm_topic = getattr(message, "direct_messages_topic", None)
        topic_id = getattr(dm_topic, "topic_id", None) if dm_topic else None
        if isinstance(topic_id, int) and topic_id > 0:
            return topic_id
        # Telegram omits message_thread_id for the General topic in forum
        # supergroups; its canonical thread ID is 1.
        chat = update.effective_chat
        if chat and getattr(chat, "is_forum", False):
            return 1
        return None

    async def _reject_for_thread_mode(self, update: Update, message: str) -> None:
        """Send a guidance response when strict thread routing rejects an update."""
        query = update.callback_query
        if query:
            try:
                await query.answer()
            except Exception:
                pass
            if query.message:
                await query.message.reply_text(message, parse_mode="HTML")
            return

        if update.effective_message:
            await update.effective_message.reply_text(message, parse_mode="HTML")

    def _get_boundary_root(self, context: ContextTypes.DEFAULT_TYPE) -> Path:
        """Resolve the approved root for the current thread or chat."""
        if self.settings.enable_project_threads:
            thread_context = context.user_data.get("_thread_context")
            if thread_context:
                return Path(thread_context["project_root"]).resolve()
        return self.settings.approved_directory

    def register_handlers(self, app: Application) -> None:
        """Register handlers based on mode."""
        if self.settings.agentic_mode:
            self._register_agentic_handlers(app)
        else:
            self._register_classic_handlers(app)

    def _register_agentic_handlers(self, app: Application) -> None:
        """Register agentic handlers: commands + text/file/photo."""
        from .handlers import command

        # Commands
        handlers = [
            ("start", self.agentic_start),
            ("new", self.agentic_new),
            ("status", self.agentic_status),
            ("diag", command.diag_command),
            ("recent", command.recent_activity_command),
            ("playbooks", command.playbooks_command),
            ("run", command.run_playbook_command),
            ("verbose", self.agentic_verbose),
            ("stats", self.agentic_stats),
            ("template", self.agentic_template),
            ("context", self.agentic_context),
            ("repo", self.agentic_repo),
            ("restart", command.restart_command),
        ]
        if self.settings.enable_project_threads:
            handlers.append(("sync_threads", command.sync_threads))

        for cmd, handler in handlers:
            app.add_handler(CommandHandler(cmd, self._inject_deps(handler)))

        # Text messages -> Claude
        app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._inject_deps(self.agentic_text),
            ),
            group=10,
        )

        # File uploads -> Claude
        app.add_handler(
            MessageHandler(
                filters.Document.ALL, self._inject_deps(self.agentic_document)
            ),
            group=10,
        )

        # Photo uploads -> Claude
        app.add_handler(
            MessageHandler(filters.PHOTO, self._inject_deps(self.agentic_photo)),
            group=10,
        )

        # Voice messages -> transcribe -> Claude
        app.add_handler(
            MessageHandler(filters.VOICE, self._inject_deps(self.agentic_voice)),
            group=10,
        )

        # Video notes (circles) and videos -> extract frames -> Claude
        app.add_handler(
            MessageHandler(
                filters.VIDEO_NOTE | filters.VIDEO,
                self._inject_deps(self.agentic_video_note),
            ),
            group=10,
        )

        # Quick action buttons
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._agentic_quick_action),
                pattern=r"^act:",
            )
        )

        # Project selection callbacks
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._agentic_callback),
                pattern=r"^cd:",
            )
        )

        logger.info("Agentic handlers registered")

    def _register_classic_handlers(self, app: Application) -> None:
        """Register full classic handler set (moved from core.py)."""
        from .handlers import callback, command, message

        handlers = [
            ("start", command.start_command),
            ("help", command.help_command),
            ("new", command.new_session),
            ("continue", command.continue_session),
            ("end", command.end_session),
            ("ls", command.list_files),
            ("cd", command.change_directory),
            ("pwd", command.print_working_directory),
            ("projects", command.show_projects),
            ("status", command.session_status),
            ("diag", command.diag_command),
            ("recent", command.recent_activity_command),
            ("playbooks", command.playbooks_command),
            ("run", command.run_playbook_command),
            ("export", command.export_session),
            ("actions", command.quick_actions),
            ("git", command.git_command),
            ("restart", command.restart_command),
        ]
        if self.settings.enable_project_threads:
            handlers.append(("sync_threads", command.sync_threads))

        for cmd, handler in handlers:
            app.add_handler(CommandHandler(cmd, self._inject_deps(handler)))

        app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._inject_deps(message.handle_text_message),
            ),
            group=10,
        )
        app.add_handler(
            MessageHandler(
                filters.Document.ALL, self._inject_deps(message.handle_document)
            ),
            group=10,
        )
        app.add_handler(
            MessageHandler(filters.PHOTO, self._inject_deps(message.handle_photo)),
            group=10,
        )
        app.add_handler(
            MessageHandler(filters.VOICE, self._inject_deps(message.handle_voice)),
            group=10,
        )
        app.add_handler(
            CallbackQueryHandler(self._inject_deps(callback.handle_callback_query))
        )

        logger.info("Classic handlers registered (18 commands + full handler set)")

    async def get_bot_commands(self) -> list:  # type: ignore[type-arg]
        """Return bot commands appropriate for current mode."""
        if self.settings.agentic_mode:
            commands = [
                BotCommand("start", "Запустить бота"),
                BotCommand("new", "Начать новую сессию"),
                BotCommand("status", "Показать статус сессии"),
                BotCommand("diag", "Диагностика бота и проекта"),
                BotCommand("recent", "Недавние запросы и команды"),
                BotCommand("playbooks", "Список сценариев"),
                BotCommand("run", "Запустить сценарий проекта"),
                BotCommand("verbose", "Уровень подробности (0/1/2)"),
                BotCommand("stats", "Статистика и расходы"),
                BotCommand("template", "Управление шаблонами"),
                BotCommand("context", "Постоянные инструкции"),
                BotCommand("repo", "Список проектов / переключение"),
                BotCommand("restart", "Перезапустить бота"),
            ]
            if self.settings.enable_project_threads:
                commands.append(BotCommand("sync_threads", "Синхронизировать топики"))
            return commands
        else:
            commands = [
                BotCommand("start", "Start bot and show help"),
                BotCommand("help", "Show available commands"),
                BotCommand("new", "Clear context and start fresh session"),
                BotCommand("continue", "Explicitly continue last session"),
                BotCommand("end", "End current session and clear context"),
                BotCommand("ls", "List files in current directory"),
                BotCommand("cd", "Change directory (resumes project session)"),
                BotCommand("pwd", "Show current directory"),
                BotCommand("projects", "Show all projects"),
                BotCommand("status", "Show session status"),
                BotCommand("diag", "Show bot and workspace diagnostics"),
                BotCommand("recent", "Show recent prompts and commands"),
                BotCommand("playbooks", "List deterministic playbooks"),
                BotCommand("run", "Run a workspace playbook"),
                BotCommand("export", "Export current session"),
                BotCommand("actions", "Show quick actions"),
                BotCommand("git", "Git repository commands"),
                BotCommand("restart", "Restart the bot"),
            ]
            if self.settings.enable_project_threads:
                commands.append(BotCommand("sync_threads", "Sync project topics"))
            return commands

    # --- Agentic handlers ---

    async def agentic_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Brief welcome with a persistent bottom keyboard."""
        user = update.effective_user
        sync_line = ""
        if (
            self.settings.enable_project_threads
            and self.settings.project_threads_mode == "private"
        ):
            if (
                not update.effective_chat
                or getattr(update.effective_chat, "type", "") != "private"
            ):
                await update.message.reply_text(
                    "🚫 <b>Private Topics Mode</b>\n\n"
                    "Use this bot in a private chat and run <code>/start</code> there.",
                    parse_mode="HTML",
                )
                return
            manager = context.bot_data.get("project_threads_manager")
            if manager:
                try:
                    result = await manager.sync_topics(
                        context.bot,
                        chat_id=update.effective_chat.id,
                    )
                    sync_line = (
                        "\n\n🧵 Topics synced"
                        f" (created {result.created}, reused {result.reused})."
                    )
                except PrivateTopicsUnavailableError:
                    await update.message.reply_text(
                        manager.private_topics_unavailable_message(),
                        parse_mode="HTML",
                    )
                    return
                except Exception:
                    sync_line = "\n\n🧵 Topic sync failed. Run /sync_threads to retry."
        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        dir_display = f"<code>{current_dir}/</code>"

        safe_name = escape_html(user.first_name)
        keyboard = self._build_agentic_reply_keyboard(context)
        await update.message.reply_text(
            f"Привет, {safe_name}! Я твой AI-помощник по коду.\n"
            "Можешь просто написать задачу обычным текстом, команды не обязательны.\n"
            "Если нужно, упомяни проект по имени, и я сам переключу workspace.\n"
            "Автопилот включен: я определяю проект, делаю checkpoint для рискованных правок, "
            "проверяю результат и откатываю изменения, если проверка не прошла.\n"
            "Основные кнопки управления закреплены внизу.\n\n"
            f"Текущая папка: {dir_display}"
            f"{sync_line}",
            parse_mode="HTML",
            reply_markup=keyboard,
        )
        self._mark_agentic_reply_keyboard_ready(context)

    async def agentic_new(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Reset session, one-line confirmation."""
        context.user_data["claude_session_id"] = None
        context.user_data["session_started"] = True
        context.user_data["force_new_session"] = True

        await update.message.reply_text(
            "Сессия сброшена. Что делаем дальше?",
            reply_markup=self._build_agentic_reply_keyboard(context),
        )
        self._mark_agentic_reply_keyboard_ready(context)

    async def agentic_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Compact status with task/queue info."""
        user_id = update.effective_user.id
        text = await self._build_agentic_status_text(context, user_id)
        _current_dir, _current_workspace, _boundary_root, _project_automation, profile = (
            self._get_agentic_workspace_profile(context)
        )
        await update.message.reply_text(
            text,
            parse_mode="HTML",
            reply_markup=self._build_agentic_control_panel_markup(profile),
        )

    def _get_verbose_level(self, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Return effective verbose level: per-user override or global default."""
        user_override = context.user_data.get("verbose_level")
        if user_override is not None:
            return int(user_override)
        return self.settings.verbose_level

    async def agentic_verbose(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Set output verbosity: /verbose [0|1|2]."""
        args = update.message.text.split()[1:] if update.message.text else []
        if not args:
            current = self._get_verbose_level(context)
            labels = {0: "коротко", 1: "нормально", 2: "подробно"}
            await update.message.reply_text(
                f"Режим ответа: <b>{current}</b> ({labels.get(current, '?')})\n\n"
                "Использование: <code>/verbose 0|1|2</code>\n"
                "  0 = коротко (только финальный ответ)\n"
                "  1 = нормально (инструменты + ход мысли)\n"
                "  2 = подробно (инструменты с вводом + ход мысли)",
                parse_mode="HTML",
            )
            return

        try:
            level = int(args[0])
            if level not in (0, 1, 2):
                raise ValueError
        except ValueError:
            await update.message.reply_text(
                "Используй: /verbose 0, /verbose 1 или /verbose 2"
            )
            return

        context.user_data["verbose_level"] = level
        labels = {0: "коротко", 1: "нормально", 2: "подробно"}
        await update.message.reply_text(
            f"Режим ответа: <b>{level}</b> ({labels[level]})",
            parse_mode="HTML",
        )

    async def agentic_stats(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show usage analytics: costs, sessions, top tools."""
        user_id = update.effective_user.id
        storage = context.bot_data.get("storage")
        if not storage:
            await update.message.reply_text("Хранилище недоступно.")
            return

        try:
            stats = await storage.analytics.get_user_stats(user_id)
            summary = stats.get("summary", {})
            daily = stats.get("daily_usage", [])
            top_tools = stats.get("top_tools", [])

            total_cost = summary.get("total_cost") or 0.0
            total_msgs = summary.get("total_messages") or 0
            total_sessions = summary.get("total_sessions") or 0
            avg_duration = summary.get("avg_duration") or 0
            avg_duration_s = avg_duration / 1000 if avg_duration else 0

            # Cost by recent days
            cost_lines = []
            for day in daily[:7]:
                d = day.get("date", "?")
                c = day.get("cost") or 0.0
                m = day.get("messages") or 0
                bar = "#" * min(int(c * 20), 30) if c else ""
                cost_lines.append(f"  {d}  ${c:.3f}  ({m} msg) {bar}")

            cost_chart = "\n".join(cost_lines) if cost_lines else "  Пока данных нет"

            # Top tools
            tool_lines = []
            for t in top_tools[:8]:
                icon = _tool_icon(t["tool_name"])
                tool_lines.append(f"  {icon} {t['tool_name']}: {t['usage_count']}x")
            tools_text = "\n".join(tool_lines) if tool_lines else "  Инструменты пока не отслеживались"

            text = (
                f"<b>Твоя статистика</b>\n\n"
                f"<b>Итого:</b> ${total_cost:.4f} | "
                f"{total_msgs} сообщений | {total_sessions} сессий\n"
                f"<b>Средний ответ:</b> {avg_duration_s:.1f}с\n\n"
                f"<b>Последние 7 дней:</b>\n<pre>{cost_chart}</pre>\n\n"
                f"<b>Топ инструментов:</b>\n{tools_text}"
            )
            _current_dir, _current_workspace, _boundary_root, _project_automation, profile = (
                self._get_agentic_workspace_profile(context)
            )
            await update.message.reply_text(
                text,
                parse_mode="HTML",
                reply_markup=self._build_agentic_control_panel_markup(profile),
            )
        except Exception as e:
            logger.error("Stats command failed", error=str(e))
            await update.message.reply_text(f"Ошибка загрузки статистики: {e}")

    async def agentic_template(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Manage prompt templates: /template [list|add|del|run] ..."""
        args = update.message.text.split(maxsplit=2)[1:] if update.message.text else []
        templates = context.user_data.setdefault("templates", {})

        if not args or args[0] == "list":
            if not templates:
                await update.message.reply_text(
                    "<b>No templates saved.</b>\n\n"
                    "Usage:\n"
                    "<code>/template add name Your prompt here</code>\n"
                    "<code>/template run name</code>\n"
                    "<code>/template del name</code>",
                    parse_mode="HTML",
                )
                return
            lines = []
            for name, prompt in templates.items():
                short = prompt[:60] + "..." if len(prompt) > 60 else prompt
                lines.append(f"  <b>{name}</b> - {escape_html(short)}")
            await update.message.reply_text(
                "<b>Your templates:</b>\n" + "\n".join(lines) + "\n\n"
                "Use: <code>/template run name</code>",
                parse_mode="HTML",
            )
            return

        action = args[0].lower()
        rest = args[1] if len(args) > 1 else ""

        if action == "add":
            parts = rest.split(maxsplit=1)
            if len(parts) < 2:
                await update.message.reply_text(
                    "Usage: <code>/template add name Your prompt</code>",
                    parse_mode="HTML",
                )
                return
            name, prompt = parts[0], parts[1]
            templates[name] = prompt
            await update.message.reply_text(
                f"Template <b>{escape_html(name)}</b> saved.",
                parse_mode="HTML",
            )

        elif action == "del":
            name = rest.strip()
            if name in templates:
                del templates[name]
                await update.message.reply_text(f"Template '{name}' deleted.")
            else:
                await update.message.reply_text(f"Template '{name}' not found.")

        elif action == "run":
            name = rest.strip()
            prompt = templates.get(name)
            if not prompt:
                await update.message.reply_text(f"Template '{name}' not found.")
                return
            # Simulate text message with the template prompt
            update.message.text = prompt
            await self.agentic_text(update, context)

        else:
            await update.message.reply_text(
                "Unknown action. Use: list, add, del, run"
            )

    async def agentic_context(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Manage persistent session instructions: /context [add|list|clear] ..."""
        args = update.message.text.split(maxsplit=1)[1:] if update.message.text else []
        instructions = context.user_data.setdefault("custom_instructions", [])

        if not args or args[0].strip().lower() == "list":
            if not instructions:
                await update.message.reply_text(
                    "<b>No custom instructions set.</b>\n\n"
                    "Usage:\n"
                    "<code>/context add Always reply in Russian</code>\n"
                    "<code>/context clear</code>",
                    parse_mode="HTML",
                )
                return
            lines = [f"  {i+1}. {escape_html(inst)}" for i, inst in enumerate(instructions)]
            await update.message.reply_text(
                "<b>Active instructions:</b>\n" + "\n".join(lines),
                parse_mode="HTML",
            )
            return

        text = args[0]
        if text.lower().startswith("add "):
            instruction = text[4:].strip()
            if instruction:
                instructions.append(instruction)
                await update.message.reply_text(
                    f"Instruction added ({len(instructions)} total)."
                )
            else:
                await update.message.reply_text("Provide an instruction after 'add'.")

        elif text.lower().startswith("clear"):
            instructions.clear()
            await update.message.reply_text("All custom instructions cleared.")

        elif text.lower().startswith("del "):
            try:
                idx = int(text[4:].strip()) - 1
                if 0 <= idx < len(instructions):
                    removed = instructions.pop(idx)
                    await update.message.reply_text(f"Removed: {removed[:50]}")
                else:
                    await update.message.reply_text("Invalid index.")
            except ValueError:
                await update.message.reply_text("Usage: /context del <number>")

        else:
            # Shortcut: /context <text> = add
            instructions.append(text.strip())
            await update.message.reply_text(
                f"Instruction added ({len(instructions)} total)."
            )

    def _build_agentic_start_keyboard(self) -> InlineKeyboardMarkup:
        """Return the default control buttons shown in agentic mode."""
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("🎛️ Панель", callback_data="act:panel"),
                    InlineKeyboardButton("📁 Проекты", callback_data="act:projects"),
                    InlineKeyboardButton("🧵 Задачи", callback_data="act:jobs"),
                ],
                [
                    InlineKeyboardButton("🩺 Диагностика", callback_data="act:doctor"),
                    InlineKeyboardButton("🕘 Недавнее", callback_data="act:recent"),
                    InlineKeyboardButton("📂 Статус", callback_data="act:status"),
                ],
                [
                    InlineKeyboardButton("📡 Запущено", callback_data="act:running"),
                    InlineKeyboardButton("📊 Статистика", callback_data="act:stats"),
                    InlineKeyboardButton("🆕 Новая сессия", callback_data="act:new"),
                ],
                [
                    InlineKeyboardButton("🔇 Коротко", callback_data="act:v0"),
                    InlineKeyboardButton("🔉 Нормально", callback_data="act:v1"),
                    InlineKeyboardButton("🔊 Подробно", callback_data="act:v2"),
                ],
            ]
        )

    def _build_agentic_reply_keyboard(
        self, context: ContextTypes.DEFAULT_TYPE
    ) -> ReplyKeyboardMarkup:
        """Return a persistent bottom keyboard for the most common actions."""
        _current_dir, _current_workspace, _boundary_root, _project_automation, profile = (
            self._get_agentic_workspace_profile(context)
        )
        rows: List[List[str]] = [
            ["🎛️ Панель", "📂 Статус", "📡 Запущено"],
            ["📁 Проекты", "🧵 Задачи", "🕘 Недавнее"],
            ["✅ Проверить", "🩺 Диагностика", "🆕 Новая сессия"],
        ]

        primary_service = self._select_agentic_primary_service(profile)
        if primary_service:
            service_row: List[str] = []
            if primary_service.command_for("status"):
                service_row.append("📟 Сервис")
            if primary_service.command_for("logs"):
                service_row.append("📜 Логи")
            if primary_service.command_for("restart"):
                service_row.append("🔄 Рестарт")
            if service_row:
                rows.append(service_row)

        if profile:
            operator_row: List[str] = []
            if "start" in profile.commands:
                operator_row.append("▶️ Запуск")
            if "dev" in profile.commands:
                operator_row.append("🛠️ Разработка")
            if "deploy" in profile.commands:
                operator_row.append("🚀 Деплой")
            if operator_row:
                rows.append(operator_row)

        return ReplyKeyboardMarkup(
            rows,
            resize_keyboard=True,
            is_persistent=True,
            input_field_placeholder="Напиши задачу или нажми кнопку ниже",
        )

    @staticmethod
    def _mark_agentic_reply_keyboard_ready(context: ContextTypes.DEFAULT_TYPE) -> None:
        """Remember that the persistent reply keyboard has already been shown."""
        context.user_data["agentic_reply_keyboard_ready"] = True

    def _build_agentic_reply_action(
        self,
        message_text: str,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> Optional[str]:
        """Map reply-keyboard button text to an internal action."""
        normalized = message_text.strip()
        action = {
            "🎛️ Панель": "panel",
            "📂 Статус": "status",
            "📡 Запущено": "running",
            "📁 Проекты": "projects",
            "🧵 Задачи": "jobs",
            "🕘 Недавнее": "recent",
            "✅ Проверить": "verify",
            "🩺 Диагностика": "doctor",
            "🆕 Новая сессия": "new",
            "▶️ Запуск": "start",
            "🛠️ Разработка": "dev",
            "🚀 Деплой": "deploy",
            "🎛️ Panel": "panel",
            "📂 Status": "status",
            "📡 Running": "running",
            "📁 Projects": "projects",
            "🧵 Jobs": "jobs",
            "🕘 Recent": "recent",
            "✅ Verify": "verify",
            "🩺 Doctor": "doctor",
            "🆕 New": "new",
            "▶️ Start": "start",
            "🛠️ Dev": "dev",
            "🚀 Deploy": "deploy",
        }.get(normalized)
        if action:
            return action

        _current_dir, _current_workspace, _boundary_root, _project_automation, profile = (
            self._get_agentic_workspace_profile(context)
        )
        primary_service = self._select_agentic_primary_service(profile)
        if not primary_service:
            return None

        service_actions = {
            "📟 Сервис": "status",
            "📜 Логи": "logs",
            "🔄 Рестарт": "restart",
            "📟 Service": "status",
            "📜 Logs": "logs",
            "🔄 Restart": "restart",
        }
        service_action = service_actions.get(normalized)
        if service_action and primary_service.command_for(service_action):
            return f"svc:{primary_service.key}:{service_action}"
        return None

    def _get_agentic_workspace_profile(
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

        profile = project_automation.build_profile(current_dir, boundary_root)
        return current_dir, profile.root_path, boundary_root, project_automation, profile

    def _get_agentic_operator_runtime(
        self, context: ContextTypes.DEFAULT_TYPE
    ) -> Optional[Any]:
        """Resolve the persistent background operator runtime."""
        features = context.bot_data.get("features")
        return (
            getattr(features, "get_workspace_operator", lambda: None)()
            if features
            else None
        )

    def _build_agentic_context_markup(
        self, context: ContextTypes.DEFAULT_TYPE
    ) -> InlineKeyboardMarkup:
        """Build the current control panel markup for reply messages."""
        _current_dir, _current_workspace, _boundary_root, _project_automation, profile = (
            self._get_agentic_workspace_profile(context)
        )
        return self._build_agentic_control_panel_markup(profile)

    def _format_agentic_relative_path(self, path: Path, boundary_root: Path) -> str:
        """Format a workspace-relative path for Telegram output."""
        try:
            relative = path.relative_to(boundary_root)
            return "/" if str(relative) == "." else relative.as_posix()
        except ValueError:
            return str(path)

    def _build_agentic_control_panel_markup(
        self, profile: Optional[Any]
    ) -> InlineKeyboardMarkup:
        """Build a persistent control panel for agentic mode."""
        rows = [
            [
                InlineKeyboardButton("🎛️ Панель", callback_data="act:panel"),
                InlineKeyboardButton("📁 Проекты", callback_data="act:projects"),
                InlineKeyboardButton("🧵 Задачи", callback_data="act:jobs"),
            ],
            [
                InlineKeyboardButton("🕘 Недавнее", callback_data="act:recent"),
                InlineKeyboardButton("📂 Статус", callback_data="act:status"),
                InlineKeyboardButton("📡 Запущено", callback_data="act:running"),
            ],
            [
                InlineKeyboardButton("📊 Статистика", callback_data="act:stats"),
                InlineKeyboardButton("🩺 Диагностика", callback_data="act:doctor"),
            ],
        ]

        if profile:
            action_row = [InlineKeyboardButton("🆕 Новая сессия", callback_data="act:new")]
            if profile.has_git_repo:
                action_row.insert(
                    0, InlineKeyboardButton("🧾 Ревью", callback_data="act:review")
                )
            rows.append(action_row)

            dynamic_row = []
            if "install" in profile.commands:
                dynamic_row.append(
                    InlineKeyboardButton("📦 Подготовка", callback_data="act:setup")
                )
            if "test" in profile.commands:
                dynamic_row.append(
                    InlineKeyboardButton("🧪 Тесты", callback_data="act:test")
                )
            if "lint" in profile.commands or "format" in profile.commands:
                dynamic_row.append(
                    InlineKeyboardButton("🧹 Качество", callback_data="act:quality")
                )
            if dynamic_row:
                rows.append(dynamic_row[:3])
            if self._build_agentic_verify_steps(profile):
                rows.append([InlineKeyboardButton("✅ Проверить", callback_data="act:verify")])
            operator_row = []
            if "health" in profile.commands:
                operator_row.append(
                    InlineKeyboardButton("❤️ Проверка", callback_data="act:health")
                )
            if "build" in profile.commands:
                operator_row.append(
                    InlineKeyboardButton("🏗️ Сборка", callback_data="act:build")
                )
            if operator_row:
                rows.append(operator_row[:3])
            primary_service = self._select_agentic_primary_service(profile)
            if primary_service:
                service_row = []
                for action_key, label in (
                    ("status", "📟 Сервис"),
                    ("logs", "📜 Логи"),
                    ("restart", "🔄 Рестарт"),
                ):
                    if primary_service.command_for(action_key):
                        service_row.append(
                            InlineKeyboardButton(
                                label,
                                callback_data=f"act:svc:{primary_service.key}:{action_key}",
                            )
                        )
                if service_row:
                    rows.append(service_row[:3])
            if profile.services:
                rows.append(
                    [InlineKeyboardButton("🧩 Сервисы", callback_data="act:services")]
                )
            background_row = []
            if "start" in profile.commands:
                background_row.append(
                    InlineKeyboardButton("▶️ Запуск", callback_data="act:start")
                )
            if "dev" in profile.commands:
                background_row.append(
                    InlineKeyboardButton("🛠️ Разработка", callback_data="act:dev")
                )
            if "deploy" in profile.commands:
                background_row.append(
                    InlineKeyboardButton("🚀 Деплой", callback_data="act:deploy")
                )
            if background_row:
                rows.append(background_row[:3])
        else:
            rows.append([InlineKeyboardButton("🆕 Новая сессия", callback_data="act:new")])

        rows.append(
            [
                InlineKeyboardButton("🔇 Коротко", callback_data="act:v0"),
                InlineKeyboardButton("🔉 Нормально", callback_data="act:v1"),
                InlineKeyboardButton("🔊 Подробно", callback_data="act:v2"),
            ]
        )
        return InlineKeyboardMarkup(rows)

    async def _build_agentic_status_text(
        self, context: ContextTypes.DEFAULT_TYPE, user_id: int
    ) -> str:
        """Build compact agentic status text with workspace metadata."""
        current_dir, current_workspace, boundary_root, project_automation, profile = (
            self._get_agentic_workspace_profile(context)
        )
        session_id = context.user_data.get("claude_session_id")
        session_status = "активна" if session_id else "нет"

        claude_integration = context.bot_data.get("claude_integration")
        if not session_id and claude_integration:
            existing = await claude_integration._find_resumable_session(
                user_id, current_workspace
            )
            if existing:
                session_status = f"можно восстановить {existing.session_id[:8]}..."

        cost_str = ""
        rate_limiter = context.bot_data.get("rate_limiter")
        if rate_limiter:
            try:
                user_status = rate_limiter.get_user_status(user_id)
                cost_usage = user_status.get("cost_usage", {})
                current_cost = cost_usage.get("current", 0.0)
                cost_str = f"${current_cost:.2f}"
            except Exception:
                cost_str = "н/д"

        task_parts = []
        active = self._active_tasks.get(user_id)
        if active and not active.task.done():
            elapsed = int(time.time() - active.started_at)
            task_parts.append(f"задача выполняется {elapsed}с")
            queue_size = len(self._pending_messages.get(user_id, []))
            if queue_size:
                task_parts.append(f"в очереди {queue_size}")

        lines = [
            "<b>Статус</b>",
            "",
            f"📦 Проект: <code>{escape_html(self._format_agentic_relative_path(current_workspace, boundary_root))}</code>",
            f"📂 Папка: <code>{escape_html(self._format_agentic_relative_path(current_dir, boundary_root))}</code>",
            f"🤖 Сессия: <code>{escape_html(session_status)}</code>",
            f"🔊 Режим ответа: <code>{escape_html({0: 'коротко', 1: 'нормально', 2: 'подробно'}[self._get_verbose_level(context)])}</code>",
            f"💰 Стоимость: <code>{escape_html(cost_str or 'н/д')}</code>",
        ]
        if task_parts:
            lines.append(f"⚙️ Выполнение: <code>{escape_html(' · '.join(task_parts))}</code>")
        if profile and project_automation:
            playbooks = ", ".join(
                playbook.slug for playbook in project_automation.list_playbooks(profile)
            )
            lines.append(
                f"🧭 Сценарии: <code>{escape_html(playbooks or 'нет')}</code>"
            )
            operator_commands = ", ".join(
                key for key, _command in project_automation.list_operator_commands(profile)
            )
            if operator_commands:
                lines.append(
                    f"🧰 Операции: <code>{escape_html(operator_commands)}</code>"
                )
            if profile.services:
                service_names = ", ".join(
                    service.display_name for service in profile.services
                )
                lines.append(
                    f"🧩 Сервисы: <code>{escape_html(service_names)}</code>"
                )
        operator_runtime = self._get_agentic_operator_runtime(context)
        if operator_runtime:
            latest_job = operator_runtime.get_latest_job(current_workspace)
            if latest_job:
                lines.append(
                    f"🧵 Задача: <code>{escape_html(self._format_agentic_job_status(latest_job, boundary_root))}</code>"
                )
        return "\n".join(lines)

    async def _build_agentic_panel_text(
        self, context: ContextTypes.DEFAULT_TYPE, user_id: int
    ) -> str:
        """Build the main control panel text for agentic mode."""
        _current_dir, current_workspace, boundary_root, project_automation, profile = (
            self._get_agentic_workspace_profile(context)
        )
        lines = ["<b>Панель управления</b>", ""]
        if profile:
            lines.append(
                f"📦 Проект: <code>{escape_html(self._format_agentic_relative_path(current_workspace, boundary_root))}</code>"
            )
            lines.append(
                f"🧱 Стек: <code>{escape_html(', '.join(profile.stacks))}</code>"
            )
        lines.append("🛡️ Автопилот: <code>включен</code>")
        lines.append(
            f"🔊 Режим ответа: <code>{escape_html({0: 'коротко', 1: 'нормально', 2: 'подробно'}[self._get_verbose_level(context)])}</code>"
        )

        claude_integration = context.bot_data.get("claude_integration")
        session_id = context.user_data.get("claude_session_id")
        session_text = "нет"
        if session_id:
            session_text = f"активна {session_id[:8]}..."
        elif claude_integration:
            existing = await claude_integration._find_resumable_session(
                user_id, current_workspace
            )
            if existing:
                session_text = f"можно восстановить {existing.session_id[:8]}..."
        lines.append(f"🤖 Сессия: <code>{escape_html(session_text)}</code>")

        if profile and project_automation:
            playbooks = ", ".join(
                playbook.slug for playbook in project_automation.list_playbooks(profile)
            )
            lines.append(
                f"🧭 Сценарии: <code>{escape_html(playbooks or 'нет')}</code>"
            )
            operator_commands = ", ".join(
                key for key, _command in project_automation.list_operator_commands(profile)
            )
            if operator_commands:
                lines.append(
                    f"🧰 Операции: <code>{escape_html(operator_commands)}</code>"
                )
            if profile.services:
                service_names = ", ".join(
                    service.display_name for service in profile.services
                )
                lines.append(
                    f"🧩 Сервисы: <code>{escape_html(service_names)}</code>"
                )
            if profile.operator_notes:
                note_preview = profile.operator_notes[:160]
                if len(profile.operator_notes) > 160:
                    note_preview += "..."
                lines.extend(
                    [
                        "",
                        f"📝 Заметки: {escape_html(note_preview)}",
                    ]
                )
        operator_runtime = self._get_agentic_operator_runtime(context)
        if operator_runtime:
            latest_job = operator_runtime.get_latest_job(current_workspace)
            if latest_job:
                lines.extend(
                    [
                        "",
                        f"🧵 Последняя задача: <code>{escape_html(self._format_agentic_job_status(latest_job, boundary_root))}</code>",
                    ]
                )

        lines.extend(
            [
                "",
                "Используй кнопки ниже, чтобы посмотреть проект, переключить workspace, запустить сценарий или фоновую операцию.",
            ]
        )
        return "\n".join(lines)

    async def _build_agentic_recent_text(
        self, context: ContextTypes.DEFAULT_TYPE, user_id: int
    ) -> str:
        """Build a compact recent activity view for the control panel."""
        storage = context.bot_data.get("storage")
        if not storage:
            return "Хранилище недоступно."

        current_dir, _current_workspace, boundary_root, _project_automation, _profile = (
            self._get_agentic_workspace_profile(context)
        )
        audit_entries = await storage.audit.get_user_audit_log(user_id, limit=10)
        messages = await storage.messages.get_user_messages(user_id, limit=4)
        command_entries = [
            entry for entry in audit_entries if entry.event_type == "command"
        ][:4]
        automation_entries = [
            entry for entry in audit_entries if entry.event_type == "automation_run"
        ][:4]

        lines = ["<b>Недавняя активность</b>"]
        if automation_entries:
            lines.extend(["", "<b>Автопилот</b>"])
            for entry in automation_entries:
                details = (entry.event_data or {}).get("details", {})
                playbook = escape_html(str(details.get("playbook", "general")))
                workspace = escape_html(
                    self._format_agentic_relative_path(
                        Path(str(details.get("workspace_root", current_dir))),
                        boundary_root,
                    )
                )
                result = "✅" if entry.success else "⚠️"
                lines.append(
                    f"{result} <code>{playbook}</code> · <code>{workspace}</code>"
                )

        if command_entries:
            lines.extend(["", "<b>Команды</b>"])
            for entry in command_entries:
                details = (entry.event_data or {}).get("details", {})
                command_name = escape_html(str(details.get("command", "command")))
                lines.append(f"• <code>{command_name}</code>")

        if messages:
            lines.extend(["", "<b>Запросы</b>"])
            for message in messages:
                preview = escape_html(" ".join(message.prompt.split())[:72])
                lines.append(f"• {preview}")

        if len(lines) == 1:
            lines.extend(["", "Пока недавней активности нет."])

        lines.append("")
        lines.append(
            f"Текущий проект: <code>{escape_html(self._format_agentic_relative_path(_current_workspace, boundary_root))}</code>"
        )
        return "\n".join(lines)

    def _format_agentic_job_status(self, job: Any, boundary_root: Path) -> str:
        """Build a compact one-line status for a persisted operator job."""
        workspace = self._format_agentic_relative_path(job.workspace_root, boundary_root)
        status = f"{job.status} {job.action_key}"
        verification_label = self._format_agentic_job_verification(job)
        if verification_label:
            status += f" · {verification_label}"
        return f"{workspace} · {status} · {job.job_id[:8]}"

    @staticmethod
    def _format_agentic_job_verification(job: Any) -> Optional[str]:
        """Return a compact verification label for background jobs."""
        if not getattr(job, "verification_command", None):
            return None

        status = getattr(job, "verification_status", None) or "pending"
        label = {
            "pending": "проверка ожидается",
            "running": "идет проверка",
            "passed": "проверка пройдена",
            "failed": "проверка не пройдена",
        }.get(status, f"проверка {status}")

        attempts = getattr(job, "verification_attempts", 0) or 0
        if attempts and status in {"running", "passed", "failed"}:
            label += f" ({attempts}x)"
        return label

    async def _build_agentic_jobs_text(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        header: Optional[str] = None,
    ) -> tuple[str, InlineKeyboardMarkup]:
        """Render recent background workspace jobs and management buttons."""
        _current_dir, current_workspace, boundary_root, _project_automation, profile = (
            self._get_agentic_workspace_profile(context)
        )
        operator_runtime = self._get_agentic_operator_runtime(context)
        if not operator_runtime:
            return "Фоновые задачи недоступны.", self._build_agentic_start_keyboard()

        jobs = operator_runtime.list_jobs(limit=8)
        current_jobs = operator_runtime.list_jobs(workspace_root=current_workspace, limit=4)

        lines = ["<b>Фоновые задачи</b>"]
        if header:
            lines.extend(["", header])

        if current_jobs:
            lines.extend(["", "<b>Текущий проект</b>"])
            for job in current_jobs:
                lines.append(
                    f"• <code>{escape_html(self._format_agentic_job_status(job, boundary_root))}</code>"
                )
        if jobs:
            other_jobs = [job for job in jobs if job.workspace_root != current_workspace][:4]
            if other_jobs:
                lines.extend(["", "<b>Недавние</b>"])
                for job in other_jobs:
                    lines.append(
                        f"• <code>{escape_html(self._format_agentic_job_status(job, boundary_root))}</code>"
                    )
        if len(lines) == 1:
            lines.extend(["", "Фоновых задач пока нет."])

        latest_job = current_jobs[0] if current_jobs else None
        if latest_job:
            lines.extend(
                [
                    "",
                    "<b>Последняя задача</b>",
                    f"Действие: <code>{escape_html(latest_job.action_key)}</code>",
                    f"Статус: <code>{escape_html(latest_job.status)}</code>",
                ]
            )
            if latest_job.exit_code is not None:
                lines.append(f"Код выхода: <code>{latest_job.exit_code}</code>")
            if latest_job.verification_command:
                verify_status = self._format_agentic_job_verification(latest_job)
                if verify_status:
                    lines.append(f"Проверка: <code>{escape_html(verify_status)}</code>")
                if latest_job.verification_exit_code is not None:
                    lines.append(
                        f"Код проверки: <code>{latest_job.verification_exit_code}</code>"
                    )
                if latest_job.verification_error:
                    lines.append(
                        f"Ошибка проверки: <code>{escape_html(latest_job.verification_error)}</code>"
                    )
            log_tail = operator_runtime.read_log_tail(latest_job, limit=500)
            if log_tail:
                lines.extend(
                    [
                        "",
                        "<b>Последние логи</b>",
                        f"<pre>{escape_html(log_tail)}</pre>",
                    ]
                )

        keyboard_rows: List[list] = []
        if latest_job and latest_job.is_active:
            keyboard_rows.append(
                [
                    InlineKeyboardButton(
                        f"🛑 Остановить {latest_job.action_key}",
                        callback_data=f"act:stop:{latest_job.job_id}",
                    )
                ]
            )

        if profile:
            action_row = []
            for key, label in (
                ("start", "▶️ Запуск"),
                ("dev", "🛠️ Разработка"),
                ("deploy", "🚀 Деплой"),
            ):
                if key in profile.commands:
                    action_row.append(
                        InlineKeyboardButton(label, callback_data=f"act:{key}")
                    )
            if action_row:
                keyboard_rows.append(action_row[:3])

        keyboard_rows.append(
            [
                InlineKeyboardButton("🔄 Обновить", callback_data="act:jobs"),
                InlineKeyboardButton("🎛️ Панель", callback_data="act:panel"),
                InlineKeyboardButton("📂 Статус", callback_data="act:status"),
            ]
        )
        keyboard_rows.append(
            [InlineKeyboardButton("📁 Проекты", callback_data="act:projects")]
        )
        return "\n".join(lines), InlineKeyboardMarkup(keyboard_rows)

    @staticmethod
    def _format_agentic_service_action_label(service: Any, action_key: str) -> str:
        """Build a compact button label for a managed service action."""
        short = service.display_name.strip() or service.key
        if len(short) > 12:
            short = short.split()[0]
        labels = {
            "status": f"📟 {short}",
            "health": f"🩺 {short}",
            "logs": f"📜 {short}",
            "restart": f"🔄 {short}",
            "start": f"▶️ {short}",
            "stop": f"🛑 {short}",
        }
        return labels.get(action_key, f"{short} {action_key}")

    async def _build_agentic_services_text(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        header: Optional[str] = None,
    ) -> tuple[str, InlineKeyboardMarkup]:
        """Render managed service definitions for the current workspace."""
        _current_dir, current_workspace, boundary_root, _project_automation, profile = (
            self._get_agentic_workspace_profile(context)
        )
        if not profile or not profile.services:
            return (
                "Для этого проекта управляемые сервисы не настроены.",
                self._build_agentic_control_panel_markup(profile),
            )

        lines = [
            "<b>Управляемые сервисы</b>",
            "",
            f"Проект: <code>{escape_html(self._format_agentic_relative_path(current_workspace, boundary_root))}</code>",
            "Действия запуска и рестарта автоматически выполняют проверку и прикладывают логи при ошибке.",
        ]
        if header:
            lines.extend(["", header])

        for service in profile.services:
            actions = ", ".join(service.available_actions) or "none"
            lines.extend(
                [
                    "",
                    f"• <b>{escape_html(service.display_name)}</b> · <code>{escape_html(service.service_type)}</code>",
                    f"  действия: <code>{escape_html(actions)}</code>",
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
                            self._format_agentic_service_action_label(
                                service, action_key
                            ),
                            callback_data=f"act:svc:{service.key}:{action_key}",
                        )
                    )
            for action_key in ("restart", "start", "stop"):
                if service.command_for(action_key):
                    lifecycle_row.append(
                        InlineKeyboardButton(
                            self._format_agentic_service_action_label(
                                service, action_key
                            ),
                            callback_data=f"act:svc:{service.key}:{action_key}",
                        )
                    )
            if inspect_row:
                keyboard_rows.append(inspect_row[:3])
            if lifecycle_row:
                keyboard_rows.append(lifecycle_row[:3])

        keyboard_rows.append(
            [
                InlineKeyboardButton("🔄 Обновить", callback_data="act:services"),
                InlineKeyboardButton("🎛️ Панель", callback_data="act:panel"),
                InlineKeyboardButton("📂 Статус", callback_data="act:status"),
            ]
        )
        keyboard_rows.append(
            [InlineKeyboardButton("📁 Проекты", callback_data="act:projects")]
        )
        return "\n".join(lines), InlineKeyboardMarkup(keyboard_rows)

    async def _build_agentic_running_services_text(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        header: Optional[str] = None,
    ) -> tuple[str, InlineKeyboardMarkup]:
        """Render a live view of managed and system-level running services."""
        _current_dir, current_workspace, boundary_root, _project_automation, profile = (
            self._get_agentic_workspace_profile(context)
        )

        lines = [
            "<b>Запущенные сервисы</b>",
            "",
            f"Проект: <code>{escape_html(self._format_agentic_relative_path(current_workspace, boundary_root))}</code>",
        ]
        if header:
            lines.extend(["", header])

        if profile and profile.services:
            lines.extend(["", "<b>Управляемые сервисы</b>"])
            for service in profile.services:
                command = service.health_command or service.status_command
                if not command:
                    lines.append(
                        f"• <code>{escape_html(service.display_name)}</code>: <code>нет live-проверки</code>"
                    )
                    continue
                result = await self._execute_agentic_shell_action(
                    workspace_root=profile.root_path,
                    command=command,
                    timeout_seconds=45,
                )
                state = "ok" if result.success else "ошибка"
                if result.timed_out:
                    state = "таймаут"
                lines.append(
                    f"• <code>{escape_html(service.display_name)}</code>: <code>{escape_html(state)}</code>"
                )
                summary = self._summarize_agentic_shell_result(result)
                if summary and summary != "нет вывода":
                    lines.append(f"  <code>{escape_html(summary)}</code>")
        else:
            lines.extend(["", "Для этого проекта управляемые сервисы не настроены."])

        running_units_result = await self._list_agentic_running_systemd_units(
            current_workspace
        )
        running_units = self._parse_agentic_systemd_units(running_units_result)
        lines.extend(["", "<b>Системные сервисы сервера</b>"])
        if running_units:
            for unit in running_units:
                lines.append(f"• <code>{escape_html(unit)}</code>")
        else:
            summary = self._summarize_agentic_shell_result(running_units_result)
            label = "список systemd недоступен"
            if running_units_result.success:
                label = "запущенные сервисы не найдены"
            lines.append(f"<code>{escape_html(label)}</code>")
            if summary and summary not in {"нет вывода", label}:
                lines.append(f"<code>{escape_html(summary)}</code>")

        failed_units_result = await self._list_agentic_failed_systemd_units(
            current_workspace
        )
        failed_units = self._parse_agentic_systemd_units(failed_units_result, limit=6)
        if failed_units:
            lines.extend(["", "<b>Сервисы с ошибками</b>"])
            for unit in failed_units:
                lines.append(f"• <code>{escape_html(unit)}</code>")

        keyboard_rows = [
            [
                InlineKeyboardButton("🔄 Обновить", callback_data="act:running"),
                InlineKeyboardButton("🎛️ Панель", callback_data="act:panel"),
                InlineKeyboardButton("📂 Статус", callback_data="act:status"),
            ],
            [
                InlineKeyboardButton("🧩 Сервисы", callback_data="act:services"),
                InlineKeyboardButton("📁 Проекты", callback_data="act:projects"),
            ],
        ]
        return "\n".join(lines), InlineKeyboardMarkup(keyboard_rows)

    @staticmethod
    def _resolve_agentic_service(profile: Any, service_key: str) -> Optional[Any]:
        """Resolve a managed service from the current project profile."""
        if not profile:
            return None
        for service in getattr(profile, "services", ()):
            if service.key == service_key:
                return service
        return None

    async def _build_agentic_workspace_catalog(
        self, context: ContextTypes.DEFAULT_TYPE
    ) -> tuple[str, InlineKeyboardMarkup]:
        """Build the workspace catalog view used by /repo and control buttons."""
        current_dir, current_workspace, boundary_root, project_automation, profile = (
            self._get_agentic_workspace_profile(context)
        )
        if project_automation:
            summaries = project_automation.list_workspace_summaries(boundary_root)
            if summaries:
                lines: List[str] = ["<b>Проекты</b>", ""]
                for summary in summaries:
                    lines.extend(
                        project_automation.describe_workspace_summary_lines(
                            summary, current_workspace=current_workspace
                        )
                    )
                lines.extend(
                    [
                        "",
                        "Автопилот умеет выбирать проект по имени, алиасу или относительному пути.",
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
                        InlineKeyboardButton("🔄 Обновить", callback_data="act:projects"),
                        InlineKeyboardButton("🎛️ Панель", callback_data="act:panel"),
                        InlineKeyboardButton("📂 Статус", callback_data="act:status"),
                    ]
                )
                keyboard_rows.append(
                    [InlineKeyboardButton("🕘 Недавнее", callback_data="act:recent")]
                )
                return "\n".join(lines), InlineKeyboardMarkup(keyboard_rows)

        lines = ["<b>Проекты</b>", ""]
        try:
            entries = sorted(
                [
                    d
                    for d in boundary_root.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ],
                key=lambda d: d.name.casefold(),
            )
        except OSError as e:
            return f"Ошибка чтения проекта: {escape_html(str(e))}", self._build_agentic_start_keyboard()

        keyboard_rows: List[list] = []
        for entry in entries:
            marker = " ◀" if entry == current_workspace else ""
            lines.append(f"• <code>{escape_html(entry.name)}</code>{marker}")
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
            keyboard_rows.append(row)
        keyboard_rows.append(
            [
                InlineKeyboardButton("🔄 Обновить", callback_data="act:projects"),
                InlineKeyboardButton("🎛️ Панель", callback_data="act:panel"),
            ]
        )
        return "\n".join(lines), InlineKeyboardMarkup(keyboard_rows)

    async def _run_agentic_playbook_action(
        self,
        query: Any,
        context: ContextTypes.DEFAULT_TYPE,
        playbook_slug: str,
    ) -> None:
        """Run a deterministic playbook from an agentic control button."""
        user_id = query.from_user.id
        current_dir, current_workspace, boundary_root, project_automation, profile = (
            self._get_agentic_workspace_profile(context)
        )
        claude_integration = context.bot_data.get("claude_integration")
        if not claude_integration or not project_automation or not profile:
            await query.edit_message_text("Автоматизация проекта недоступна.")
            return

        playbook = project_automation.get_playbook(playbook_slug, profile)
        if playbook is None:
            await query.answer("Сценарий недоступен для этого проекта.", show_alert=True)
            return

        prompt = project_automation.build_playbook_prompt(playbook_slug, profile)
        status_msg = await query.message.reply_text(
            "▶️ <b>Запуск сценария</b>\n\n"
            f"Сценарий: <code>{escape_html(playbook.slug)}</code>\n"
            f"Проект: <code>{escape_html(self._format_agentic_relative_path(profile.root_path, boundary_root))}</code>",
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
                    logger.warning("Failed to log button playbook interaction", error=str(e))

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
                    (result for result in verification_results if not result.success),
                    None,
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

            from .utils.formatting import ResponseFormatter

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
                            "command": result.command,
                            "success": result.success,
                            "returncode": result.returncode,
                        }
                        for result in guard_report.verification_results
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

    @staticmethod
    def _tail_command_output(text: str, limit: int = 900) -> str:
        """Keep only the tail of command output for Telegram summaries."""
        normalized = text.strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[-limit:]

    async def _execute_agentic_shell_action(
        self,
        workspace_root: Path,
        command: str,
        timeout_seconds: int = 120,
    ) -> _ShellActionResult:
        """Execute a deterministic shell action and capture a redacted result."""
        timed_out = False
        try:
            process = await asyncio.create_subprocess_exec(
                "/bin/sh",
                "-lc",
                command,
                cwd=workspace_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                timed_out = True
                process.kill()
                stdout, stderr = await process.communicate()
            returncode = process.returncode if process.returncode is not None else -1
            stdout_text = self._tail_command_output(
                _redact_secrets(stdout.decode("utf-8", errors="replace"))
            )
            stderr_text = self._tail_command_output(
                _redact_secrets(stderr.decode("utf-8", errors="replace"))
            )
            return _ShellActionResult(
                command=command,
                returncode=returncode,
                success=not timed_out and returncode == 0,
                timed_out=timed_out,
                stdout_text=stdout_text,
                stderr_text=stderr_text,
            )
        except Exception as e:
            return _ShellActionResult(
                command=command,
                returncode=-1,
                success=False,
                timed_out=False,
                stdout_text="",
                stderr_text="",
                error=str(e),
            )

    def _build_agentic_shell_action_lines(
        self,
        title: str,
        workspace_root: Path,
        boundary_root: Path,
        result: _ShellActionResult,
    ) -> List[str]:
        """Format a shell action result into Telegram-friendly lines."""
        if result.error:
            return [
                "❌ <b>Ошибка команды</b>",
                "",
                f"Действие: <code>{escape_html(title)}</code>",
                f"Ошибка: <code>{escape_html(result.error)}</code>",
            ]

        lines = [
            f"✅ <b>{escape_html(title)}</b>"
            if result.success
            else f"❌ <b>{escape_html(title)}</b>",
            "",
            f"Проект: <code>{escape_html(self._format_agentic_relative_path(workspace_root, boundary_root))}</code>",
            f"Команда: <code>{escape_html(result.command)}</code>",
            f"Код выхода: <code>{result.returncode}</code>",
        ]
        if result.timed_out:
            lines.append("Вышло время ожидания.")
        if result.stdout_text:
            lines.extend(["", "<b>stdout</b>", f"<pre>{escape_html(result.stdout_text)}</pre>"])
        if result.stderr_text:
            lines.extend(["", "<b>stderr</b>", f"<pre>{escape_html(result.stderr_text)}</pre>"])
        return lines

    @staticmethod
    def _summarize_agentic_shell_result(result: _ShellActionResult, limit: int = 140) -> str:
        """Build a compact one-line summary of a shell action result."""
        if result.error:
            summary = result.error
        else:
            summary = result.stdout_text or result.stderr_text or ""
        compact = " ".join(summary.split())
        if not compact:
            compact = "нет вывода"
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."

    @staticmethod
    def _select_agentic_background_verification(profile: Any) -> Optional[str]:
        """Select the best health command for a background action."""
        command = profile.commands.get("health")
        if command:
            return command

        for service in getattr(profile, "services", ()):
            if getattr(service, "health_command", None):
                return service.health_command
        for service in getattr(profile, "services", ()):
            if getattr(service, "status_command", None):
                return service.status_command
        return None

    @staticmethod
    def _select_agentic_primary_service(profile: Any) -> Optional[Any]:
        """Choose the most useful managed service for one-tap shortcuts."""
        services = list(getattr(profile, "services", ()))
        if not services:
            return None
        for key in ("app", "api", "web"):
            for service in services:
                if service.key == key:
                    return service
        return services[0] if len(services) == 1 else None

    def _build_agentic_verify_steps(self, profile: Any) -> List[_VerifyStep]:
        """Build the deterministic verification sequence for the workspace."""
        steps: List[_VerifyStep] = []
        seen_commands: set[str] = set()

        health_command = profile.commands.get("health")
        if health_command:
            steps.append(_VerifyStep(label="health", command=health_command))
            seen_commands.add(health_command)
        else:
            for service in getattr(profile, "services", ()):
                service_command = service.health_command or service.status_command
                if not service_command or service_command in seen_commands:
                    continue
                label = (
                    f"{service.display_name} проверка"
                    if service.health_command
                    else f"{service.display_name} статус"
                )
                steps.append(
                    _VerifyStep(
                        label=label,
                        command=service_command,
                        logs_command=service.logs_command,
                    )
                )
                seen_commands.add(service_command)

        label_map = {
            "lint": "линт",
            "typecheck": "typecheck",
            "test": "тесты",
            "build": "сборка",
        }
        for key, label in label_map.items():
            command = profile.commands.get(key)
            if command and command not in seen_commands:
                steps.append(_VerifyStep(label=label, command=command))
                seen_commands.add(command)

        return steps

    async def _run_agentic_service_follow_up_checks(
        self,
        service: Any,
        workspace_root: Path,
        action_key: str,
    ) -> tuple[List[tuple[str, _ShellActionResult]], Optional[_ShellActionResult], bool]:
        """Run post-action checks for managed service lifecycle operations."""
        checks: List[tuple[str, _ShellActionResult]] = []
        logs_result: Optional[_ShellActionResult] = None
        all_required_checks_passed = True

        if action_key in {"start", "restart"}:
            await asyncio.sleep(2.0)
            for label, command in (
                ("status", service.command_for("status")),
                ("health", service.command_for("health")),
            ):
                if not command:
                    continue
                result = await self._execute_agentic_shell_action(
                    workspace_root,
                    command,
                    timeout_seconds=45,
                )
                checks.append((label, result))
                if not result.success:
                    all_required_checks_passed = False
            if not all_required_checks_passed and service.command_for("logs"):
                logs_result = await self._execute_agentic_shell_action(
                    workspace_root,
                    service.command_for("logs"),
                    timeout_seconds=30,
                )
        elif action_key == "stop" and service.command_for("status"):
            await asyncio.sleep(1.0)
            status_result = await self._execute_agentic_shell_action(
                workspace_root,
                service.command_for("status"),
                timeout_seconds=30,
            )
            checks.append(("status", status_result))

        return checks, logs_result, all_required_checks_passed

    async def _list_agentic_running_systemd_units(
        self, workspace_root: Path
    ) -> _ShellActionResult:
        """Return the currently running systemd services."""
        return await self._execute_agentic_shell_action(
            workspace_root=workspace_root,
            command=(
                "systemctl list-units --type=service --state=running "
                "--no-pager --plain --no-legend"
            ),
            timeout_seconds=30,
        )

    async def _list_agentic_failed_systemd_units(
        self, workspace_root: Path
    ) -> _ShellActionResult:
        """Return currently failed systemd services."""
        return await self._execute_agentic_shell_action(
            workspace_root=workspace_root,
            command=(
                "systemctl list-units --type=service --state=failed "
                "--no-pager --plain --no-legend"
            ),
            timeout_seconds=30,
        )

    @staticmethod
    def _parse_agentic_systemd_units(result: _ShellActionResult, limit: int = 12) -> List[str]:
        """Extract systemd unit names from list-units output."""
        if not result.success or not result.stdout_text:
            return []
        units: List[str] = []
        for line in result.stdout_text.splitlines():
            compact = line.strip()
            if not compact:
                continue
            unit = compact.split()[0]
            if unit.endswith(".service"):
                units.append(unit)
            if len(units) >= limit:
                break
        return units

    async def _run_agentic_shell_action(
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
        status_msg = await query.message.reply_text(
            "▶️ <b>Выполнение команды</b>\n\n"
            f"Действие: <code>{escape_html(title)}</code>\n"
            f"Проект: <code>{escape_html(self._format_agentic_relative_path(workspace_root, boundary_root))}</code>\n"
            f"Команда: <code>{escape_html(command)}</code>",
            parse_mode="HTML",
        )

        result = await self._execute_agentic_shell_action(
            workspace_root=workspace_root,
            command=command,
            timeout_seconds=timeout_seconds,
        )
        await status_msg.edit_text(
            "\n".join(
                self._build_agentic_shell_action_lines(
                    title=title,
                    workspace_root=workspace_root,
                    boundary_root=boundary_root,
                    result=result,
                )
            ),
            parse_mode="HTML",
        )

        success = result.success
        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=user_id,
                command=audit_command,
                args=[command, str(workspace_root)],
                success=success,
            )

    async def _run_agentic_command_action(
        self,
        query: Any,
        context: ContextTypes.DEFAULT_TYPE,
        command_key: str,
    ) -> None:
        """Run a direct workspace command for safe operator actions."""
        user_id = query.from_user.id
        _current_dir, _current_workspace, boundary_root, project_automation, profile = (
            self._get_agentic_workspace_profile(context)
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
        await self._run_agentic_shell_action(
            query=query,
            context=context,
            workspace_root=profile.root_path,
            boundary_root=boundary_root,
            title=title,
            command=command,
            audit_command=f"workspace_{command_key}",
        )

    async def _run_agentic_service_action(
        self,
        query: Any,
        context: ContextTypes.DEFAULT_TYPE,
        service_key: str,
        action_key: str,
    ) -> None:
        """Run a managed service action from the explicit service catalog."""
        _current_dir, _current_workspace, boundary_root, _project_automation, profile = (
            self._get_agentic_workspace_profile(context)
        )
        service = self._resolve_agentic_service(profile, service_key)
        if not profile or not service:
            await query.answer("Сервис не настроен для этого проекта.", show_alert=True)
            return

        command = service.command_for(action_key)
        if not command:
            await query.answer("Это действие недоступно для сервиса.", show_alert=True)
            return

        title = f"{service.display_name}: {action_key}"
        user_id = query.from_user.id
        status_msg = await query.message.reply_text(
            "▶️ <b>Выполнение действия сервиса</b>\n\n"
            f"Сервис: <code>{escape_html(service.display_name)}</code>\n"
            f"Действие: <code>{escape_html(action_key)}</code>\n"
            f"Проект: <code>{escape_html(self._format_agentic_relative_path(profile.root_path, boundary_root))}</code>\n"
            f"Команда: <code>{escape_html(command)}</code>",
            parse_mode="HTML",
        )

        main_result = await self._execute_agentic_shell_action(
            workspace_root=profile.root_path,
            command=command,
            timeout_seconds=120,
        )

        checks: List[tuple[str, _ShellActionResult]] = []
        logs_result: Optional[_ShellActionResult] = None
        checks_ok = True
        if action_key in {"start", "restart", "stop"} and main_result.success:
            await status_msg.edit_text(
                "⏳ <b>Ожидание проверки сервиса</b>\n\n"
                f"Сервис: <code>{escape_html(service.display_name)}</code>\n"
                f"Действие: <code>{escape_html(action_key)}</code>",
                parse_mode="HTML",
            )
            checks, logs_result, checks_ok = await self._run_agentic_service_follow_up_checks(
                service=service,
                workspace_root=profile.root_path,
                action_key=action_key,
            )
        elif not main_result.success and action_key in {"start", "restart"} and service.command_for("logs"):
            logs_result = await self._execute_agentic_shell_action(
                profile.root_path,
                service.command_for("logs"),
                timeout_seconds=30,
            )

        final_success = main_result.success and checks_ok
        lines = [
            "✅ <b>Действие сервиса выполнено</b>"
            if final_success
            else "❌ <b>Ошибка действия сервиса</b>",
            "",
            f"Сервис: <code>{escape_html(service.display_name)}</code>",
            f"Действие: <code>{escape_html(action_key)}</code>",
            f"Проект: <code>{escape_html(self._format_agentic_relative_path(profile.root_path, boundary_root))}</code>",
            f"Команда: <code>{escape_html(command)}</code>",
            f"Код выхода: <code>{main_result.returncode}</code>",
        ]
        if main_result.error:
            lines.append(f"Ошибка: <code>{escape_html(main_result.error)}</code>")
        elif main_result.timed_out:
            lines.append("Вышло время ожидания.")
        if main_result.stdout_text:
            lines.extend(["", "<b>stdout</b>", f"<pre>{escape_html(main_result.stdout_text)}</pre>"])
        if main_result.stderr_text:
            lines.extend(["", "<b>stderr</b>", f"<pre>{escape_html(main_result.stderr_text)}</pre>"])

        if checks:
            lines.extend(["", "<b>Дополнительные проверки</b>"])
            for label, result in checks:
                check_state = "ok" if result.success else "ошибка"
                if result.timed_out:
                    check_state = "таймаут"
                lines.append(
                    f"• <code>{escape_html(label)}</code>: <code>{escape_html(check_state)}</code> "
                    f"(exit <code>{result.returncode}</code>)"
                )
                summary = self._summarize_agentic_shell_result(result)
                if summary and summary != "нет вывода":
                    lines.append(f"  <code>{escape_html(summary)}</code>")

        if logs_result and (logs_result.stdout_text or logs_result.stderr_text or logs_result.error):
            log_body = logs_result.stdout_text or logs_result.stderr_text
            if logs_result.error:
                log_body = logs_result.error
            lines.extend(
                [
                    "",
                    "<b>Логи сервиса</b>",
                    f"<pre>{escape_html(log_body)}</pre>",
                ]
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

    async def _run_agentic_verify_action(
        self,
        query: Any,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Run the deterministic verification suite for the current workspace."""
        _current_dir, _current_workspace, boundary_root, project_automation, profile = (
            self._get_agentic_workspace_profile(context)
        )
        if not project_automation or not profile:
            await query.edit_message_text("Автоматизация проекта недоступна.")
            return

        steps = self._build_agentic_verify_steps(profile)
        if not steps:
            await query.answer("Для этого проекта шаги проверки не найдены.", show_alert=True)
            return

        status_msg = await query.message.reply_text(
            "▶️ <b>Запуск проверки</b>\n\n"
            f"Проект: <code>{escape_html(self._format_agentic_relative_path(profile.root_path, boundary_root))}</code>\n"
            f"Шаги: <code>{escape_html(', '.join(step.label for step in steps))}</code>",
            parse_mode="HTML",
        )

        results: List[tuple[_VerifyStep, _ShellActionResult]] = []
        logs_result: Optional[_ShellActionResult] = None
        failed_step: Optional[_VerifyStep] = None

        for index, step in enumerate(steps, start=1):
            await status_msg.edit_text(
                "⏳ <b>Выполняю проверку</b>\n\n"
                f"Проект: <code>{escape_html(self._format_agentic_relative_path(profile.root_path, boundary_root))}</code>\n"
                f"Шаг <code>{index}/{len(steps)}</code>: <code>{escape_html(step.label)}</code>\n"
                f"Команда: <code>{escape_html(step.command)}</code>",
                parse_mode="HTML",
            )
            result = await self._execute_agentic_shell_action(
                workspace_root=profile.root_path,
                command=step.command,
                timeout_seconds=180,
            )
            results.append((step, result))
            if not result.success:
                failed_step = step
                if step.logs_command:
                    logs_result = await self._execute_agentic_shell_action(
                        workspace_root=profile.root_path,
                        command=step.logs_command,
                        timeout_seconds=30,
                    )
                break

        success = failed_step is None
        lines = [
            "✅ <b>Проверка завершена</b>"
            if success
            else "❌ <b>Проверка не пройдена</b>",
            "",
            f"Проект: <code>{escape_html(self._format_agentic_relative_path(profile.root_path, boundary_root))}</code>",
        ]
        if failed_step:
            lines.append(f"Ошибка на шаге: <code>{escape_html(failed_step.label)}</code>")
        lines.extend(["", "<b>Шаги</b>"])
        for step, result in results:
            state = "ok" if result.success else "ошибка"
            if result.timed_out:
                state = "таймаут"
            lines.append(
                f"• <code>{escape_html(step.label)}</code>: <code>{escape_html(state)}</code> "
                f"(exit <code>{result.returncode}</code>)"
            )
            summary = self._summarize_agentic_shell_result(result)
            if summary and summary != "нет вывода":
                lines.append(f"  <code>{escape_html(summary)}</code>")

        if results:
            last_step, last_result = results[-1]
            if (
                not last_result.success
                and (last_result.stdout_text or last_result.stderr_text or last_result.error)
            ):
                detail_text = last_result.stdout_text or last_result.stderr_text or last_result.error or ""
                lines.extend(
                    [
                        "",
                        f"<b>Вывод ошибки: {escape_html(last_step.label)}</b>",
                        f"<pre>{escape_html(detail_text)}</pre>",
                    ]
                )

        if logs_result and (logs_result.stdout_text or logs_result.stderr_text or logs_result.error):
            log_body = logs_result.stdout_text or logs_result.stderr_text or logs_result.error or ""
            lines.extend(
                [
                    "",
                    "<b>Логи сервиса</b>",
                    f"<pre>{escape_html(log_body)}</pre>",
                ]
            )

        await status_msg.edit_text("\n".join(lines), parse_mode="HTML")

        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=query.from_user.id,
                command="workspace_verify",
                args=[step.command for step in steps],
                success=success,
            )

    async def _run_agentic_background_action(
        self,
        query: Any,
        context: ContextTypes.DEFAULT_TYPE,
        action_key: str,
    ) -> None:
        """Launch a long-running workspace command in the background."""
        user_id = query.from_user.id
        _current_dir, current_workspace, boundary_root, project_automation, profile = (
            self._get_agentic_workspace_profile(context)
        )
        operator_runtime = self._get_agentic_operator_runtime(context)
        if not project_automation or not profile or not operator_runtime:
            await query.edit_message_text("Фоновые задачи недоступны.")
            return

        command = profile.commands.get(action_key)
        if not command:
            await query.answer("Это действие недоступно для проекта.", show_alert=True)
            return

        verification_command = self._select_agentic_background_verification(profile)
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
            text, reply_markup = await self._build_agentic_jobs_text(context)
            await query.edit_message_text(
                text,
                parse_mode="HTML",
                reply_markup=reply_markup,
            )
            return

        header = (
            "▶️ <b>Фоновая задача запущена</b>\n\n"
            f"Действие: <code>{escape_html(title)}</code>\n"
            f"Проект: <code>{escape_html(self._format_agentic_relative_path(profile.root_path, boundary_root))}</code>\n"
            f"Задача: <code>{escape_html(job.job_id)}</code>\n"
            f"Команда: <code>{escape_html(command)}</code>"
        )
        if verification_command and verification_mode:
            header += (
                "\n"
                f"Проверка после запуска: <code>{escape_html(verification_command)}</code>"
            )
        text, reply_markup = await self._build_agentic_jobs_text(context, header=header)
        await query.edit_message_text(
            text,
            parse_mode="HTML",
            reply_markup=reply_markup,
        )

        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=user_id,
                command=f"workspace_job_{action_key}",
                args=[command, str(current_workspace)],
                success=True,
            )

    async def _stop_agentic_background_action(
        self,
        query: Any,
        context: ContextTypes.DEFAULT_TYPE,
        job_id: str,
    ) -> None:
        """Stop a persisted background workspace job."""
        operator_runtime = self._get_agentic_operator_runtime(context)
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
        text, reply_markup = await self._build_agentic_jobs_text(context, header=header)
        await query.edit_message_text(
            text,
            parse_mode="HTML",
            reply_markup=reply_markup,
        )

        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=query.from_user.id,
                command="workspace_job_stop",
                args=[job_id],
                success=True,
            )

    def _format_verbose_progress(
        self,
        activity_log: List[Dict[str, Any]],
        verbose_level: int,
        start_time: float,
    ) -> str:
        """Build the progress message text based on activity so far."""
        if not activity_log:
            return "Working..."

        elapsed = time.time() - start_time
        lines: List[str] = [f"Working... ({elapsed:.0f}s)\n"]

        for entry in activity_log[-15:]:  # Show last 15 entries max
            kind = entry.get("kind", "tool")
            if kind == "text":
                # Claude's intermediate reasoning/commentary
                snippet = entry.get("detail", "")
                if verbose_level >= 2:
                    lines.append(f"\U0001f4ac {snippet}")
                else:
                    # Level 1: one short line
                    lines.append(f"\U0001f4ac {snippet[:80]}")
            else:
                # Tool call
                icon = _tool_icon(entry["name"])
                if verbose_level >= 2 and entry.get("detail"):
                    lines.append(f"{icon} {entry['name']}: {entry['detail']}")
                else:
                    lines.append(f"{icon} {entry['name']}")

        if len(activity_log) > 15:
            lines.insert(1, f"... ({len(activity_log) - 15} earlier entries)\n")

        return "\n".join(lines)

    @staticmethod
    def _summarize_tool_input(tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Return a short summary of tool input for verbose level 2."""
        if not tool_input:
            return ""
        if tool_name in ("Read", "Write", "Edit", "MultiEdit"):
            path = tool_input.get("file_path") or tool_input.get("path", "")
            if path:
                # Show just the filename, not the full path
                return path.rsplit("/", 1)[-1]
        if tool_name in ("Glob", "Grep"):
            pattern = tool_input.get("pattern", "")
            if pattern:
                return pattern[:60]
        if tool_name == "Bash":
            cmd = tool_input.get("command", "")
            if cmd:
                return _redact_secrets(cmd[:100])[:80]
        if tool_name in ("WebFetch", "WebSearch"):
            return (tool_input.get("url", "") or tool_input.get("query", ""))[:60]
        if tool_name == "Task":
            desc = tool_input.get("description", "")
            if desc:
                return desc[:60]
        # Generic: show first key's value
        for v in tool_input.values():
            if isinstance(v, str) and v:
                return v[:60]
        return ""

    @staticmethod
    def _start_typing_heartbeat(
        chat: Any,
        interval: float = 2.0,
    ) -> "asyncio.Task[None]":
        """Start a background typing indicator task.

        Sends typing every *interval* seconds, independently of
        stream events. Cancel the returned task in a ``finally``
        block.
        """

        async def _heartbeat() -> None:
            try:
                while True:
                    await asyncio.sleep(interval)
                    try:
                        await chat.send_action("typing")
                    except Exception:
                        pass
            except asyncio.CancelledError:
                pass

        return asyncio.create_task(_heartbeat())

    def _make_stream_callback(
        self,
        verbose_level: int,
        progress_msg: Any,
        tool_log: List[Dict[str, Any]],
        start_time: float,
        mcp_images: Optional[List[ImageAttachment]] = None,
        approved_directory: Optional[Path] = None,
        draft_streamer: Optional[DraftStreamer] = None,
    ) -> Optional[Callable[[StreamUpdate], Any]]:
        """Create a stream callback for verbose progress updates.

        When *mcp_images* is provided, the callback also intercepts
        ``send_image_to_user`` tool calls and collects validated
        :class:`ImageAttachment` objects for later Telegram delivery.

        When *draft_streamer* is provided, tool activity and assistant
        text are streamed to the user in real time via
        ``sendMessageDraft``.

        Returns None when verbose_level is 0 **and** no MCP image
        collection or draft streaming is requested.
        Typing indicators are handled by a separate heartbeat task.
        """
        need_mcp_intercept = mcp_images is not None and approved_directory is not None

        if verbose_level == 0 and not need_mcp_intercept and draft_streamer is None:
            return None

        last_edit_time = [0.0]  # mutable container for closure

        async def _on_stream(update_obj: StreamUpdate) -> None:
            # Intercept send_image_to_user MCP tool calls.
            # The SDK namespaces MCP tools as "mcp__<server>__<tool>",
            # so match both the bare name and the namespaced variant.
            if update_obj.tool_calls and need_mcp_intercept:
                for tc in update_obj.tool_calls:
                    tc_name = tc.get("name", "")
                    if tc_name == "send_image_to_user" or tc_name.endswith(
                        "__send_image_to_user"
                    ):
                        tc_input = tc.get("input", {})
                        file_path = tc_input.get("file_path", "")
                        caption = tc_input.get("caption", "")
                        img = validate_image_path(
                            file_path, approved_directory, caption
                        )
                        if img:
                            mcp_images.append(img)

            # Capture tool calls
            if update_obj.tool_calls:
                for tc in update_obj.tool_calls:
                    name = tc.get("name", "unknown")
                    detail = self._summarize_tool_input(name, tc.get("input", {}))
                    if verbose_level >= 1:
                        tool_log.append(
                            {"kind": "tool", "name": name, "detail": detail}
                        )
                    if draft_streamer:
                        icon = _tool_icon(name)
                        line = (
                            f"{icon} {name}: {detail}" if detail else f"{icon} {name}"
                        )
                        await draft_streamer.append_tool(line)

            # Capture assistant text (reasoning / commentary)
            if update_obj.type == "assistant" and update_obj.content:
                text = update_obj.content.strip()
                if text:
                    first_line = text.split("\n", 1)[0].strip()
                    if first_line:
                        if verbose_level >= 1:
                            tool_log.append(
                                {"kind": "text", "detail": first_line[:120]}
                            )
                        if draft_streamer:
                            await draft_streamer.append_tool(
                                f"\U0001f4ac {first_line[:120]}"
                            )

            # Stream text to user via draft (prefer token deltas;
            # skip full assistant messages to avoid double-appending)
            if draft_streamer and update_obj.content:
                if update_obj.type == "stream_delta":
                    await draft_streamer.append_text(update_obj.content)

            # Throttle progress message edits to avoid Telegram rate limits
            if not draft_streamer and verbose_level >= 1:
                now = time.time()
                if (now - last_edit_time[0]) >= 2.0 and tool_log:
                    last_edit_time[0] = now
                    new_text = self._format_verbose_progress(
                        tool_log, verbose_level, start_time
                    )
                    try:
                        await progress_msg.edit_text(new_text)
                    except Exception:
                        pass

        return _on_stream

    async def _send_images(
        self,
        update: Update,
        images: List[ImageAttachment],
        reply_to_message_id: Optional[int] = None,
        caption: Optional[str] = None,
        caption_parse_mode: Optional[str] = None,
    ) -> bool:
        """Send extracted images as a media group (album) or documents.

        If *caption* is provided and fits (≤1024 chars), it is attached to the
        photo / first album item so text + images appear as one message.

        Returns True if the caption was successfully embedded in the photo message.
        """
        photos: List[ImageAttachment] = []
        documents: List[ImageAttachment] = []
        for img in images:
            if should_send_as_photo(img.path):
                photos.append(img)
            else:
                documents.append(img)

        # Telegram caption limit
        use_caption = bool(
            caption and len(caption) <= 1024 and photos and not documents
        )
        caption_sent = False

        # Send raster photos as a single album (Telegram groups 2-10 items)
        if photos:
            try:
                if len(photos) == 1:
                    with open(photos[0].path, "rb") as f:
                        await update.message.reply_photo(
                            photo=f,
                            reply_to_message_id=reply_to_message_id,
                            caption=caption if use_caption else None,
                            parse_mode=caption_parse_mode if use_caption else None,
                        )
                    caption_sent = use_caption
                else:
                    media = []
                    file_handles = []
                    for idx, img in enumerate(photos[:10]):
                        fh = open(img.path, "rb")  # noqa: SIM115
                        file_handles.append(fh)
                        media.append(
                            InputMediaPhoto(
                                media=fh,
                                caption=caption if use_caption and idx == 0 else None,
                                parse_mode=(
                                    caption_parse_mode
                                    if use_caption and idx == 0
                                    else None
                                ),
                            )
                        )
                    try:
                        await update.message.chat.send_media_group(
                            media=media,
                            reply_to_message_id=reply_to_message_id,
                        )
                        caption_sent = use_caption
                    finally:
                        for fh in file_handles:
                            fh.close()
            except Exception as e:
                logger.warning("Failed to send photo album", error=str(e))

        # Send SVGs / large files as documents (one by one — can't mix in album)
        for img in documents:
            try:
                with open(img.path, "rb") as f:
                    await update.message.reply_document(
                        document=f,
                        filename=img.path.name,
                        reply_to_message_id=reply_to_message_id,
                    )
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning(
                    "Failed to send document image",
                    path=str(img.path),
                    error=str(e),
                )

        return caption_sent

    async def agentic_text(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Direct Claude passthrough. Simple progress. No suggestions."""
        user_id = update.effective_user.id
        message_text = update.message.text
        reply_action = self._build_agentic_reply_action(message_text, context)

        if reply_action:
            await self._dispatch_agentic_action(
                reply_action,
                _MessageActionProxy(update),
                context,
            )
            self._mark_agentic_reply_keyboard_ready(context)
            return

        logger.info(
            "Agentic text message",
            user_id=user_id,
            message_length=len(message_text),
        )

        # --- Interrupt / queue handling (lock-protected) ---
        lock = self._user_locks.setdefault(user_id, asyncio.Lock())
        async with lock:
            active = self._active_tasks.get(user_id)
            if active and not active.task.done():
                if _STOP_PATTERNS.match(message_text.strip()):
                    # User wants to cancel the running task
                    active.cancel_requested = True
                    active.task.cancel()
                    self._active_tasks.pop(user_id, None)
                    self._pending_messages.pop(user_id, None)
                    await update.message.reply_text(
                        "Остановлено. Можешь дать новое задание."
                    )
                    logger.info("User cancelled active task", user_id=user_id)
                    return
                else:
                    # Queue this message — processed after current task
                    pending = self._pending_messages.setdefault(user_id, [])
                    pending.append(_PendingMessage(update=update, context=context))
                    queue_pos = len(pending)
                    await update.message.reply_text(
                        f"Принял (в очереди: {queue_pos}). Отвечу после текущей задачи.",
                        disable_notification=True,
                    )
                    logger.info(
                        "Message queued while task active",
                        user_id=user_id,
                        queue_size=queue_pos,
                    )
                    return

        # Rate limit check
        rate_limiter = context.bot_data.get("rate_limiter")
        if rate_limiter:
            allowed, limit_message = await rate_limiter.check_rate_limit(user_id, 0.001)
            if not allowed:
                await update.message.reply_text(f"⏱️ {limit_message}")
                return

        chat = update.message.chat
        await chat.send_action("typing")

        # Check for YouTube URLs and extract transcript
        message_text = await self._enrich_with_video_transcript(message_text)

        verbose_level = self._get_verbose_level(context)
        progress_msg = await update.message.reply_text("⌛", disable_notification=True)

        claude_integration = context.bot_data.get("claude_integration")
        if not claude_integration:
            await progress_msg.edit_text(
                "Claude integration not available. Check configuration."
            )
            return

        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        session_id = context.user_data.get("claude_session_id")
        working_dir = current_dir
        guard_report: Optional[ChangeGuardReport] = None
        checkpoint = None

        features = context.bot_data.get("features")
        project_automation = (
            getattr(features, "get_project_automation", lambda: None)()
            if features
            else None
        )
        change_guard = (
            getattr(features, "get_project_change_guard", lambda: None)()
            if features
            else None
        )

        # Check if /new was used — skip auto-resume for this first message.
        # Flag is only cleared after a successful run so retries keep the intent.
        force_new = bool(context.user_data.get("force_new_session"))

        # --- Verbose progress tracking via stream callback ---
        tool_log: List[Dict[str, Any]] = []
        start_time = time.time()
        mcp_images: List[ImageAttachment] = []

        # Stream drafts (private chats only)
        draft_streamer: Optional[DraftStreamer] = None
        if self.settings.enable_stream_drafts and chat.type == "private":
            draft_streamer = DraftStreamer(
                bot=context.bot,
                chat_id=chat.id,
                draft_id=generate_draft_id(),
                message_thread_id=update.message.message_thread_id,
                throttle_interval=self.settings.stream_draft_interval,
            )

        on_stream = self._make_stream_callback(
            verbose_level,
            progress_msg,
            tool_log,
            start_time,
            mcp_images=mcp_images,
            approved_directory=self.settings.approved_directory,
            draft_streamer=draft_streamer,
        )

        # Independent typing heartbeat — stays alive even with no stream events
        heartbeat = self._start_typing_heartbeat(chat)

        # Prepend custom instructions to prompt if set
        custom_instructions = context.user_data.get("custom_instructions", [])
        final_prompt = message_text
        autopilot_plan = None
        if project_automation:
            autopilot_plan = project_automation.build_automation_plan(
                message_text,
                current_dir,
                self._get_boundary_root(context),
            )
            final_prompt = autopilot_plan.prompt
            working_dir = autopilot_plan.workspace_root
            if autopilot_plan.workspace_changed:
                context.user_data["current_directory"] = working_dir
                session_id = None
                if not force_new:
                    try:
                        existing_session = await claude_integration._find_resumable_session(
                            user_id, working_dir
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to resume session for autopilot workspace",
                            error=str(e),
                            user_id=user_id,
                            working_dir=str(working_dir),
                        )
                    else:
                        if existing_session:
                            session_id = existing_session.session_id
                context.user_data["claude_session_id"] = session_id
            if autopilot_plan.should_checkpoint and change_guard:
                checkpoint = await change_guard.create_checkpoint(working_dir)
        if custom_instructions:
            instructions_block = "\n".join(
                f"- {inst}" for inst in custom_instructions
            )
            final_prompt = (
                f"[User instructions: {instructions_block}]\n\n{final_prompt}"
            )

        # Enrich prompt with mem0 semantic memory
        final_prompt = await self._enrich_with_memory(final_prompt, context)

        # Register this task for interrupt/redirect support
        current_task = asyncio.current_task()
        if current_task:
            self._active_tasks[user_id] = _ActiveTask(
                task=current_task,
                original_prompt=message_text,
                started_at=start_time,
            )

        success = True
        try:
            claude_response = await claude_integration.run_command(
                prompt=final_prompt,
                working_directory=working_dir,
                user_id=user_id,
                session_id=session_id,
                on_stream=on_stream,
                force_new=force_new,
            )

            # New session created successfully — clear the one-shot flag
            if force_new:
                context.user_data["force_new_session"] = False

            context.user_data["claude_session_id"] = claude_response.session_id

            # Track directory changes
            from .handlers.message import _update_working_directory_from_claude_response

            _update_working_directory_from_claude_response(
                claude_response, context, self.settings, user_id
            )

            # Store interaction
            storage = context.bot_data.get("storage")
            if storage:
                try:
                    await storage.save_claude_interaction(
                        user_id=user_id,
                        session_id=claude_response.session_id,
                        prompt=message_text,
                        response=claude_response,
                        ip_address=None,
                    )
                except Exception as e:
                    logger.warning("Failed to log interaction", error=str(e))

            if autopilot_plan and autopilot_plan.should_verify and change_guard:
                verification_results = await change_guard.run_verification_commands(
                    working_dir,
                    project_automation.get_verification_commands(autopilot_plan.profile),
                )
                guard_report = ChangeGuardReport(
                    checkpoint_created=checkpoint is not None,
                    checkpoint_id=checkpoint.checkpoint_id if checkpoint else None,
                    verification_results=verification_results,
                )
                failed_result = next(
                    (result for result in verification_results if not result.success),
                    None,
                )
                if failed_result and checkpoint:
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

            # Format response (no reply_markup — strip keyboards)
            from .utils.formatting import ResponseFormatter

            formatter = ResponseFormatter(self.settings)
            formatted_messages = formatter.format_claude_response(
                claude_response.content
            )

        except asyncio.CancelledError:
            # Task was cancelled by user interrupt/redirect
            success = False
            logger.info("Claude task cancelled by user", user_id=user_id)
            heartbeat.cancel()
            self._active_tasks.pop(user_id, None)
            try:
                await progress_msg.delete()
            except Exception:
                pass
            return  # Don't send any response — user already got feedback

        except Exception as e:
            success = False
            logger.error("Claude integration failed", error=str(e), user_id=user_id)
            from .handlers.message import _format_error_message
            from .utils.formatting import FormattedMessage

            if checkpoint and change_guard:
                guard_report = await change_guard.rollback(
                    checkpoint,
                    reason=f"claude error: {type(e).__name__}",
                )

            # Enrich timeout errors with tool context
            error_msg = _format_error_message(e)
            if tool_log and "timeout" in type(e).__name__.lower():
                tool_names = [t.get("name", "?") for t in tool_log[-5:]]
                elapsed = int(time.time() - start_time)
                error_msg += (
                    f"\n\n<b>Context:</b> {elapsed}s elapsed, "
                    f"last tools: {', '.join(tool_names)}"
                )

            formatted_messages = [
                FormattedMessage(error_msg, parse_mode="HTML")
            ]
        finally:
            self._active_tasks.pop(user_id, None)
            heartbeat.cancel()
            if draft_streamer:
                try:
                    await draft_streamer.flush()
                except Exception:
                    logger.debug("Draft flush failed in finally block", user_id=user_id)

        try:
            await progress_msg.delete()
        except Exception:
            logger.debug("Failed to delete progress message, ignoring")

        # Use MCP-collected images (from send_image_to_user tool calls)
        images: List[ImageAttachment] = mcp_images
        panel_markup = self._build_agentic_context_markup(context)

        # Try to combine text + images in one message when possible
        caption_sent = False
        if images and len(formatted_messages) == 1:
            msg = formatted_messages[0]
            if msg.text and len(msg.text) <= 1024:
                try:
                    caption_sent = await self._send_images(
                        update,
                        images,
                        reply_to_message_id=update.message.message_id,
                        caption=msg.text,
                        caption_parse_mode=msg.parse_mode,
                    )
                except Exception as img_err:
                    logger.warning("Image+caption send failed", error=str(img_err))

        # Send text messages (skip if caption was already embedded in photos)
        sent_any_text = False
        if not caption_sent:
            visible_messages = [
                message
                for message in formatted_messages
                if message.text and message.text.strip()
            ]
            for i, message in enumerate(formatted_messages):
                if not message.text or not message.text.strip():
                    continue
                reply_id = update.message.message_id if i == 0 else None
                sent_any_text = True
                is_last_visible = (
                    bool(visible_messages)
                    and message is visible_messages[-1]
                    and not (guard_report and change_guard)
                )
                sent = await self._send_with_retry(
                    update,
                    message.text,
                    message.parse_mode,
                    reply_id,
                    reply_markup=panel_markup if is_last_visible else None,
                )
                if not sent:
                    # Last resort: send truncated plain text
                    try:
                        await update.message.reply_text(
                            message.text[:4000],
                            reply_to_message_id=reply_id,
                            reply_markup=panel_markup if is_last_visible else None,
                        )
                    except Exception:
                        pass
                if i < len(formatted_messages) - 1:
                    await asyncio.sleep(0.5)

            # Send images separately if caption wasn't used
            if images:
                try:
                    await self._send_images(
                        update,
                        images,
                        reply_to_message_id=update.message.message_id,
                    )
                except Exception as img_err:
                    logger.warning("Image send failed", error=str(img_err))

        if guard_report and change_guard:
            await update.message.reply_text(
                change_guard.format_report_html(guard_report),
                parse_mode="HTML",
                reply_to_message_id=update.message.message_id,
                reply_markup=panel_markup,
            )
        elif caption_sent and not sent_any_text:
            await update.message.reply_text(
                "Actions:",
                reply_markup=panel_markup,
                reply_to_message_id=update.message.message_id,
            )

        if not context.user_data.get("agentic_reply_keyboard_ready"):
            await update.message.reply_text(
                "⌨️ Quick buttons are now pinned below.",
                reply_markup=self._build_agentic_reply_keyboard(context),
                reply_to_message_id=update.message.message_id,
            )
            self._mark_agentic_reply_keyboard_ready(context)

        # Audit log
        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            if autopilot_plan:
                verification_results = []
                if guard_report:
                    verification_results = [
                        {
                            "command": result.command,
                            "success": result.success,
                            "returncode": result.returncode,
                        }
                        for result in guard_report.verification_results
                    ]

                await audit_logger.log_automation_run(
                    user_id=user_id,
                    request=message_text,
                    workspace_root=str(working_dir),
                    matched_playbook=autopilot_plan.matched_playbook,
                    read_only=autopilot_plan.read_only,
                    success=success,
                    mode="agentic",
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
                    workspace_changed=autopilot_plan.workspace_changed,
                )
            await audit_logger.log_command(
                user_id=user_id,
                command="text_message",
                args=[message_text[:100]],
                success=success,
            )

        # Process queued messages (received while this task was running)
        await self._process_pending_messages(user_id)

    @staticmethod
    async def _send_with_retry(
        update: Update,
        text: str,
        parse_mode: Optional[str],
        reply_to_message_id: Optional[int],
        reply_markup: Optional[Any] = None,
        max_retries: int = 3,
    ) -> bool:
        """Send a message with exponential backoff retry.

        Tries HTML first, then plain text on parse error. Returns True on success.
        """
        for attempt in range(max_retries):
            try:
                pm = parse_mode if attempt == 0 else None
                await update.message.reply_text(
                    text,
                    parse_mode=pm,
                    reply_markup=reply_markup,
                    reply_to_message_id=reply_to_message_id,
                )
                return True
            except Exception as e:
                err_str = str(e).lower()
                # Parse errors won't be fixed by retrying with same parse_mode
                if "parse" in err_str or "can't" in err_str:
                    if parse_mode:
                        parse_mode = None
                        continue
                # Transient errors — backoff and retry
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))
                else:
                    logger.warning(
                        "Failed to send after retries",
                        error=str(e),
                        attempts=max_retries,
                    )
        return False

    async def _process_pending_messages(self, user_id: int) -> None:
        """Process messages that were queued while a task was running."""
        pending = self._pending_messages.pop(user_id, [])
        if not pending:
            return

        logger.info(
            "Processing queued messages",
            user_id=user_id,
            count=len(pending),
        )

        for msg in pending:
            try:
                await self.agentic_text(msg.update, msg.context)
            except Exception as e:
                logger.error(
                    "Failed to process queued message",
                    user_id=user_id,
                    error=str(e),
                )

    async def agentic_document(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Process file upload -> Claude, minimal chrome."""
        user_id = update.effective_user.id
        document = update.message.document

        logger.info(
            "Agentic document upload",
            user_id=user_id,
            filename=document.file_name,
        )

        # Security validation
        security_validator = context.bot_data.get("security_validator")
        if security_validator:
            valid, error = security_validator.validate_filename(document.file_name)
            if not valid:
                await update.message.reply_text(f"File rejected: {error}")
                return

        # Size check
        max_size = 10 * 1024 * 1024
        if document.file_size > max_size:
            await update.message.reply_text(
                f"File too large ({document.file_size / 1024 / 1024:.1f}MB). Max: 10MB."
            )
            return

        chat = update.message.chat
        await chat.send_action("typing")
        progress_msg = await update.message.reply_text("⌛", disable_notification=True)

        # Try enhanced file handler, fall back to basic
        features = context.bot_data.get("features")
        file_handler = features.get_file_handler() if features else None
        prompt: Optional[str] = None

        if file_handler:
            try:
                processed_file = await file_handler.handle_document_upload(
                    document,
                    user_id,
                    update.message.caption or "Please review this file:",
                )
                prompt = processed_file.prompt
            except Exception:
                file_handler = None

        if not file_handler:
            file = await document.get_file()
            file_bytes = await file.download_as_bytearray()
            try:
                content = file_bytes.decode("utf-8")
                if len(content) > 50000:
                    content = content[:50000] + "\n... (truncated)"
                caption = update.message.caption or "Please review this file:"
                prompt = (
                    f"{caption}\n\n**File:** `{document.file_name}`\n\n"
                    f"```\n{content}\n```"
                )
            except UnicodeDecodeError:
                await progress_msg.edit_text(
                    "Unsupported file format. Must be text-based (UTF-8)."
                )
                return

        # Process with Claude
        claude_integration = context.bot_data.get("claude_integration")
        if not claude_integration:
            await progress_msg.edit_text(
                "Claude integration not available. Check configuration."
            )
            return

        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        session_id = context.user_data.get("claude_session_id")

        # Check if /new was used — skip auto-resume for this first message.
        # Flag is only cleared after a successful run so retries keep the intent.
        force_new = bool(context.user_data.get("force_new_session"))

        verbose_level = self._get_verbose_level(context)
        tool_log: List[Dict[str, Any]] = []
        mcp_images_doc: List[ImageAttachment] = []
        on_stream = self._make_stream_callback(
            verbose_level,
            progress_msg,
            tool_log,
            time.time(),
            mcp_images=mcp_images_doc,
            approved_directory=self.settings.approved_directory,
        )

        heartbeat = self._start_typing_heartbeat(chat)
        try:
            claude_response = await claude_integration.run_command(
                prompt=prompt,
                working_directory=current_dir,
                user_id=user_id,
                session_id=session_id,
                on_stream=on_stream,
                force_new=force_new,
            )

            if force_new:
                context.user_data["force_new_session"] = False

            context.user_data["claude_session_id"] = claude_response.session_id

            from .handlers.message import _update_working_directory_from_claude_response

            _update_working_directory_from_claude_response(
                claude_response, context, self.settings, user_id
            )

            from .utils.formatting import ResponseFormatter

            formatter = ResponseFormatter(self.settings)
            formatted_messages = formatter.format_claude_response(
                claude_response.content
            )

            try:
                await progress_msg.delete()
            except Exception:
                logger.debug("Failed to delete progress message, ignoring")

            # Use MCP-collected images (from send_image_to_user tool calls)
            images: List[ImageAttachment] = mcp_images_doc

            caption_sent = False
            if images and len(formatted_messages) == 1:
                msg = formatted_messages[0]
                if msg.text and len(msg.text) <= 1024:
                    try:
                        caption_sent = await self._send_images(
                            update,
                            images,
                            reply_to_message_id=update.message.message_id,
                            caption=msg.text,
                            caption_parse_mode=msg.parse_mode,
                        )
                    except Exception as img_err:
                        logger.warning("Image+caption send failed", error=str(img_err))

            if not caption_sent:
                for i, message in enumerate(formatted_messages):
                    await update.message.reply_text(
                        message.text,
                        parse_mode=message.parse_mode,
                        reply_markup=None,
                        reply_to_message_id=(
                            update.message.message_id if i == 0 else None
                        ),
                    )
                    if i < len(formatted_messages) - 1:
                        await asyncio.sleep(0.5)

                if images:
                    try:
                        await self._send_images(
                            update,
                            images,
                            reply_to_message_id=update.message.message_id,
                        )
                    except Exception as img_err:
                        logger.warning("Image send failed", error=str(img_err))

        except Exception as e:
            from .handlers.message import _format_error_message

            await progress_msg.edit_text(_format_error_message(e), parse_mode="HTML")
            logger.error("Claude file processing failed", error=str(e), user_id=user_id)
        finally:
            heartbeat.cancel()

    async def agentic_photo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Process photo -> Claude, minimal chrome."""
        user_id = update.effective_user.id

        features = context.bot_data.get("features")
        image_handler = features.get_image_handler() if features else None

        if not image_handler:
            await update.message.reply_text("Photo processing is not available.")
            return

        chat = update.message.chat
        await chat.send_action("typing")
        progress_msg = await update.message.reply_text("⌛", disable_notification=True)

        try:
            photo = update.message.photo[-1]
            processed_image = await image_handler.process_image(
                photo, update.message.caption
            )

            # Build image content block for Claude multimodal
            media_type = {
                "png": "image/png",
                "jpeg": "image/jpeg",
                "gif": "image/gif",
                "webp": "image/webp",
            }.get(processed_image.metadata.get("format", ""), "image/jpeg")

            image_blocks = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": processed_image.base64_data,
                    },
                }
            ]

            await self._handle_agentic_media_message(
                update=update,
                context=context,
                prompt=processed_image.prompt,
                progress_msg=progress_msg,
                user_id=user_id,
                chat=chat,
                images=image_blocks,
            )

        except Exception as e:
            from .handlers.message import _format_error_message

            await progress_msg.edit_text(_format_error_message(e), parse_mode="HTML")
            logger.error(
                "Claude photo processing failed", error=str(e), user_id=user_id
            )

    async def agentic_voice(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Transcribe voice message -> Claude, minimal chrome."""
        user_id = update.effective_user.id

        features = context.bot_data.get("features")
        voice_handler = features.get_voice_handler() if features else None

        if not voice_handler:
            await update.message.reply_text(self._voice_unavailable_message())
            return

        chat = update.message.chat
        await chat.send_action("typing")
        progress_msg = await update.message.reply_text("⌛")

        try:
            voice = update.message.voice
            processed_voice = await voice_handler.process_voice_message(
                voice, update.message.caption
            )
            await self._handle_agentic_media_message(
                update=update,
                context=context,
                prompt=processed_voice.prompt,
                progress_msg=progress_msg,
                user_id=user_id,
                chat=chat,
            )

        except Exception as e:
            from .handlers.message import _format_error_message

            await progress_msg.edit_text(_format_error_message(e), parse_mode="HTML")
            logger.error(
                "Claude voice processing failed", error=str(e), user_id=user_id
            )

    async def agentic_video_note(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Process video note (circle) or video -> extract frames -> Claude."""
        from .features.video_note_handler import extract_frames_from_video

        user_id = update.effective_user.id
        chat = update.message.chat
        await chat.send_action("typing")
        progress_msg = await update.message.reply_text("⌛", disable_notification=True)

        try:
            video = update.message.video_note or update.message.video
            if not video:
                await progress_msg.edit_text("No video found in message.")
                return

            extracted = await extract_frames_from_video(
                video, update.message.caption
            )

            # Build image content blocks for Claude multimodal
            image_blocks = []
            for frame_b64 in extracted.frames_base64:
                image_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": frame_b64,
                    },
                })

            prompt = (
                extracted.prompt
                + "\n\nДетально проанализируй все кадры из видео. "
                "Определи и назови точные бренды и модели всей техники "
                "(ноутбуки, телефоны, наушники, зарядки, павербанки — "
                "по форме корпуса, портам, логотипам, цвету, размеру). "
                "Определи названия книг, приложений на экранах, надписи на предметах. "
                "Опиши людей, обстановку, действия. "
                "Объедини наблюдения в единое связное описание."
            )

            await self._handle_agentic_media_message(
                update=update,
                context=context,
                prompt=prompt,
                progress_msg=progress_msg,
                user_id=user_id,
                chat=chat,
                images=image_blocks,
            )

        except Exception as e:
            from .handlers.message import _format_error_message

            await progress_msg.edit_text(_format_error_message(e), parse_mode="HTML")
            logger.error(
                "Video note processing failed", error=str(e), user_id=user_id
            )

    async def _handle_agentic_media_message(
        self,
        *,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        prompt: str,
        progress_msg: Any,
        user_id: int,
        chat: Any,
        images: Optional[List[dict]] = None,
    ) -> None:
        """Run a media-derived prompt through Claude and send responses."""
        claude_integration = context.bot_data.get("claude_integration")
        if not claude_integration:
            await progress_msg.edit_text(
                "Claude integration not available. Check configuration."
            )
            return

        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        session_id = context.user_data.get("claude_session_id")
        force_new = bool(context.user_data.get("force_new_session"))

        verbose_level = self._get_verbose_level(context)
        tool_log: List[Dict[str, Any]] = []
        mcp_images_media: List[ImageAttachment] = []
        on_stream = self._make_stream_callback(
            verbose_level,
            progress_msg,
            tool_log,
            time.time(),
            mcp_images=mcp_images_media,
            approved_directory=self.settings.approved_directory,
        )

        heartbeat = self._start_typing_heartbeat(chat)
        try:
            claude_response = await claude_integration.run_command(
                prompt=prompt,
                working_directory=current_dir,
                user_id=user_id,
                session_id=session_id,
                on_stream=on_stream,
                force_new=force_new,
                images=images,
            )
        finally:
            heartbeat.cancel()

        if force_new:
            context.user_data["force_new_session"] = False

        context.user_data["claude_session_id"] = claude_response.session_id

        from .handlers.message import _update_working_directory_from_claude_response

        _update_working_directory_from_claude_response(
            claude_response, context, self.settings, user_id
        )

        from .utils.formatting import ResponseFormatter

        formatter = ResponseFormatter(self.settings)
        formatted_messages = formatter.format_claude_response(claude_response.content)

        try:
            await progress_msg.delete()
        except Exception:
            logger.debug("Failed to delete progress message, ignoring")

        # Use MCP-collected images (from send_image_to_user tool calls).
        images: List[ImageAttachment] = mcp_images_media

        caption_sent = False
        if images and len(formatted_messages) == 1:
            msg = formatted_messages[0]
            if msg.text and len(msg.text) <= 1024:
                try:
                    caption_sent = await self._send_images(
                        update,
                        images,
                        reply_to_message_id=update.message.message_id,
                        caption=msg.text,
                        caption_parse_mode=msg.parse_mode,
                    )
                except Exception as img_err:
                    logger.warning("Image+caption send failed", error=str(img_err))

        if not caption_sent:
            for i, message in enumerate(formatted_messages):
                if not message.text or not message.text.strip():
                    continue
                await update.message.reply_text(
                    message.text,
                    parse_mode=message.parse_mode,
                    reply_markup=None,
                    reply_to_message_id=(update.message.message_id if i == 0 else None),
                )
                if i < len(formatted_messages) - 1:
                    await asyncio.sleep(0.5)

            if images:
                try:
                    await self._send_images(
                        update,
                        images,
                        reply_to_message_id=update.message.message_id,
                    )
                except Exception as img_err:
                    logger.warning("Image send failed", error=str(img_err))

    def _voice_unavailable_message(self) -> str:
        """Return provider-aware guidance when voice feature is unavailable."""
        return (
            "Voice processing is not available. "
            f"Set {self.settings.voice_provider_api_key_env} "
            f"for {self.settings.voice_provider_display_name} and install "
            'voice extras with: pip install "claude-code-telegram[voice]"'
        )

    async def agentic_repo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """List workspaces or switch to one.

        /repo               — list discovered workspaces
        /repo <name|path>   — switch to that workspace and resume its session
        """
        args = update.message.text.split()[1:] if update.message.text else []
        base = self.settings.approved_directory
        _current_dir, current_workspace, _boundary_root, project_automation, _profile = (
            self._get_agentic_workspace_profile(context)
        )

        if args:
            # Switch to named repo
            target_name = " ".join(args).strip()
            summary = (
                project_automation.resolve_workspace_reference(target_name, base)
                if project_automation
                else None
            )
            target_path = summary.root_path if summary else base / target_name
            if not target_path.is_dir():
                await update.message.reply_text(
                    f"Directory not found: <code>{escape_html(target_name)}</code>",
                    parse_mode="HTML",
                )
                return

            if project_automation:
                target_path = project_automation.detect_workspace_root(target_path, base)
            context.user_data["current_directory"] = target_path

            # Try to find a resumable session
            claude_integration = context.bot_data.get("claude_integration")
            session_id = None
            if claude_integration:
                existing = await claude_integration._find_resumable_session(
                    update.effective_user.id, target_path
                )
                if existing:
                    session_id = existing.session_id
            context.user_data["claude_session_id"] = session_id

            is_git = (target_path / ".git").is_dir()
            git_badge = " (git)" if is_git else ""
            session_badge = " · сессия восстановлена" if session_id else ""
            relative_display = (
                summary.relative_path
                if summary
                else (
                    "/"
                    if target_path == base
                    else str(target_path.relative_to(base)).replace("\\", "/")
                )
            )
            (
                _current_dir,
                _current_workspace,
                _boundary_root,
                _project_automation,
                switched_profile,
            ) = self._get_agentic_workspace_profile(context)

            await update.message.reply_text(
                f"Переключился на <code>{escape_html(relative_display)}</code>"
                f"{git_badge}{session_badge}",
                parse_mode="HTML",
                reply_markup=self._build_agentic_control_panel_markup(switched_profile),
            )
            return

        text, reply_markup = await self._build_agentic_workspace_catalog(context)
        await update.message.reply_text(
            text,
            parse_mode="HTML",
            reply_markup=reply_markup,
        )
        return

    async def _dispatch_agentic_action(
        self,
        action: str,
        query: Any,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle a quick action triggered from inline or reply keyboards."""
        _current_dir, _current_workspace, _boundary_root, _project_automation, profile = (
            self._get_agentic_workspace_profile(context)
        )

        if action == "new":
            context.user_data["claude_session_id"] = None
            context.user_data["session_started"] = True
            context.user_data["force_new_session"] = True
            await query.edit_message_text(
                "Сессия сброшена. Что делаем дальше?",
                reply_markup=self._build_agentic_start_keyboard(),
            )

        elif action == "panel":
            text = await self._build_agentic_panel_text(context, query.from_user.id)
            await query.edit_message_text(
                text,
                parse_mode="HTML",
                reply_markup=self._build_agentic_control_panel_markup(profile),
            )

        elif action == "projects":
            text, reply_markup = await self._build_agentic_workspace_catalog(context)
            await query.edit_message_text(
                text,
                parse_mode="HTML",
                reply_markup=reply_markup,
            )

        elif action == "jobs":
            text, reply_markup = await self._build_agentic_jobs_text(context)
            await query.edit_message_text(
                text,
                parse_mode="HTML",
                reply_markup=reply_markup,
            )

        elif action == "running":
            text, reply_markup = await self._build_agentic_running_services_text(context)
            await query.edit_message_text(
                text,
                parse_mode="HTML",
                reply_markup=reply_markup,
            )

        elif action == "services":
            text, reply_markup = await self._build_agentic_services_text(context)
            await query.edit_message_text(
                text,
                parse_mode="HTML",
                reply_markup=reply_markup,
            )

        elif action == "recent":
            text = await self._build_agentic_recent_text(context, query.from_user.id)
            await query.edit_message_text(
                text,
                parse_mode="HTML",
                reply_markup=self._build_agentic_control_panel_markup(profile),
            )

        elif action == "status":
            text = await self._build_agentic_status_text(context, query.from_user.id)
            await query.edit_message_text(
                text,
                parse_mode="HTML",
                reply_markup=self._build_agentic_control_panel_markup(profile),
            )

        elif action == "stats":
            storage = context.bot_data.get("storage")
            if not storage:
                await query.edit_message_text("Хранилище недоступно.")
                return
            try:
                stats = await storage.analytics.get_user_stats(query.from_user.id)
                summary = stats.get("summary", {})
                total_cost = summary.get("total_cost") or 0.0
                total_msgs = summary.get("total_messages") or 0
                total_sessions = summary.get("total_sessions") or 0
                avg_duration = summary.get("avg_duration") or 0
                avg_s = avg_duration / 1000 if avg_duration else 0
                await query.edit_message_text(
                    f"<b>Статистика:</b> ${total_cost:.4f} | "
                    f"{total_msgs} сообщений | {total_sessions} сессий | "
                    f"среднее {avg_s:.1f}с",
                    parse_mode="HTML",
                    reply_markup=self._build_agentic_control_panel_markup(profile),
                )
            except Exception:
                await query.edit_message_text("Не удалось загрузить статистику.")

        elif action in {"doctor", "review", "setup", "test", "quality"}:
            await self._run_agentic_playbook_action(query, context, action)

        elif action == "verify":
            await self._run_agentic_verify_action(query, context)

        elif action in {"health", "build"}:
            await self._run_agentic_command_action(query, context, action)

        elif action in {"start", "dev", "deploy"}:
            await self._run_agentic_background_action(query, context, action)

        elif action.startswith("svc:"):
            _svc, service_key, service_action = action.split(":", 2)
            await self._run_agentic_service_action(
                query, context, service_key, service_action
            )

        elif action.startswith("stop:"):
            await self._stop_agentic_background_action(
                query, context, action.split(":", 1)[1]
            )

        elif action.startswith("v"):
            level = int(action[1])
            context.user_data["verbose_level"] = level
            labels = {0: "коротко", 1: "нормально", 2: "подробно"}
            await query.edit_message_text(
                f"Режим ответа: <b>{level}</b> ({labels[level]})",
                parse_mode="HTML",
                reply_markup=self._build_agentic_control_panel_markup(profile),
            )

    async def _agentic_quick_action(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle quick action button presses."""
        query = update.callback_query
        await query.answer()
        action = query.data.removeprefix("act:")
        await self._dispatch_agentic_action(action, query, context)

    async def _agentic_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle cd: callbacks — switch directory and resume session if available."""
        query = update.callback_query
        await query.answer()

        data = query.data
        _, project_name = data.split(":", 1)

        base = self.settings.approved_directory
        _current_dir, _current_workspace, _boundary_root, project_automation, _profile = (
            self._get_agentic_workspace_profile(context)
        )

        if project_name == "/":
            new_path = base
        else:
            summary = (
                project_automation.resolve_workspace_reference(project_name, base)
                if project_automation
                else None
            )
            new_path = summary.root_path if summary else base / project_name

        if not new_path.is_dir():
            await query.edit_message_text(
                f"Папка не найдена: <code>{escape_html(project_name)}</code>",
                parse_mode="HTML",
            )
            return

        if project_automation:
            new_path = project_automation.detect_workspace_root(new_path, base)
        context.user_data["current_directory"] = new_path

        # Look for a resumable session instead of always clearing
        claude_integration = context.bot_data.get("claude_integration")
        session_id = None
        if claude_integration:
            existing = await claude_integration._find_resumable_session(
                query.from_user.id, new_path
            )
            if existing:
                session_id = existing.session_id
        context.user_data["claude_session_id"] = session_id

        is_git = (new_path / ".git").is_dir()
        git_badge = " (git)" if is_git else ""
        session_badge = " · сессия восстановлена" if session_id else ""
        relative_display = (
            project_name
            if project_name == "/"
            else self._format_agentic_relative_path(new_path, base)
        )
        (
            _current_dir,
            _current_workspace,
            _boundary_root,
            _project_automation,
            profile,
        ) = self._get_agentic_workspace_profile(context)

        await query.edit_message_text(
            f"Переключился на <code>{escape_html(relative_display)}</code>"
            f"{git_badge}{session_badge}",
            parse_mode="HTML",
            reply_markup=self._build_agentic_control_panel_markup(profile),
        )

        # Audit log
        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=query.from_user.id,
                command="cd",
                args=[project_name],
                success=True,
            )

    async def _enrich_with_memory(
        self,
        prompt: str,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> str:
        """Search mem0 for relevant memories and prepend them to prompt."""
        mem0_client = context.bot_data.get("mem0_client")
        if not mem0_client:
            return prompt

        try:
            from ..memory.mem0_client import format_memories_for_prompt

            results = await mem0_client.search(prompt, limit=5)
            memory_block = format_memories_for_prompt(results, min_score=0.3)
            if memory_block:
                return f"{memory_block}\n\n{prompt}"
        except Exception as e:
            logger.warning("mem0 enrichment failed", error=str(e))

        return prompt

    async def _enrich_with_video_transcript(self, message_text: str) -> str:
        """If message contains a YouTube URL, extract transcript and append it."""
        from .features.video_handler import extract_youtube_id, get_youtube_transcript

        video_id = extract_youtube_id(message_text)
        if not video_id:
            return message_text

        try:
            transcript = await get_youtube_transcript(video_id)
            logger.info(
                "YouTube transcript extracted",
                video_id=video_id,
                language=transcript.language,
                duration=transcript.duration_text,
                transcript_length=len(transcript.transcript),
            )
            return (
                f"{message_text}\n\n"
                f"--- YouTube Video Transcript ({transcript.duration_text}, "
                f"{transcript.language}) ---\n"
                f"{transcript.transcript}"
            )
        except Exception as e:
            logger.warning(
                "Failed to extract YouTube transcript",
                video_id=video_id,
                error=str(e),
            )
            return message_text
