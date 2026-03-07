"""Message orchestrator — single entry point for all Telegram updates.

Routes messages based on agentic vs classic mode. In agentic mode, provides
a minimal conversational interface (3 commands, no inline keyboards). In
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
                BotCommand("start", "Start the bot"),
                BotCommand("new", "Start a fresh session"),
                BotCommand("status", "Show session status"),
                BotCommand("diag", "Show bot and workspace diagnostics"),
                BotCommand("recent", "Show recent prompts and commands"),
                BotCommand("playbooks", "List deterministic playbooks"),
                BotCommand("run", "Run a workspace playbook"),
                BotCommand("verbose", "Set output verbosity (0/1/2)"),
                BotCommand("stats", "Usage analytics & costs"),
                BotCommand("template", "Manage prompt templates"),
                BotCommand("context", "Persistent session instructions"),
                BotCommand("repo", "List repos / switch workspace"),
                BotCommand("restart", "Restart the bot"),
            ]
            if self.settings.enable_project_threads:
                commands.append(BotCommand("sync_threads", "Sync project topics"))
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
        """Brief welcome, no buttons."""
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
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("🔄 New Session", callback_data="act:new"),
                InlineKeyboardButton("📊 Stats", callback_data="act:stats"),
            ],
            [
                InlineKeyboardButton("📂 Status", callback_data="act:status"),
                InlineKeyboardButton("🔇 Quiet", callback_data="act:v0"),
                InlineKeyboardButton("🔉 Normal", callback_data="act:v1"),
                InlineKeyboardButton("🔊 Detail", callback_data="act:v2"),
            ],
        ])
        await update.message.reply_text(
            f"Hi {safe_name}! I'm your AI coding assistant.\n"
            f"Just tell me what you need — commands are optional.\n"
            "Mention a project name when needed and I will route the request automatically.\n"
            "Autopilot is on: I will detect the workspace, checkpoint risky edits, "
            "run final verification, and roll back automatically if verification fails.\n\n"
            f"Working in: {dir_display}"
            f"{sync_line}",
            parse_mode="HTML",
            reply_markup=keyboard,
        )

    async def agentic_new(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Reset session, one-line confirmation."""
        context.user_data["claude_session_id"] = None
        context.user_data["session_started"] = True
        context.user_data["force_new_session"] = True

        await update.message.reply_text("Session reset. What's next?")

    async def agentic_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Compact status with task/queue info."""
        user_id = update.effective_user.id
        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        dir_display = str(current_dir)

        session_id = context.user_data.get("claude_session_id")
        session_status = "active" if session_id else "none"

        # Cost info
        cost_str = ""
        rate_limiter = context.bot_data.get("rate_limiter")
        if rate_limiter:
            try:
                user_status = rate_limiter.get_user_status(user_id)
                cost_usage = user_status.get("cost_usage", {})
                current_cost = cost_usage.get("current", 0.0)
                cost_str = f" · Cost: ${current_cost:.2f}"
            except Exception:
                pass

        # Active task info
        task_str = ""
        active = self._active_tasks.get(user_id)
        if active and not active.task.done():
            elapsed = int(time.time() - active.started_at)
            task_str = f"\n⚙️ Task running ({elapsed}s)"
            queue_size = len(self._pending_messages.get(user_id, []))
            if queue_size:
                task_str += f" · Queued: {queue_size}"

        await update.message.reply_text(
            f"📂 {dir_display} · Session: {session_status}{cost_str}{task_str}"
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
            labels = {0: "quiet", 1: "normal", 2: "detailed"}
            await update.message.reply_text(
                f"Verbosity: <b>{current}</b> ({labels.get(current, '?')})\n\n"
                "Usage: <code>/verbose 0|1|2</code>\n"
                "  0 = quiet (final response only)\n"
                "  1 = normal (tools + reasoning)\n"
                "  2 = detailed (tools with inputs + reasoning)",
                parse_mode="HTML",
            )
            return

        try:
            level = int(args[0])
            if level not in (0, 1, 2):
                raise ValueError
        except ValueError:
            await update.message.reply_text(
                "Please use: /verbose 0, /verbose 1, or /verbose 2"
            )
            return

        context.user_data["verbose_level"] = level
        labels = {0: "quiet", 1: "normal", 2: "detailed"}
        await update.message.reply_text(
            f"Verbosity set to <b>{level}</b> ({labels[level]})",
            parse_mode="HTML",
        )

    async def agentic_stats(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show usage analytics: costs, sessions, top tools."""
        user_id = update.effective_user.id
        storage = context.bot_data.get("storage")
        if not storage:
            await update.message.reply_text("Storage not available.")
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

            cost_chart = "\n".join(cost_lines) if cost_lines else "  No data yet"

            # Top tools
            tool_lines = []
            for t in top_tools[:8]:
                icon = _tool_icon(t["tool_name"])
                tool_lines.append(f"  {icon} {t['tool_name']}: {t['usage_count']}x")
            tools_text = "\n".join(tool_lines) if tool_lines else "  No tools tracked"

            text = (
                f"<b>Your Stats</b>\n\n"
                f"<b>Total:</b> ${total_cost:.4f} | "
                f"{total_msgs} messages | {total_sessions} sessions\n"
                f"<b>Avg response:</b> {avg_duration_s:.1f}s\n\n"
                f"<b>Last 7 days:</b>\n<pre>{cost_chart}</pre>\n\n"
                f"<b>Top tools:</b>\n{tools_text}"
            )
            await update.message.reply_text(text, parse_mode="HTML")
        except Exception as e:
            logger.error("Stats command failed", error=str(e))
            await update.message.reply_text(f"Error loading stats: {e}")

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
        if not caption_sent:
            for i, message in enumerate(formatted_messages):
                if not message.text or not message.text.strip():
                    continue
                reply_id = update.message.message_id if i == 0 else None
                sent = await self._send_with_retry(
                    update, message.text, message.parse_mode, reply_id
                )
                if not sent:
                    # Last resort: send truncated plain text
                    try:
                        await update.message.reply_text(
                            message.text[:4000],
                            reply_to_message_id=reply_id,
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
            )

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
                    reply_markup=None,
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
        current_dir = context.user_data.get("current_directory", base)
        features = context.bot_data.get("features")
        project_automation = (
            getattr(features, "get_project_automation", lambda: None)()
            if features
            else None
        )
        current_workspace = (
            project_automation.detect_workspace_root(current_dir, base)
            if project_automation
            else current_dir
        )
        workspace_summaries = (
            project_automation.list_workspace_summaries(base)
            if project_automation
            else []
        )

        if args:
            # Switch to named repo
            target_name = args[0]
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
            session_badge = " · session resumed" if session_id else ""
            relative_display = (
                summary.relative_path
                if summary
                else (
                    "/"
                    if target_path == base
                    else str(target_path.relative_to(base)).replace("\\", "/")
                )
            )

            await update.message.reply_text(
                f"Switched to <code>{escape_html(relative_display)}</code>"
                f"{git_badge}{session_badge}",
                parse_mode="HTML",
            )
            return

        if project_automation and workspace_summaries:
            lines: List[str] = []
            keyboard_rows: List[list] = []  # type: ignore[type-arg]

            for summary in workspace_summaries:
                lines.extend(
                    project_automation.describe_workspace_summary_lines(
                        summary, current_workspace=current_workspace
                    )
                )

            for i in range(0, len(workspace_summaries), 2):
                row = []
                for j in range(2):
                    if i + j < len(workspace_summaries):
                        summary = workspace_summaries[i + j]
                        row.append(
                            InlineKeyboardButton(
                                summary.button_label,
                                callback_data=f"cd:{summary.relative_path}",
                            )
                        )
                keyboard_rows.append(row)

            reply_markup = InlineKeyboardMarkup(keyboard_rows)
            await update.message.reply_text(
                "<b>Workspaces</b>\n\n"
                + "\n".join(lines)
                + "\n\nAutopilot can route by project name or relative path.",
                parse_mode="HTML",
                reply_markup=reply_markup,
            )
            return

        # No args — fall back to top-level directories if workspace catalog is unavailable
        try:
            entries = sorted(
                [
                    d
                    for d in base.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ],
                key=lambda d: d.name,
            )
        except OSError as e:
            await update.message.reply_text(f"Error reading workspace: {e}")
            return

        if not entries:
            await update.message.reply_text(
                f"No repos in <code>{escape_html(str(base))}</code>.\n"
                'Clone one by telling me, e.g. <i>"clone org/repo"</i>.',
                parse_mode="HTML",
            )
            return

        lines: List[str] = []
        keyboard_rows: List[list] = []  # type: ignore[type-arg]
        current_name = current_dir.name if current_dir != base else None

        for d in entries:
            is_git = (d / ".git").is_dir()
            icon = "\U0001f4e6" if is_git else "\U0001f4c1"
            marker = " \u25c0" if d.name == current_name else ""
            lines.append(f"{icon} <code>{escape_html(d.name)}/</code>{marker}")

        # Build inline keyboard (2 per row)
        for i in range(0, len(entries), 2):
            row = []
            for j in range(2):
                if i + j < len(entries):
                    name = entries[i + j].name
                    row.append(InlineKeyboardButton(name, callback_data=f"cd:{name}"))
            keyboard_rows.append(row)

        reply_markup = InlineKeyboardMarkup(keyboard_rows)

        await update.message.reply_text(
            "<b>Repos</b>\n\n" + "\n".join(lines),
            parse_mode="HTML",
            reply_markup=reply_markup,
        )

    async def _agentic_quick_action(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle quick action button presses."""
        query = update.callback_query
        await query.answer()
        action = query.data.removeprefix("act:")

        if action == "new":
            context.user_data["claude_session_id"] = None
            context.user_data["session_started"] = True
            context.user_data["force_new_session"] = True
            await query.edit_message_text("Session reset. What's next?")

        elif action == "status":
            current_dir = context.user_data.get(
                "current_directory", self.settings.approved_directory
            )
            session_id = context.user_data.get("claude_session_id")
            session_status = "active" if session_id else "none"
            cost_str = ""
            rate_limiter = context.bot_data.get("rate_limiter")
            if rate_limiter:
                try:
                    user_status = rate_limiter.get_user_status(query.from_user.id)
                    cost_usage = user_status.get("cost_usage", {})
                    current_cost = cost_usage.get("current", 0.0)
                    cost_str = f" · Cost: ${current_cost:.2f}"
                except Exception:
                    pass
            await query.edit_message_text(
                f"📂 {current_dir} · Session: {session_status}{cost_str}"
            )

        elif action == "stats":
            storage = context.bot_data.get("storage")
            if not storage:
                await query.edit_message_text("Storage not available.")
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
                    f"<b>Stats:</b> ${total_cost:.4f} | "
                    f"{total_msgs} msgs | {total_sessions} sessions | "
                    f"avg {avg_s:.1f}s",
                    parse_mode="HTML",
                )
            except Exception:
                await query.edit_message_text("Failed to load stats.")

        elif action.startswith("v"):
            level = int(action[1])
            context.user_data["verbose_level"] = level
            labels = {0: "quiet", 1: "normal", 2: "detailed"}
            await query.edit_message_text(
                f"Verbosity: <b>{level}</b> ({labels[level]})",
                parse_mode="HTML",
            )

    async def _agentic_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle cd: callbacks — switch directory and resume session if available."""
        query = update.callback_query
        await query.answer()

        data = query.data
        _, project_name = data.split(":", 1)

        base = self.settings.approved_directory
        new_path = base / project_name

        if not new_path.is_dir():
            await query.edit_message_text(
                f"Directory not found: <code>{escape_html(project_name)}</code>",
                parse_mode="HTML",
            )
            return

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
        session_badge = " · session resumed" if session_id else ""

        await query.edit_message_text(
            f"Switched to <code>{escape_html(project_name)}/</code>"
            f"{git_badge}{session_badge}",
            parse_mode="HTML",
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
