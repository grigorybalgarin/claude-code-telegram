"""Main entry point for Claude Code Telegram Bot."""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional, Union

import structlog

from src import __version__
from src.bot.core import ClaudeCodeBot
from src.claude import (
    ClaudeIntegration,
    SessionManager,
)
from src.claude.sdk_integration import ClaudeSDKManager
from src.config.features import FeatureFlags
from src.config.settings import Settings
from src.agents.registry import load_agent_registry
from src.agents.router import AgentRouter
from src.events.bus import EventBus
from src.events.handlers import AgentHandler
from src.events.middleware import EventSecurityMiddleware
from src.exceptions import ConfigurationError
from src.notifications.service import NotificationService
from src.projects import ProjectThreadManager, load_project_registry
from src.scheduler.scheduler import JobScheduler
from src.security.audit import AuditLogger, SQLiteAuditStorage
from src.security.auth import (
    AuthenticationManager,
    SQLiteTokenStorage,
    TokenAuthProvider,
    WhitelistAuthProvider,
)
from src.security.rate_limiter import RateLimiter
from src.security.validators import SecurityValidator
from src.storage.facade import Storage
from src.storage.session_storage import SQLiteSessionStorage
from src.utils.process_lock import SingleInstanceLock
from src.utils.redaction import redact_sensitive_text, redact_sensitive_value


class _RedactingLogFilter(logging.Filter):
    """Best-effort redaction for standard-library log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_sensitive_text(record.msg, max_length=4000)
        if record.args:
            record.args = redact_sensitive_value(record.args, max_string_length=2000)
        return True


def _redact_structlog_event(
    _logger: Any,
    _method_name: str,
    event_dict: MutableMapping[str, Any],
) -> Union[MutableMapping[str, Any], str, bytes]:
    """Redact likely secrets from structured log payloads."""
    return redact_sensitive_value(event_dict, max_string_length=4000)


def setup_logging(debug: bool = False) -> None:
    """Configure structured logging."""
    level = logging.DEBUG if debug else logging.INFO

    # Configure standard logging
    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stdout,
    )

    root_logger = logging.getLogger()
    log_filter = _RedactingLogFilter()
    for handler in root_logger.handlers:
        handler.addFilter(log_filter)

    # Avoid noisy third-party request logs that can leak tokens in URLs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            _redact_structlog_event,
            (
                structlog.processors.JSONRenderer()
                if not debug
                else structlog.dev.ConsoleRenderer()
            ),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Claude Code Telegram Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version", action="version", version=f"Claude Code Telegram Bot {__version__}"
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    parser.add_argument("--config-file", type=Path, help="Path to configuration file")

    return parser.parse_args()


async def create_application(config: Settings) -> Dict[str, Any]:
    """Create and configure the application components."""
    logger = structlog.get_logger()
    logger.info("Creating application components")

    features = FeatureFlags(config)

    # Initialize storage system
    storage = Storage(config.database_url)
    await storage.initialize()

    # Create security components
    providers = []

    # Add whitelist provider if users are configured
    if config.allowed_users:
        providers.append(WhitelistAuthProvider(config.allowed_users))

    # Add token provider if enabled
    if config.enable_token_auth:
        token_storage = SQLiteTokenStorage(storage.db_manager)
        providers.append(TokenAuthProvider(config.auth_secret_str or "", token_storage))

    # Fall back to allowing all users in development mode
    if not providers and config.development_mode:
        logger.warning(
            "No auth providers configured"
            " - creating development-only allow-all provider"
        )
        providers.append(WhitelistAuthProvider([], allow_all_dev=True))
    elif not providers:
        raise ConfigurationError("No authentication providers configured")

    auth_manager = AuthenticationManager(providers)
    security_validator = SecurityValidator(
        config.approved_directory,
        disable_security_patterns=config.disable_security_patterns,
    )
    rate_limiter = RateLimiter(config)

    # Create audit storage and logger
    audit_storage = SQLiteAuditStorage(storage.db_manager)
    audit_logger = AuditLogger(audit_storage)

    # Create Claude integration components with persistent storage
    session_storage = SQLiteSessionStorage(storage.db_manager)
    session_manager = SessionManager(config, session_storage)

    # Create Claude SDK manager and integration facade
    logger.info("Using Claude Python SDK integration")
    sdk_manager = ClaudeSDKManager(config, security_validator=security_validator)

    claude_integration = ClaudeIntegration(
        config=config,
        sdk_manager=sdk_manager,
        session_manager=session_manager,
    )

    # --- Event bus and agentic platform components ---
    event_bus = EventBus()

    # Event security middleware
    event_security = EventSecurityMiddleware(
        event_bus=event_bus,
        security_validator=security_validator,
        auth_manager=auth_manager,
    )
    event_security.register()

    # Agent registry and router (multi-agent system)
    agent_router: Optional[AgentRouter] = None
    agents_config = config.approved_directory / "config" / "agents.yaml"
    if agents_config.exists():
        try:
            agent_registry = load_agent_registry(agents_config)
            agent_router = AgentRouter(agent_registry)
            logger.info(
                "Agent registry loaded",
                agents=len(agent_registry.list_enabled()),
                rules=len(agent_registry.routing_rules),
            )
        except Exception:
            logger.exception("Failed to load agent registry, continuing without routing")

    # Agent handler — translates events into Claude executions
    agent_handler = AgentHandler(
        event_bus=event_bus,
        claude_integration=claude_integration,
        default_working_directory=config.approved_directory,
        default_user_id=config.allowed_users[0] if config.allowed_users else 0,
        agent_router=agent_router,
        project_base_dir=config.approved_directory,
    )
    agent_handler.register()

    # Create mem0 client if enabled
    mem0_client = None
    if config.enable_mem0:
        from .memory.mem0_client import Mem0Client

        mem0_client = Mem0Client(
            base_url=config.mem0_api_url,
            default_user_id=config.mem0_user_id,
        )
        healthy = await mem0_client.health()
        if healthy:
            logger.info("mem0 connected", url=config.mem0_api_url)
        else:
            logger.warning("mem0 not reachable", url=config.mem0_api_url)

    # Create bot with all dependencies
    dependencies = {
        "auth_manager": auth_manager,
        "security_validator": security_validator,
        "rate_limiter": rate_limiter,
        "audit_logger": audit_logger,
        "claude_integration": claude_integration,
        "storage": storage,
        "event_bus": event_bus,
        "mem0_client": mem0_client,
        "project_registry": None,
        "project_threads_manager": None,
    }

    bot = ClaudeCodeBot(config, dependencies)

    # Notification service and scheduler need the bot's Telegram Bot instance,
    # which is only available after bot.initialize(). We store placeholders
    # and wire them up in run_application() after initialization.

    logger.info("Application components created successfully")

    if config.database_path:
        lock_path = config.database_path.parent / "claude-code-telegram.lock"
    else:
        lock_path = config.approved_directory / ".claude-code-telegram.lock"

    return {
        "bot": bot,
        "claude_integration": claude_integration,
        "storage": storage,
        "config": config,
        "features": features,
        "event_bus": event_bus,
        "agent_handler": agent_handler,
        "auth_manager": auth_manager,
        "security_validator": security_validator,
        "instance_lock": SingleInstanceLock(lock_path),
    }


def _kill_orphaned_claude_processes() -> None:
    """Kill any orphaned claude CLI processes from previous bot runs."""
    import os
    import signal as sig

    try:
        import subprocess

        result = subprocess.run(
            ["pgrep", "-f", "^claude"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            my_pid = os.getpid()
            for line in result.stdout.strip().split("\n"):
                pid = int(line.strip())
                if pid != my_pid:
                    try:
                        os.kill(pid, sig.SIGTERM)
                        structlog.get_logger().info(
                            "Killed orphaned claude process", pid=pid
                        )
                    except ProcessLookupError:
                        pass
    except Exception:
        pass  # Best-effort cleanup


def _start_workspace_monitor(
    bot: "ClaudeCodeBot",
    config: Settings,
    storage: "Storage",
    notification_service: Optional[NotificationService],
) -> Optional[Any]:
    """Create workspace monitor if any profile has monitoring enabled."""
    from src.bot.agentic.monitoring import WorkspaceMonitor
    from src.bot.agentic.server_diagnostics import DiagnosticsCollector
    from src.bot.agentic.shell_executor import ShellExecutor
    from src.bot.agentic.verify_pipeline import VerifyPipeline

    log = structlog.get_logger()

    features_reg = bot.deps.get("features")
    pa = (
        getattr(features_reg, "get_project_automation", lambda: None)()
        if features_reg
        else None
    )
    if not pa:
        return None

    boundary_root = Path(config.approved_directory).resolve()
    summaries = pa.list_workspace_summaries(boundary_root)
    profiles_to_monitor = []
    for summary in summaries:
        profile = pa.build_profile(summary.root_path, boundary_root)
        ops = getattr(profile, "operations", None)
        if ops and getattr(ops, "monitoring_interval_seconds", 0) > 0:
            profiles_to_monitor.append(profile)

    if not profiles_to_monitor:
        return None

    min_interval = min(
        getattr(p.operations, "monitoring_interval_seconds", 300)
        for p in profiles_to_monitor
    )

    shell = ShellExecutor()
    verify = VerifyPipeline(shell)
    diag = DiagnosticsCollector(shell)

    monitor = WorkspaceMonitor(
        shell=shell,
        verify=verify,
        diagnostics=diag,
        check_interval_seconds=float(min_interval),
    )
    monitor.set_profiles(profiles_to_monitor)

    # Wire notification callback
    if notification_service:

        async def _notify(text: str) -> None:
            from src.events.types import AgentResponseEvent

            await notification_service.handle_response(
                AgentResponseEvent(chat_id=0, text=text)
            )

        monitor.set_notify_callback(_notify)

    # Wire save callback
    if storage and hasattr(storage, "operations"):

        async def _save(**kwargs: Any) -> None:
            await storage.operations.save(**kwargs)

        monitor.set_save_callback(_save)

    if storage and hasattr(storage, "incidents"):

        async def _save_incident(incident: Any) -> None:
            await storage.incidents.upsert(
                incident_id=incident.incident_id,
                workspace_path=incident.workspace_path,
                state=incident.state.value,
                severity=incident.severity.value,
                dedup_key=incident.dedup_key,
                detected_at=incident.detected_at,
                healed_at=incident.healed_at,
                heal_attempts=incident.heal_attempts,
                suppressed_count=incident.suppressed_count,
                details=incident.to_dict(),
            )

        async def _load_incidents(workspace_paths: list[str]) -> list[dict[str, Any]]:
            return await storage.incidents.list_active(workspace_paths)

        monitor.set_incident_callback(_save_incident)
        monitor.set_active_incidents_loader(_load_incidents)

    log.info(
        "Workspace monitor configured",
        profiles=len(profiles_to_monitor),
        interval=min_interval,
    )
    return monitor


def _start_maintenance_loop(
    storage: "Storage",
    notification_service: Optional[NotificationService],
) -> Optional[Any]:
    """Create the autonomous maintenance loop for self-review."""
    from src.bot.agentic.autonomy import (
        AutonomyTracker,
        ImprovementBacklog,
        MaintenanceLoop,
        SelfReviewEngine,
    )
    from src.bot.agentic.ops_model import AutonomyGuardrails

    guardrails = AutonomyGuardrails()
    review = SelfReviewEngine(guardrails)
    backlog = ImprovementBacklog()
    tracker = AutonomyTracker(guardrails)

    loop = MaintenanceLoop(
        guardrails=guardrails,
        review_engine=review,
        backlog=backlog,
        tracker=tracker,
        review_interval_seconds=3600.0,  # 1 hour
    )

    # Wire ops fetch callback
    if storage and hasattr(storage, "operations"):

        async def _get_ops() -> list:
            return await storage.operations.get_all_recent(limit=100)

        loop.set_ops_callback(_get_ops)

    # Wire save callback
    if storage and hasattr(storage, "operations"):

        async def _save(**kwargs: Any) -> None:
            await storage.operations.save(**kwargs)

        loop.set_save_callback(_save)

    if storage and hasattr(storage, "improvements"):

        async def _save_improvement(candidate: Any) -> None:
            await storage.improvements.upsert(
                improvement_id=candidate.improvement_id,
                improvement_type=candidate.improvement_type.value,
                description=candidate.description,
                category=candidate.category or None,
                confidence=candidate.confidence,
                priority=candidate.priority,
                safe_to_auto_apply=candidate.safe_to_auto_apply,
                status=candidate.status,
                details={
                    "source_incidents": list(candidate.source_incident_ids),
                    "requires_user_approval": candidate.requires_user_approval,
                    "suggested_change": candidate.suggested_change,
                    "created_at": candidate.created_at,
                },
            )

        async def _load_improvements(limit: int) -> list[dict[str, Any]]:
            return await storage.improvements.list_pending(limit=limit)

        loop.set_improvement_save_callback(_save_improvement)
        loop.set_improvement_load_callback(_load_improvements)

    if storage:

        async def _cleanup(days: int) -> dict[str, int]:
            return await storage.cleanup_old_data(days=days)

        loop.set_cleanup_callback(_cleanup)

    # Wire notification
    if notification_service:

        async def _notify(text: str) -> None:
            from src.events.types import AgentResponseEvent

            await notification_service.handle_response(
                AgentResponseEvent(chat_id=0, text=text)
            )

        loop.set_notify_callback(_notify)

    return loop


async def run_application(app: Dict[str, Any]) -> None:
    """Run the application with graceful shutdown handling."""
    logger = structlog.get_logger()
    bot: ClaudeCodeBot = app["bot"]
    claude_integration: ClaudeIntegration = app["claude_integration"]
    storage: Storage = app["storage"]
    config: Settings = app["config"]
    features: FeatureFlags = app["features"]
    event_bus: EventBus = app["event_bus"]
    instance_lock: SingleInstanceLock = app["instance_lock"]

    mem0_client = bot.deps.get("mem0_client")
    notification_service: Optional[NotificationService] = None
    scheduler: Optional[JobScheduler] = None
    project_threads_manager: Optional[ProjectThreadManager] = None
    workspace_monitor: Optional[Any] = None
    maintenance_loop: Optional[Any] = None

    # Set up signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(signum: int, frame: Any) -> None:
        logger.info("Shutdown signal received", signal=signum)
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        logger.info("Starting Claude Code Telegram Bot")

        instance_lock.acquire()

        # Kill any orphaned claude CLI processes from previous runs
        _kill_orphaned_claude_processes()

        # Initialize the bot first (creates the Telegram Application)
        await bot.initialize()

        if config.enable_project_threads:
            if not config.projects_config_path:
                raise ConfigurationError(
                    "Project thread mode enabled but required settings are missing"
                )
            registry = load_project_registry(
                config_path=config.projects_config_path,
                approved_directory=config.approved_directory,
            )
            project_threads_manager = ProjectThreadManager(
                registry=registry,
                repository=storage.project_threads,
                sync_action_interval_seconds=(
                    config.project_threads_sync_action_interval_seconds
                ),
            )

            bot.deps["project_registry"] = registry
            bot.deps["project_threads_manager"] = project_threads_manager

            if config.project_threads_mode == "group":
                if config.project_threads_chat_id is None:
                    raise ConfigurationError(
                        "Group thread mode requires PROJECT_THREADS_CHAT_ID"
                    )
                sync_result = await project_threads_manager.sync_topics(
                    bot.app.bot,
                    chat_id=config.project_threads_chat_id,
                )
                logger.info(
                    "Project thread startup sync complete",
                    mode=config.project_threads_mode,
                    chat_id=config.project_threads_chat_id,
                    created=sync_result.created,
                    reused=sync_result.reused,
                    renamed=sync_result.renamed,
                    failed=sync_result.failed,
                    deactivated=sync_result.deactivated,
                )

        # Now wire up components that need the Telegram Bot instance
        telegram_bot = bot.app.bot

        # Start event bus
        await event_bus.start()

        # Notification service
        notification_service = NotificationService(
            event_bus=event_bus,
            bot=telegram_bot,
            default_chat_ids=config.notification_chat_ids or [],
        )
        notification_service.register()
        await notification_service.start()

        # Collect concurrent tasks
        tasks = []

        # Bot task — use start() which handles its own initialization check
        bot_task = asyncio.create_task(bot.start())
        tasks.append(bot_task)

        # API server (if enabled)
        if features.api_server_enabled:
            from src.api.server import run_api_server

            api_task = asyncio.create_task(
                run_api_server(event_bus, config, storage.db_manager)
            )
            tasks.append(api_task)
            logger.info("API server enabled", port=config.api_server_port)

        # Scheduler (if enabled)
        if features.scheduler_enabled:
            scheduler = JobScheduler(
                event_bus=event_bus,
                db_manager=storage.db_manager,
                default_working_directory=config.approved_directory,
            )
            await scheduler.start()
            logger.info("Job scheduler enabled")

        # Proactive workspace monitor
        workspace_monitor = _start_workspace_monitor(
            bot, config, storage, notification_service
        )
        if workspace_monitor:
            await workspace_monitor.start()

        # Autonomous maintenance loop (self-review, improvement backlog)
        maintenance_loop = _start_maintenance_loop(storage, notification_service)
        if maintenance_loop:
            await maintenance_loop.start()

        # Shutdown task
        shutdown_task = asyncio.create_task(shutdown_event.wait())
        tasks.append(shutdown_task)

        # Wait for any task to complete or shutdown signal
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        # Check completed tasks for exceptions
        for task in done:
            if task.cancelled():
                continue
            exc = task.exception()
            if exc is not None:
                logger.error(
                    "Task failed",
                    task=task.get_name(),
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

        # Cancel remaining tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except Exception as e:
        logger.error("Application error", error=str(e))
        raise
    finally:
        # Ordered shutdown: scheduler -> API -> notification -> bot -> claude -> storage
        logger.info("Shutting down application")

        try:
            if maintenance_loop:
                await maintenance_loop.stop()
            if workspace_monitor:
                await workspace_monitor.stop()
            if scheduler:
                await scheduler.stop()
            if notification_service:
                await notification_service.stop()
            await event_bus.stop()
            await bot.stop()
            await claude_integration.shutdown()
            if mem0_client:
                await mem0_client.close()
            await storage.close()
        except Exception as e:
            logger.error("Error during shutdown", error=str(e))
        finally:
            instance_lock.release()

        logger.info("Application shutdown complete")


async def main() -> None:
    """Main application entry point."""
    args = parse_args()
    setup_logging(debug=args.debug)

    logger = structlog.get_logger()
    logger.info("Starting Claude Code Telegram Bot", version=__version__)

    try:
        # Load configuration
        from src.config import FeatureFlags, load_config

        config = load_config(config_file=args.config_file)
        features = FeatureFlags(config)

        logger.info(
            "Configuration loaded",
            environment="production" if config.is_production else "development",
            enabled_features=features.get_enabled_features(),
            debug=config.debug,
        )

        # Initialize bot and Claude integration
        app = await create_application(config)
        await run_application(app)

    except ConfigurationError as e:
        logger.error("Configuration error", error=str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error", error=str(e))
        sys.exit(1)


def run() -> None:
    """Synchronous entry point for setuptools."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        sys.exit(0)


if __name__ == "__main__":
    run()
