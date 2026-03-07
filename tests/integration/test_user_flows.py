"""Integration tests for key user flows: Status, Verify, Resolve, Service actions.

These tests exercise the full chain from button press through ActionRunner
to real shell execution and response formatting. They verify behavior,
not implementation details.
"""

import asyncio
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.bot.agentic.action_runner import ActionRunner
from src.bot.agentic.context import (
    AgenticWorkspaceContext,
    ResolveResult,
    ShellActionResult,
    VerifyReport,
    VerifyStep,
)
from src.bot.agentic.panel_builder import PanelBuilder
from src.bot.agentic.problem_classifier import (
    ProblemType,
    classify_problem,
    format_resolve_summary,
    format_service_summary,
    format_verify_summary,
)
from src.bot.agentic.resolve_runner import ResolveRunner
from src.bot.agentic.response_sender import ResponseSender
from src.bot.agentic.service_controller import ServiceController, ServiceFollowUpResult
from src.bot.agentic.shell_executor import ShellExecutor
from src.bot.agentic.stream_handler import StreamHandler
from src.bot.agentic.verify_pipeline import VerifyPipeline
from src.bot.orchestrator import MessageOrchestrator
from src.config import create_test_config


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def workspace(tmp_dir):
    """Create a workspace with verify steps that can pass or fail."""
    ws = tmp_dir / "project"
    ws.mkdir()
    (ws / ".git").mkdir()
    # A script that always passes
    pass_script = ws / "check_pass.sh"
    pass_script.write_text("#!/bin/bash\necho 'all good'\nexit 0\n")
    pass_script.chmod(0o755)
    # A script that always fails with a code error
    fail_script = ws / "check_fail.sh"
    fail_script.write_text(
        "#!/bin/bash\necho 'SyntaxError: unexpected EOF'\nexit 1\n"
    )
    fail_script.chmod(0o755)
    # A script that fails with dependency error
    dep_fail = ws / "check_dep.sh"
    dep_fail.write_text(
        "#!/bin/bash\necho 'ModuleNotFoundError: No module named flask'\nexit 1\n"
    )
    dep_fail.chmod(0o755)
    # A script that fails with service error
    svc_fail = ws / "check_svc.sh"
    svc_fail.write_text(
        "#!/bin/bash\necho 'Connection refused on port 8080'\nexit 1\n"
    )
    svc_fail.chmod(0o755)
    return ws


def _make_profile(workspace, commands=None, services=None):
    """Create a mock ProjectProfile with real paths."""
    default_commands = {
        "health": f"bash {workspace / 'check_pass.sh'}",
        "test": f"bash {workspace / 'check_pass.sh'}",
    }
    if commands:
        default_commands.update(commands)
    return SimpleNamespace(
        root_path=workspace,
        display_name="TestProject",
        has_git_repo=True,
        stacks=("python",),
        commands=default_commands,
        services=services or [],
    )


def _make_query():
    """Create a mock Telegram callback query."""
    query = MagicMock()
    query.from_user = MagicMock()
    query.from_user.id = 12345
    query.message = MagicMock()
    query.edit_message_text = AsyncMock()
    query.answer = AsyncMock()

    status_msg = MagicMock()
    status_msg.edit_text = AsyncMock()
    status_msg.delete = AsyncMock()
    query.message.reply_text = AsyncMock(return_value=status_msg)
    return query, status_msg


def _make_context(workspace, profile, boundary_root=None, claude_integration=None):
    """Create a mock Telegram context with workspace."""
    context = MagicMock()
    context.user_data = {
        "current_directory": workspace,
        "claude_session_id": None,
    }
    context.bot_data = {
        "features": SimpleNamespace(
            get_project_automation=lambda: SimpleNamespace(
                build_profile=lambda cd, br: profile,
                list_workspace_summaries=lambda br: [],
                get_verification_commands=lambda p: [],
                build_general_autopilot_prompt=lambda req, p: req,
            ),
            get_workspace_operator=lambda: None,
            get_project_change_guard=lambda: None,
        ),
        "claude_integration": claude_integration or MagicMock(),
        "storage": None,
        "audit_logger": None,
    }
    return context


def _make_runner(approved_directory=None):
    """Create ActionRunner with real shell/verify/services."""
    shell = ShellExecutor()
    verify = VerifyPipeline(shell)
    services = ServiceController(shell)
    resolver = MagicMock()
    panel = PanelBuilder(verify, shell, services)
    settings = MagicMock()
    settings.approved_directory = approved_directory or Path("/tmp")
    settings.enable_project_threads = False
    return ActionRunner(settings, shell, verify, services, resolver, panel)


# =====================================================================
# Verify flow
# =====================================================================


class TestVerifyFlow:
    """End-to-end verify: button press -> shell execution -> summary."""

    async def test_verify_all_pass(self, workspace):
        """When all checks pass, verify shows success summary."""
        profile = _make_profile(workspace)
        runner = _make_runner()
        query, status_msg = _make_query()
        context = _make_context(workspace, profile)

        await runner.run_verify(query, context)

        result_text = status_msg.edit_text.call_args.args[0]
        assert "пройдены" in result_text
        # Verify result persisted
        last = context.user_data["last_verify"]
        assert last["success"] is True

    async def test_verify_failure_shows_diagnosis(self, workspace):
        """When a check fails, verify shows problem type and cause."""
        profile = _make_profile(workspace, commands={
            "health": f"bash {workspace / 'check_pass.sh'}",
            "test": f"bash {workspace / 'check_fail.sh'}",
        })
        runner = _make_runner()
        query, status_msg = _make_query()
        context = _make_context(workspace, profile)

        await runner.run_verify(query, context)

        result_text = status_msg.edit_text.call_args.args[0]
        assert "не так" in result_text or "не прошел" in result_text
        last = context.user_data["last_verify"]
        assert last["success"] is False
        assert last["problem_type"] == "code"
        assert "SyntaxError" in last["short_cause"]

    async def test_verify_dependency_error_classified(self, workspace):
        """Dependency errors are correctly classified."""
        profile = _make_profile(workspace, commands={
            "health": f"bash {workspace / 'check_dep.sh'}",
        })
        runner = _make_runner()
        query, status_msg = _make_query()
        context = _make_context(workspace, profile)

        await runner.run_verify(query, context)

        last = context.user_data["last_verify"]
        assert last["success"] is False
        assert last["problem_type"] == "dependency"

    async def test_verify_service_error_classified(self, workspace):
        """Service errors are correctly classified."""
        profile = _make_profile(workspace, commands={
            "health": f"bash {workspace / 'check_svc.sh'}",
        })
        runner = _make_runner()
        query, status_msg = _make_query()
        context = _make_context(workspace, profile)

        await runner.run_verify(query, context)

        last = context.user_data["last_verify"]
        assert last["success"] is False
        assert last["problem_type"] == "service"

    async def test_verify_suggests_resolve_on_fixable(self, workspace):
        """Code errors suggest using Resolve button."""
        profile = _make_profile(workspace, commands={
            "test": f"bash {workspace / 'check_fail.sh'}",
        })
        runner = _make_runner()
        query, status_msg = _make_query()
        context = _make_context(workspace, profile)

        await runner.run_verify(query, context)

        result_text = status_msg.edit_text.call_args.args[0]
        assert "Разберись" in result_text


# =====================================================================
# Resolve flow
# =====================================================================


class TestResolveFlow:
    """End-to-end resolve: verify -> diagnose -> claude fix -> re-verify."""

    async def test_resolve_all_passing_skips(self, workspace):
        """If everything passes, resolve says no problems found."""
        profile = _make_profile(workspace)
        boundary_root = workspace.parent
        runner = _make_runner(approved_directory=boundary_root)
        query, status_msg = _make_query()
        context = _make_context(workspace, profile, boundary_root=boundary_root)

        await runner.run_resolve(query, context)

        # status_msg.edit_text is called with "Проблем не найдено" on success path
        assert status_msg.edit_text.await_count >= 1
        last_call = status_msg.edit_text.call_args_list[-1]
        result_text = last_call.args[0]
        assert "Проблем не найдено" in result_text

    async def test_resolve_runs_claude_on_failure(self, workspace):
        """On verify failure, resolve calls Claude to fix."""
        profile = _make_profile(workspace, commands={
            "test": f"bash {workspace / 'check_fail.sh'}",
        })
        runner = _make_runner()
        query, status_msg = _make_query()
        context = _make_context(workspace, profile)

        # Mock Claude integration
        mock_claude = AsyncMock()
        mock_response = MagicMock()
        mock_response.session_id = "test-session-123"
        mock_response.content = "Fixed the SyntaxError"
        mock_claude.run_command = AsyncMock(return_value=mock_response)
        context.bot_data["claude_integration"] = mock_claude

        # Mock resolver to return a structured result
        mock_result = ResolveResult(
            initial_failure=VerifyStep(label="тесты", command="test"),
            claude_response=mock_response,
            final_report=VerifyReport(
                results=[],
                failed_step=VerifyStep(label="тесты", command="test"),
                logs_result=None,
            ),
            rollback_report=None,
            success=False,
            attempts=2,
        )
        runner.resolver.run = AsyncMock(return_value=mock_result)

        await runner.run_resolve(query, context)

        # Claude was called via resolver
        runner.resolver.run.assert_awaited_once()
        # Result persisted
        last = context.user_data["last_resolve"]
        assert last["success"] is False
        assert last["attempts"] == 2

    async def test_resolve_success_persists_result(self, workspace):
        """Successful resolve persists success status."""
        profile = _make_profile(workspace, commands={
            "test": f"bash {workspace / 'check_fail.sh'}",
        })
        runner = _make_runner()
        query, status_msg = _make_query()
        context = _make_context(workspace, profile)

        mock_claude = AsyncMock()
        mock_response = MagicMock()
        mock_response.session_id = "sess-ok"
        mock_response.content = "Fixed"
        context.bot_data["claude_integration"] = mock_claude

        mock_result = ResolveResult(
            initial_failure=VerifyStep(label="тесты", command="test"),
            claude_response=mock_response,
            final_report=VerifyReport(results=[], failed_step=None, logs_result=None),
            rollback_report=None,
            success=True,
            attempts=1,
        )
        runner.resolver.run = AsyncMock(return_value=mock_result)

        await runner.run_resolve(query, context)

        last = context.user_data["last_resolve"]
        assert last["success"] is True


# =====================================================================
# Status flow
# =====================================================================


class TestStatusFlow:
    """Status command shows workspace state with history."""

    async def test_status_with_verify_history(self, workspace):
        """Status shows last verify result when available."""
        shell = ShellExecutor()
        verify = VerifyPipeline(shell)
        services = ServiceController(shell)
        panel = PanelBuilder(verify, shell, services)

        profile = _make_profile(workspace)
        ctx = AgenticWorkspaceContext(
            current_directory=workspace,
            current_workspace=workspace,
            boundary_root=workspace.parent,
            project_automation=MagicMock(
                build_profile=lambda cd, br: profile,
                list_workspace_summaries=lambda br: [],
            ),
            profile=profile,
        )

        text = await panel.build_status_text(
            ctx,
            user_id=12345,
            session_id=None,
            last_verify={
                "success": True,
                "steps_total": 3,
                "steps_passed": 3,
                "timestamp": time.time(),
            },
        )
        assert "3/3" in text

    async def test_status_with_failed_verify(self, workspace):
        """Status shows failed verify details."""
        shell = ShellExecutor()
        verify = VerifyPipeline(shell)
        services = ServiceController(shell)
        panel = PanelBuilder(verify, shell, services)

        profile = _make_profile(workspace)
        ctx = AgenticWorkspaceContext(
            current_directory=workspace,
            current_workspace=workspace,
            boundary_root=workspace.parent,
            project_automation=MagicMock(
                build_profile=lambda cd, br: profile,
                list_workspace_summaries=lambda br: [],
            ),
            profile=profile,
        )

        text = await panel.build_status_text(
            ctx,
            user_id=12345,
            session_id="active-session",
            last_verify={
                "success": False,
                "failed_step": "тесты",
                "steps_total": 3,
                "steps_passed": 1,
                "timestamp": time.time(),
            },
            last_resolve={
                "success": False,
                "attempts": 2,
                "rollback": True,
                "timestamp": time.time(),
            },
        )
        assert "тесты" in text
        assert "1/3" in text
        assert "откат" in text.lower() or "2" in text


# =====================================================================
# Service action flow
# =====================================================================


class TestServiceActionFlow:
    """Service actions with follow-up checks."""

    async def test_service_action_success(self, workspace):
        """Successful service action shows clean summary."""
        service = SimpleNamespace(
            key="app",
            display_name="TestApp",
            health_command=None,
            status_command=None,
            logs_command=None,
            command_for=lambda action: f"echo '{action} done'",
        )
        profile = _make_profile(workspace, services=[service])
        runner = _make_runner()
        query, status_msg = _make_query()
        context = _make_context(workspace, profile)

        # Mock service resolution
        runner.services.resolve_service = MagicMock(return_value=service)

        await runner.run_service(query, context, "app", "status")

        result_text = status_msg.edit_text.call_args.args[0]
        assert "TestApp" in result_text
        assert "выполнено" in result_text

    async def test_service_action_failure_shows_cause(self, workspace):
        """Failed service action shows what went wrong."""
        service = SimpleNamespace(
            key="app",
            display_name="TestApp",
            health_command=None,
            status_command=None,
            logs_command=None,
            command_for=lambda action: "false",  # always fails
        )
        profile = _make_profile(workspace, services=[service])
        runner = _make_runner()
        query, status_msg = _make_query()
        context = _make_context(workspace, profile)

        runner.services.resolve_service = MagicMock(return_value=service)

        await runner.run_service(query, context, "app", "status")

        result_text = status_msg.edit_text.call_args.args[0]
        assert "не удалось" in result_text


# =====================================================================
# Problem Classifier
# =====================================================================


class TestProblemClassifier:
    """Problem classification from verify output."""

    def test_classify_code_error(self):
        result = ShellActionResult(
            command="pytest",
            returncode=1,
            success=False,
            timed_out=False,
            stdout_text="FAILED tests/test_foo.py::test_bar\nAssertionError: expected 1",
            stderr_text="",
        )
        step = VerifyStep(label="тесты", command="pytest")
        report = VerifyReport(
            results=[(step, result)], failed_step=step, logs_result=None
        )
        diagnosis = classify_problem(report)
        assert diagnosis.problem_type == ProblemType.CODE
        assert diagnosis.safe_to_autofix is True

    def test_classify_dependency_error(self):
        result = ShellActionResult(
            command="python main.py",
            returncode=1,
            success=False,
            timed_out=False,
            stdout_text="",
            stderr_text="ModuleNotFoundError: No module named 'flask'",
        )
        step = VerifyStep(label="health", command="python main.py")
        report = VerifyReport(
            results=[(step, result)], failed_step=step, logs_result=None
        )
        diagnosis = classify_problem(report)
        assert diagnosis.problem_type == ProblemType.DEPENDENCY
        assert "flask" in diagnosis.short_cause

    def test_classify_service_error(self):
        result = ShellActionResult(
            command="curl localhost:8080",
            returncode=1,
            success=False,
            timed_out=False,
            stdout_text="Connection refused",
            stderr_text="",
        )
        step = VerifyStep(label="health проверка", command="curl localhost:8080")
        report = VerifyReport(
            results=[(step, result)], failed_step=step, logs_result=None
        )
        diagnosis = classify_problem(report)
        assert diagnosis.problem_type == ProblemType.SERVICE

    def test_classify_environment_error(self):
        result = ShellActionResult(
            command="npm install",
            returncode=1,
            success=False,
            timed_out=False,
            stdout_text="",
            stderr_text="ENOSPC: no space left on device",
        )
        step = VerifyStep(label="install", command="npm install")
        report = VerifyReport(
            results=[(step, result)], failed_step=step, logs_result=None
        )
        diagnosis = classify_problem(report)
        assert diagnosis.problem_type == ProblemType.ENVIRONMENT
        assert diagnosis.safe_to_autofix is False

    def test_classify_success_returns_unknown(self):
        report = VerifyReport(results=[], failed_step=None, logs_result=None)
        diagnosis = classify_problem(report)
        assert diagnosis.problem_type == ProblemType.UNKNOWN

    def test_verify_summary_success(self):
        report = VerifyReport(
            results=[
                (VerifyStep(label="lint", command="flake8"), ShellActionResult(
                    command="flake8", returncode=0, success=True,
                    timed_out=False, stdout_text="", stderr_text="",
                )),
                (VerifyStep(label="test", command="pytest"), ShellActionResult(
                    command="pytest", returncode=0, success=True,
                    timed_out=False, stdout_text="5 passed", stderr_text="",
                )),
            ],
            failed_step=None,
            logs_result=None,
        )
        diagnosis = classify_problem(report)
        summary = format_verify_summary(report, diagnosis, "MyProject")
        assert "пройдены" in summary
        assert "2/2" in summary

    def test_verify_summary_failure_suggests_resolve(self):
        step = VerifyStep(label="тесты", command="pytest")
        result = ShellActionResult(
            command="pytest", returncode=1, success=False,
            timed_out=False, stdout_text="FAILED test_foo", stderr_text="",
        )
        report = VerifyReport(
            results=[(step, result)], failed_step=step, logs_result=None
        )
        diagnosis = classify_problem(report)
        summary = format_verify_summary(report, diagnosis, "MyProject")
        assert "Разберись" in summary
        assert "тесты" in summary

    def test_resolve_summary_success(self):
        diagnosis = SimpleNamespace(
            problem_type=ProblemType.CODE,
            label="Ошибка в коде",
            failed_step_label="тесты",
            short_cause="AssertionError",
        )
        summary = format_resolve_summary(
            diagnosis=diagnosis,
            success=True,
            attempts=1,
            rollback=False,
            error=None,
            passed=3,
            total=3,
        )
        assert "Исправлено" in summary
        assert "3/3" in summary

    def test_resolve_summary_rollback(self):
        diagnosis = SimpleNamespace(
            problem_type=ProblemType.CODE,
            label="Ошибка в коде",
            failed_step_label="тесты",
            short_cause="",
        )
        summary = format_resolve_summary(
            diagnosis=diagnosis,
            success=False,
            attempts=2,
            rollback=True,
            error=None,
            passed=1,
            total=3,
        )
        assert "Откат" in summary
        assert "внимания" in summary

    def test_service_summary_success(self):
        result = ShellActionResult(
            command="systemctl restart app",
            returncode=0, success=True, timed_out=False,
            stdout_text="", stderr_text="",
        )
        summary = format_service_summary(
            service_name="MyApp",
            action="restart",
            success=True,
            main_result=result,
            checks_ok=True,
        )
        assert "MyApp" in summary
        assert "выполнено" in summary

    def test_service_summary_failure(self):
        result = ShellActionResult(
            command="systemctl restart app",
            returncode=1, success=False, timed_out=False,
            stdout_text="", stderr_text="",
            error="unit not found",
        )
        summary = format_service_summary(
            service_name="MyApp",
            action="restart",
            success=False,
            main_result=result,
            checks_ok=False,
        )
        assert "не удалось" in summary
        assert "unit not found" in summary


# =====================================================================
# Response Sender
# =====================================================================


class TestResponseSender:
    """Tests for consolidated response delivery."""

    async def test_deliver_single_text(self):
        """Single text message delivered correctly."""
        stream = MagicMock()
        sender = ResponseSender(stream)
        update = MagicMock()
        update.message.message_id = 42
        update.message.reply_text = AsyncMock()

        from src.bot.utils.formatting import FormattedMessage

        messages = [FormattedMessage("Hello world", parse_mode=None)]
        await sender.deliver(update, messages)

        update.message.reply_text.assert_awaited_once()
        call_text = update.message.reply_text.call_args.kwargs.get(
            "text", update.message.reply_text.call_args.args[0]
        )
        assert "Hello world" in str(call_text)

    async def test_deliver_deletes_progress(self):
        """Progress message is deleted before sending response."""
        stream = MagicMock()
        sender = ResponseSender(stream)
        update = MagicMock()
        update.message.message_id = 42
        update.message.reply_text = AsyncMock()
        progress = MagicMock()
        progress.delete = AsyncMock()

        from src.bot.utils.formatting import FormattedMessage

        messages = [FormattedMessage("done", parse_mode=None)]
        await sender.deliver(update, messages, progress_msg=progress)

        progress.delete.assert_awaited_once()

    async def test_deliver_with_guard_report(self):
        """Guard report is sent as separate message."""
        stream = MagicMock()
        sender = ResponseSender(stream)
        update = MagicMock()
        update.message.message_id = 42
        update.message.reply_text = AsyncMock()

        from src.bot.utils.formatting import FormattedMessage

        messages = [FormattedMessage("result", parse_mode="HTML")]
        guard_report = MagicMock()
        change_guard = MagicMock()
        change_guard.format_report_html.return_value = "<b>Guard OK</b>"

        await sender.deliver(
            update, messages,
            guard_report=guard_report,
            change_guard=change_guard,
        )

        # Two calls: response + guard report
        assert update.message.reply_text.await_count == 2
