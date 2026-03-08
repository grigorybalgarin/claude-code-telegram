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
from src.bot.agentic.monitoring import (
    Incident,
    IncidentState,
    WorkspaceHealth,
    WorkspaceMonitor,
)
from src.bot.agentic.panel_builder import PanelBuilder
from src.bot.agentic.problem_classifier import (
    ProblemDiagnosis,
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
    fail_script.write_text("#!/bin/bash\necho 'SyntaxError: unexpected EOF'\nexit 1\n")
    fail_script.chmod(0o755)
    # A script that fails with dependency error
    dep_fail = ws / "check_dep.sh"
    dep_fail.write_text(
        "#!/bin/bash\necho 'ModuleNotFoundError: No module named flask'\nexit 1\n"
    )
    dep_fail.chmod(0o755)
    # A script that fails with service error
    svc_fail = ws / "check_svc.sh"
    svc_fail.write_text("#!/bin/bash\necho 'Connection refused on port 8080'\nexit 1\n")
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
        profile = _make_profile(
            workspace,
            commands={
                "health": f"bash {workspace / 'check_pass.sh'}",
                "test": f"bash {workspace / 'check_fail.sh'}",
            },
        )
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
        profile = _make_profile(
            workspace,
            commands={
                "health": f"bash {workspace / 'check_dep.sh'}",
            },
        )
        runner = _make_runner()
        query, status_msg = _make_query()
        context = _make_context(workspace, profile)

        await runner.run_verify(query, context)

        last = context.user_data["last_verify"]
        assert last["success"] is False
        assert last["problem_type"] == "dependency"

    async def test_verify_service_error_classified(self, workspace):
        """Service errors are correctly classified."""
        profile = _make_profile(
            workspace,
            commands={
                "health": f"bash {workspace / 'check_svc.sh'}",
            },
        )
        runner = _make_runner()
        query, status_msg = _make_query()
        context = _make_context(workspace, profile)

        await runner.run_verify(query, context)

        last = context.user_data["last_verify"]
        assert last["success"] is False
        assert last["problem_type"] == "service"

    async def test_verify_suggests_resolve_on_fixable(self, workspace):
        """Code errors suggest using Resolve button."""
        profile = _make_profile(
            workspace,
            commands={
                "test": f"bash {workspace / 'check_fail.sh'}",
            },
        )
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
        profile = _make_profile(
            workspace,
            commands={
                "test": f"bash {workspace / 'check_fail.sh'}",
            },
        )
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
        profile = _make_profile(
            workspace,
            commands={
                "test": f"bash {workspace / 'check_fail.sh'}",
            },
        )
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
                (
                    VerifyStep(label="lint", command="flake8"),
                    ShellActionResult(
                        command="flake8",
                        returncode=0,
                        success=True,
                        timed_out=False,
                        stdout_text="",
                        stderr_text="",
                    ),
                ),
                (
                    VerifyStep(label="test", command="pytest"),
                    ShellActionResult(
                        command="pytest",
                        returncode=0,
                        success=True,
                        timed_out=False,
                        stdout_text="5 passed",
                        stderr_text="",
                    ),
                ),
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
            command="pytest",
            returncode=1,
            success=False,
            timed_out=False,
            stdout_text="FAILED test_foo",
            stderr_text="",
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
            returncode=0,
            success=True,
            timed_out=False,
            stdout_text="",
            stderr_text="",
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
            returncode=1,
            success=False,
            timed_out=False,
            stdout_text="",
            stderr_text="",
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
            update,
            messages,
            guard_report=guard_report,
            change_guard=change_guard,
        )

        # Two calls: response + guard report
        assert update.message.reply_text.await_count == 2


# ── Deploy Pipeline Tests ──────────────────────────────────────────────


class TestDeployPipeline:
    """Tests for the staged deploy pipeline."""

    async def test_successful_deploy_all_stages(self, workspace):
        """All stages pass -> overall_success is True."""
        from src.bot.agentic.deploy_pipeline import (
            DeployPipeline,
            DeployProfile,
            DeployStage,
        )

        profile = DeployProfile(
            workspace_root=workspace,
            update_command=f"bash {workspace / 'check_pass.sh'}",
            compile_command=f"bash {workspace / 'check_pass.sh'}",
            restart_command=f"bash {workspace / 'check_pass.sh'}",
            health_command=f"bash {workspace / 'check_pass.sh'}",
        )
        pipeline = DeployPipeline(ShellExecutor())
        result = await pipeline.execute(profile)

        assert result.overall_success is True
        assert result.failed_stage is None
        assert result.rollback_performed is False
        non_skipped = [s for s in result.stages if not s.skipped]
        assert len(non_skipped) == 4
        assert all(s.success for s in non_skipped)

    async def test_deploy_fails_on_compile(self, workspace):
        """Deploy stops at failing stage and records it."""
        from src.bot.agentic.deploy_pipeline import (
            DeployPipeline,
            DeployProfile,
            DeployStage,
        )

        profile = DeployProfile(
            workspace_root=workspace,
            update_command=f"bash {workspace / 'check_pass.sh'}",
            compile_command=f"bash {workspace / 'check_fail.sh'}",
            restart_command=f"bash {workspace / 'check_pass.sh'}",
        )
        pipeline = DeployPipeline(ShellExecutor())
        result = await pipeline.execute(profile)

        assert result.overall_success is False
        assert result.failed_stage == DeployStage.COMPILE

    async def test_deploy_skips_missing_stages(self, workspace):
        """Stages with no command are marked as skipped."""
        from src.bot.agentic.deploy_pipeline import (
            DeployPipeline,
            DeployProfile,
        )

        profile = DeployProfile(
            workspace_root=workspace,
            restart_command=f"bash {workspace / 'check_pass.sh'}",
        )
        pipeline = DeployPipeline(ShellExecutor())
        result = await pipeline.execute(profile)

        assert result.overall_success is True
        skipped = [s for s in result.stages if s.skipped]
        assert len(skipped) == 5  # all except restart

    async def test_deploy_format_summary_success(self, workspace):
        """Successful deploy summary includes commit and time."""
        from src.bot.agentic.deploy_pipeline import (
            DeployPipeline,
            DeployProfile,
        )

        profile = DeployProfile(
            workspace_root=workspace,
            restart_command=f"bash {workspace / 'check_pass.sh'}",
        )
        pipeline = DeployPipeline(ShellExecutor())
        result = await pipeline.execute(profile)

        summary = result.format_summary()
        assert "успешно" in summary

    async def test_deploy_format_summary_failure(self, workspace):
        """Failed deploy summary includes failed stage."""
        from src.bot.agentic.deploy_pipeline import (
            DeployPipeline,
            DeployProfile,
        )

        profile = DeployProfile(
            workspace_root=workspace,
            compile_command=f"bash {workspace / 'check_fail.sh'}",
        )
        pipeline = DeployPipeline(ShellExecutor())
        result = await pipeline.execute(profile)

        summary = result.format_summary()
        assert "не удалось" in summary
        assert "Компиляция" in summary

    async def test_deploy_on_stage_callback(self, workspace):
        """on_stage callback is invoked for non-skipped stages."""
        from src.bot.agentic.deploy_pipeline import (
            DeployPipeline,
            DeployProfile,
        )

        profile = DeployProfile(
            workspace_root=workspace,
            restart_command=f"bash {workspace / 'check_pass.sh'}",
            health_command=f"bash {workspace / 'check_pass.sh'}",
        )
        pipeline = DeployPipeline(ShellExecutor())
        stages_seen = []

        async def on_stage(stage, label):
            stages_seen.append(label)

        await pipeline.execute(profile, on_stage=on_stage)
        assert len(stages_seen) == 2

    async def test_deploy_to_dict(self, workspace):
        """DeployResult.to_dict() produces serializable output."""
        from src.bot.agentic.deploy_pipeline import (
            DeployPipeline,
            DeployProfile,
        )

        profile = DeployProfile(
            workspace_root=workspace,
            restart_command=f"bash {workspace / 'check_pass.sh'}",
        )
        pipeline = DeployPipeline(ShellExecutor())
        result = await pipeline.execute(profile)

        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["overall_success"] is True
        assert isinstance(d["stages"], list)
        assert d["correlation_id"]


# ── Classifier Confidence Tests ────────────────────────────────────────


class TestClassifierConfidence:
    """Tests for problem classifier confidence scoring."""

    async def test_high_confidence_for_clear_code_error(self):
        """Clear code error with matching step label gives high confidence."""
        result = ShellActionResult(
            command="flake8 src",
            returncode=1,
            success=False,
            timed_out=False,
            stdout_text="",
            stderr_text="SyntaxError: invalid syntax\nTraceback: ...\nError: compilation failed",
        )
        step = VerifyStep(label="lint", command="flake8 src")
        report = VerifyReport(
            results=[(step, result)], failed_step=step, logs_result=None
        )
        diagnosis = classify_problem(report)
        assert diagnosis.problem_type == ProblemType.CODE
        assert diagnosis.confidence >= 0.6

    async def test_low_confidence_for_ambiguous_error(self):
        """Ambiguous output gives lower confidence."""
        result = ShellActionResult(
            command="./check.sh",
            returncode=1,
            success=False,
            timed_out=False,
            stdout_text="something went wrong",
            stderr_text="",
        )
        step = VerifyStep(label="check", command="./check.sh")
        report = VerifyReport(
            results=[(step, result)], failed_step=step, logs_result=None
        )
        diagnosis = classify_problem(report)
        # Should be UNKNOWN or low confidence
        assert (
            diagnosis.confidence <= 0.5 or diagnosis.problem_type == ProblemType.UNKNOWN
        )

    async def test_confidence_is_in_diagnosis(self):
        """Confidence field exists and is a float between 0 and 1."""
        result = ShellActionResult(
            command="pytest",
            returncode=1,
            success=False,
            timed_out=False,
            stdout_text="ModuleNotFoundError: No module named flask",
            stderr_text="",
        )
        step = VerifyStep(label="test", command="pytest")
        report = VerifyReport(
            results=[(step, result)], failed_step=step, logs_result=None
        )
        diagnosis = classify_problem(report)
        assert 0.0 <= diagnosis.confidence <= 1.0


# ── Persistent Status Fallback Tests ───────────────────────────────────


class TestPersistentStatusFallback:
    """Tests for status text with persistent DB fallback."""

    async def test_status_shows_deploy_info(self, workspace):
        """Status text includes deploy info when last_deploy is provided."""
        panel = PanelBuilder(MagicMock(), MagicMock(), MagicMock())
        ctx = MagicMock()
        ctx.current_workspace = workspace
        ctx.boundary_root = workspace.parent
        ctx.profile = _make_profile(workspace)
        ctx.claude_integration = None
        ctx.operator_runtime = None
        ctx.project_automation = None

        text = await panel.build_status_text(
            ctx,
            user_id=123,
            session_id=None,
            last_deploy={
                "success": True,
                "commit": "abc12345",
                "timestamp": time.time() - 60,
            },
        )
        assert "Деплой" in text
        assert "успешно" in text
        assert "abc12345" in text

    async def test_status_shows_failed_deploy(self, workspace):
        """Status text shows failed deploy with stage info."""
        panel = PanelBuilder(MagicMock(), MagicMock(), MagicMock())
        ctx = MagicMock()
        ctx.current_workspace = workspace
        ctx.boundary_root = workspace.parent
        ctx.profile = _make_profile(workspace)
        ctx.claude_integration = None
        ctx.operator_runtime = None
        ctx.project_automation = None

        text = await panel.build_status_text(
            ctx,
            user_id=123,
            session_id=None,
            last_deploy={
                "success": False,
                "failed_stage": "compile",
                "rollback": True,
                "timestamp": time.time() - 120,
            },
        )
        assert "Деплой" in text
        assert "сбой" in text
        assert "compile" in text
        assert "откат" in text

    async def test_status_without_deploy_no_crash(self, workspace):
        """Status text works fine without deploy data."""
        panel = PanelBuilder(MagicMock(), MagicMock(), MagicMock())
        ctx = MagicMock()
        ctx.current_workspace = workspace
        ctx.boundary_root = workspace.parent
        ctx.profile = _make_profile(workspace)
        ctx.claude_integration = None
        ctx.operator_runtime = None
        ctx.project_automation = None

        text = await panel.build_status_text(
            ctx,
            user_id=123,
            session_id=None,
        )
        assert "Деплой" not in text
        assert "Статус" in text


# ── Operation Config Tests ─────────────────────────────────────────────


class TestOperationConfig:
    """Tests for project-specific operation configuration."""

    def test_operation_config_defaults(self):
        """OperationConfig has sensible defaults."""
        from src.bot.features.project_automation import OperationConfig

        config = OperationConfig()
        assert config.critical_steps == ()
        assert config.diagnose_commands == {}
        assert config.self_heal_restart is False
        assert config.deploy_rollback_safe is False

    def test_operation_config_full(self):
        """OperationConfig accepts all fields."""
        from src.bot.features.project_automation import OperationConfig

        config = OperationConfig(
            critical_steps=("health", "lint"),
            diagnose_commands={"svc": "systemctl is-active app"},
            self_heal_restart=True,
            self_heal_verify_after_restart=True,
            deploy_rollback_safe=True,
        )
        assert "health" in config.critical_steps
        assert config.self_heal_restart is True

    def test_profile_has_operations_field(self):
        """ProjectProfile accepts operations config."""
        from src.bot.features.project_automation import OperationConfig

        profile = _make_profile(Path("/tmp/test"))
        # SimpleNamespace doesn't have operations by default
        assert not hasattr(profile, "operations") or profile.operations is None

    def test_classify_with_critical_step(self):
        """Classifier marks step as critical when in operations config."""
        from src.bot.features.project_automation import OperationConfig

        ops = OperationConfig(critical_steps=("health",))
        result = ShellActionResult(
            command="check",
            returncode=1,
            success=False,
            timed_out=False,
            stdout_text="Connection refused",
            stderr_text="",
        )
        step = VerifyStep(label="health", command="curl localhost:8080")
        report = VerifyReport(
            results=[(step, result)], failed_step=step, logs_result=None
        )
        diagnosis = classify_problem(report, operations_config=ops)
        assert diagnosis.is_critical_step is True

    def test_classify_without_critical_step(self):
        """Non-critical steps are not marked as critical."""
        from src.bot.features.project_automation import OperationConfig

        ops = OperationConfig(critical_steps=("health",))
        result = ShellActionResult(
            command="lint",
            returncode=1,
            success=False,
            timed_out=False,
            stdout_text="flake8 error",
            stderr_text="",
        )
        step = VerifyStep(label="lint", command="flake8 src")
        report = VerifyReport(
            results=[(step, result)], failed_step=step, logs_result=None
        )
        diagnosis = classify_problem(report, operations_config=ops)
        assert diagnosis.is_critical_step is False


# ── Server Diagnostics Tests ──────────────────────────────────────────


class TestServerDiagnostics:
    """Tests for server diagnostics data structure."""

    def test_server_diagnostics_defaults(self):
        from src.bot.agentic.server_diagnostics import ServerDiagnostics

        diag = ServerDiagnostics()
        assert diag.service_active is None
        assert diag.has_service_problem is False
        assert diag.has_disk_problem is False
        assert diag.is_flapping is False
        assert diag.summary_lines() == []
        assert diag.as_prompt_context() == ""

    def test_service_down_detection(self):
        from src.bot.agentic.server_diagnostics import ServerDiagnostics

        diag = ServerDiagnostics(service_active=False, service_state="inactive")
        assert diag.has_service_problem is True
        lines = diag.summary_lines()
        assert any("не активен" in l for l in lines)

    def test_disk_problem_detection(self):
        from src.bot.agentic.server_diagnostics import ServerDiagnostics

        diag = ServerDiagnostics(disk_usage="/dev/sda1  20G  19G  1G  97% /")
        assert diag.has_disk_problem is True

    def test_flapping_detection(self):
        from src.bot.agentic.server_diagnostics import ServerDiagnostics

        diag = ServerDiagnostics(restart_count=5)
        assert diag.is_flapping is True

    def test_no_flapping_with_few_restarts(self):
        from src.bot.agentic.server_diagnostics import ServerDiagnostics

        diag = ServerDiagnostics(restart_count=1)
        assert diag.is_flapping is False

    def test_prompt_context_formatting(self):
        from src.bot.agentic.server_diagnostics import ServerDiagnostics

        diag = ServerDiagnostics(
            service_state="active (running)",
            recent_errors="error: something broke",
        )
        ctx = diag.as_prompt_context()
        assert "Диагностика сервера" in ctx
        assert "something broke" in ctx

    def test_diagnosis_to_dict(self):
        """ProblemDiagnosis.to_dict() includes all fields."""
        result = ShellActionResult(
            command="test",
            returncode=1,
            success=False,
            timed_out=False,
            stdout_text="SyntaxError: bad",
            stderr_text="",
        )
        step = VerifyStep(label="test", command="pytest")
        report = VerifyReport(
            results=[(step, result)], failed_step=step, logs_result=None
        )
        diagnosis = classify_problem(report)
        d = diagnosis.to_dict()
        assert "problem_type" in d
        assert "confidence" in d
        assert "is_critical" in d
        assert isinstance(d["confidence"], float)


# ── Retention/Cleanup Tests ───────────────────────────────────────────


class TestRetentionCleanup:
    """Tests for operations cleanup policy."""

    async def test_cleanup_method_exists(self):
        """OperationsRepository has cleanup_old_operations method."""
        from src.storage.repositories import OperationsRepository

        assert hasattr(OperationsRepository, "cleanup_old_operations")

    async def test_storage_cleanup_includes_operations(self):
        """Storage.cleanup_old_data returns operations_cleaned count."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.storage.facade import Storage

        storage = Storage.__new__(Storage)
        storage.sessions = MagicMock()
        storage.sessions.cleanup_old_sessions = AsyncMock(return_value=5)
        storage.operations = MagicMock()
        storage.operations.cleanup_old_operations = AsyncMock(return_value=10)
        storage.incidents = MagicMock()
        storage.incidents.cleanup_old_incidents = AsyncMock(return_value=3)
        storage.improvements = MagicMock()
        storage.improvements.cleanup_old_improvements = AsyncMock(return_value=2)

        result = await storage.cleanup_old_data(days=7)
        assert result["sessions_cleaned"] == 5
        assert result["operations_cleaned"] == 10
        assert result["incidents_cleaned"] == 3
        assert result["improvements_cleaned"] == 2


# ── Profile Validation Tests ──────────────────────────────────────────


class TestProfileValidation:
    """Tests for enhanced profile validation."""

    def test_validate_self_heal_without_restart(self):
        """Validation catches self_heal_restart without restart command."""
        from src.bot.features.project_automation import (
            OperationConfig,
            ProjectAutomationManager,
        )

        pa = ProjectAutomationManager.__new__(ProjectAutomationManager)
        pa._workspace_overrides = {
            "test_project": {
                "display_name": "Test",
                "aliases": (),
                "operator_notes": "",
                "commands": {},
                "services": (),
                "sort_priority": 0,
                "operations": OperationConfig(self_heal_restart=True),
            }
        }
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "test_project").mkdir()
            warnings = pa.validate_profiles(root)
            errors = [w for w in warnings if w.startswith("[error]")]
            assert any("self_heal_restart" in w for w in errors)

    def test_validate_deploy_rollback_without_deploy_cmd(self):
        """Validation warns about deploy_rollback_safe without deploy command."""
        from src.bot.features.project_automation import (
            OperationConfig,
            ProjectAutomationManager,
        )

        pa = ProjectAutomationManager.__new__(ProjectAutomationManager)
        pa._workspace_overrides = {
            "test_project": {
                "display_name": "Test",
                "aliases": (),
                "operator_notes": "",
                "commands": {"health": "echo ok"},
                "services": (),
                "sort_priority": 0,
                "operations": OperationConfig(deploy_rollback_safe=True),
            }
        }
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "test_project").mkdir()
            warnings = pa.validate_profiles(root)
            assert any("deploy_rollback_safe" in w for w in warnings)


# =============================================================================
# Monitoring, Incident Flow, Notification Policy
# =============================================================================


class TestWorkspaceMonitor:
    """Tests for proactive monitoring, incident lifecycle, and notification policy."""

    def _make_monitor(self, workspace):
        """Create a monitor with real shell and verify pipeline."""
        shell = ShellExecutor()
        verify = VerifyPipeline(shell)
        from src.bot.agentic.server_diagnostics import DiagnosticsCollector

        diag = DiagnosticsCollector(shell)
        return WorkspaceMonitor(
            shell=shell,
            verify=verify,
            diagnostics=diag,
            check_interval_seconds=1.0,
        )

    async def test_healthy_workspace_no_incident(self, workspace):
        """A passing workspace creates no incident."""
        profile = _make_profile(
            workspace,
            commands={
                "health": f"bash {workspace / 'check_pass.sh'}",
            },
        )
        profile.operations = None
        monitor = self._make_monitor(workspace)
        monitor.set_profiles([profile])

        # Manual check (operations=None means _check_workspace skips)
        # Set operations so it proceeds
        from src.bot.features.project_automation import OperationConfig

        profile.operations = OperationConfig(monitoring_interval_seconds=60)

        await monitor._check_workspace(profile)
        health = monitor._health.get(str(workspace))
        assert health is not None
        assert health.healthy is True
        assert health.active_incident is None

    async def test_failing_workspace_creates_incident(self, workspace):
        """A failing workspace creates a DETECTED incident."""
        profile = _make_profile(
            workspace,
            commands={
                "health": f"bash {workspace / 'check_fail.sh'}",
            },
        )
        from src.bot.features.project_automation import OperationConfig

        profile.operations = OperationConfig(monitoring_interval_seconds=60)

        notifications = []

        async def capture_notify(text):
            notifications.append(text)

        monitor = self._make_monitor(workspace)
        monitor.set_profiles([profile])
        monitor.set_notify_callback(capture_notify)

        await monitor._check_workspace(profile)

        health = monitor._health[str(workspace)]
        assert health.healthy is False
        assert health.active_incident is not None
        assert health.active_incident.state in {
            IncidentState.DETECTED,
            IncidentState.ESCALATED,
        }
        # Should have notified
        assert len(notifications) >= 1
        assert "сбой обнаружен" in notifications[0]

    async def test_recovery_clears_incident(self, workspace):
        """When workspace recovers, incident transitions to HEALED."""
        profile = _make_profile(
            workspace,
            commands={
                "health": f"bash {workspace / 'check_fail.sh'}",
            },
        )
        from src.bot.features.project_automation import OperationConfig

        profile.operations = OperationConfig(monitoring_interval_seconds=60)

        notifications = []

        async def capture_notify(text):
            notifications.append(text)

        monitor = self._make_monitor(workspace)
        monitor.set_profiles([profile])
        monitor.set_notify_callback(capture_notify)

        # First check: fails
        await monitor._check_workspace(profile)
        assert monitor._health[str(workspace)].healthy is False

        # Fix the workspace
        profile.commands["health"] = f"bash {workspace / 'check_pass.sh'}"

        # Second check: passes → recovery
        await monitor._check_workspace(profile)
        health = monitor._health[str(workspace)]
        assert health.healthy is True
        assert health.active_incident is None
        # Should have recovery notification
        assert any("восстановлен" in n for n in notifications)

    async def test_no_duplicate_notification_on_repeated_failure(self, workspace):
        """Repeated failures don't re-notify (only first detection does)."""
        profile = _make_profile(
            workspace,
            commands={
                "health": f"bash {workspace / 'check_fail.sh'}",
            },
        )
        from src.bot.features.project_automation import OperationConfig

        profile.operations = OperationConfig(monitoring_interval_seconds=60)

        notifications = []

        async def capture_notify(text):
            notifications.append(text)

        monitor = self._make_monitor(workspace)
        monitor.set_profiles([profile])
        monitor.set_notify_callback(capture_notify)

        # First failure
        await monitor._check_workspace(profile)
        count_after_first = len(notifications)

        # Second failure — should NOT re-send "сбой обнаружен"
        await monitor._check_workspace(profile)
        detection_notifications = [n for n in notifications if "сбой обнаружен" in n]
        assert len(detection_notifications) == 1

    async def test_auto_heal_attempted_when_policy_allows(self, workspace):
        """Auto-heal is attempted if self_heal_restart is enabled."""
        # Create a restart script that "works"
        restart_script = workspace / "restart.sh"
        restart_script.write_text("#!/bin/bash\nexit 0\n")
        restart_script.chmod(0o755)

        health_script = workspace / "health.sh"
        health_script.write_text("#!/bin/bash\nexit 0\n")
        health_script.chmod(0o755)

        # Use check_svc.sh (service error) so it's classified as SERVICE, not CODE
        profile = _make_profile(
            workspace,
            commands={"health": f"bash {workspace / 'check_svc.sh'}"},
            services=[
                SimpleNamespace(
                    key="app",
                    display_name="TestApp",
                    service_type="command",
                    restart_command=f"bash {restart_script}",
                    health_command=f"bash {health_script}",
                    status_command=None,
                    start_command=None,
                    stop_command=None,
                    logs_command=None,
                ),
            ],
        )
        from src.bot.features.project_automation import OperationConfig

        profile.operations = OperationConfig(
            monitoring_interval_seconds=60,
            self_heal_restart=True,
            self_heal_verify_after_restart=True,
        )

        notifications = []
        saved_ops = []

        async def capture_notify(text):
            notifications.append(text)

        async def capture_save(**kwargs):
            saved_ops.append(kwargs)

        monitor = self._make_monitor(workspace)
        monitor.set_profiles([profile])
        monitor.set_notify_callback(capture_notify)
        monitor.set_save_callback(capture_save)

        await monitor._check_workspace(profile)

        # Should have created incident and attempted heal
        health = monitor._health[str(workspace)]
        assert health.active_incident is not None
        assert health.active_incident.heal_attempts >= 1
        # Should have saved incident events
        assert len(saved_ops) >= 1

    async def test_auto_heal_not_attempted_for_code_errors(self, workspace):
        """Code errors should NOT trigger auto-heal even if policy allows."""
        profile = _make_profile(
            workspace,
            commands={
                "health": f"bash {workspace / 'check_fail.sh'}",  # SyntaxError output
            },
        )
        from src.bot.features.project_automation import OperationConfig

        profile.operations = OperationConfig(
            monitoring_interval_seconds=60,
            self_heal_restart=True,
        )

        monitor = self._make_monitor(workspace)
        monitor.set_profiles([profile])

        await monitor._check_workspace(profile)

        health = monitor._health[str(workspace)]
        assert health.active_incident is not None
        # Code errors → escalated immediately, no heal attempt
        assert health.active_incident.heal_attempts == 0

    async def test_incident_serialization(self):
        """Incident.to_dict() includes all fields."""
        diagnosis = ProblemDiagnosis(
            problem_type=ProblemType.SERVICE,
            label="Проблема сервиса",
            failed_step_label="health",
            short_cause="Connection refused",
            safe_to_autofix=False,
        )
        incident = Incident(
            incident_id="abc123",
            workspace_path="/tmp/test",
            state=IncidentState.DETECTED,
            diagnosis=diagnosis,
            detected_at=1000.0,
        )
        d = incident.to_dict()
        assert d["incident_id"] == "abc123"
        assert d["state"] == "detected"
        assert d["problem_type"] == "service"
        assert d["short_cause"] == "Connection refused"


class TestRunbookHints:
    """Tests for runbook/knowledge layer integration."""

    def test_runbook_hint_in_diagnosis(self):
        """Runbook hints from operations config appear in diagnosis."""
        from src.bot.features.project_automation import OperationConfig

        ops = OperationConfig(
            runbook_hints={
                "service": "Check journalctl for details",
                "health": "Run systemctl status first",
            }
        )

        step = VerifyStep(label="health", command="check")
        result = ShellActionResult(
            command="check",
            returncode=1,
            success=False,
            timed_out=False,
            stdout_text="Connection refused on port 8080",
            stderr_text="",
        )
        report = VerifyReport(
            results=[(step, result)],
            failed_step=step,
            logs_result=None,
        )

        diagnosis = classify_problem(report, operations_config=ops)
        # Should match "service" problem type and get the hint
        assert diagnosis.runbook_hint in (
            "Check journalctl for details",
            "Run systemctl status first",
        )

    def test_runbook_hint_in_verify_summary(self):
        """Runbook hints appear in formatted verify summary."""
        diagnosis = ProblemDiagnosis(
            problem_type=ProblemType.SERVICE,
            label="Проблема сервиса",
            failed_step_label="health",
            short_cause="Connection refused",
            safe_to_autofix=False,
            runbook_hint="Проверь journalctl",
        )
        step = VerifyStep(label="health", command="check")
        result = ShellActionResult(
            command="check",
            returncode=1,
            success=False,
            timed_out=False,
            stdout_text="",
            stderr_text="",
        )
        report = VerifyReport(
            results=[(step, result)],
            failed_step=step,
            logs_result=None,
        )

        text = format_verify_summary(report, diagnosis, "/test")
        assert "Подсказка" in text
        assert "Проверь journalctl" in text

    def test_runbook_hint_in_to_dict(self):
        """to_dict() includes runbook hint."""
        diagnosis = ProblemDiagnosis(
            problem_type=ProblemType.CODE,
            label="Ошибка",
            failed_step_label="lint",
            short_cause="SyntaxError",
            safe_to_autofix=True,
            runbook_hint="Run make lint locally",
        )
        d = diagnosis.to_dict()
        assert d["runbook_hint"] == "Run make lint locally"

    def test_operations_config_new_fields(self):
        """OperationConfig parses monitoring_interval and runbook_hints."""
        from src.bot.features.project_automation import OperationConfig

        ops = OperationConfig(
            monitoring_interval_seconds=120,
            runbook_hints={"code": "test locally first"},
        )
        assert ops.monitoring_interval_seconds == 120
        assert ops.runbook_hints == {"code": "test locally first"}

    def test_yaml_parsing_runbook_hints(self):
        """YAML parsing includes runbook_hints and monitoring_interval."""
        from src.bot.features.project_automation import ProjectAutomationManager

        raw = {
            "self_heal_restart": True,
            "monitoring_interval_seconds": 600,
            "runbook_hints": {
                "service": "check logs",
                "code": "run tests",
            },
        }
        ops = ProjectAutomationManager._parse_operations_override(raw)
        assert ops.monitoring_interval_seconds == 600
        assert ops.runbook_hints == {"service": "check logs", "code": "run tests"}
        assert ops.self_heal_restart is True


class TestMonitoringIncidentLifecycle:
    """Tests for the full incident lifecycle: DETECTED → HEALING → HEALED/ESCALATED."""

    async def test_escalation_after_max_heal_attempts(self, workspace):
        """Incident escalates after exhausting heal attempts."""
        profile = _make_profile(
            workspace,
            commands={
                "health": f"bash {workspace / 'check_svc.sh'}",
            },
            services=[
                SimpleNamespace(
                    key="app",
                    display_name="TestApp",
                    service_type="command",
                    restart_command="exit 1",  # restart always fails
                    health_command=f"bash {workspace / 'check_svc.sh'}",
                    status_command=None,
                    start_command=None,
                    stop_command=None,
                    logs_command=None,
                ),
            ],
        )
        from src.bot.features.project_automation import OperationConfig

        profile.operations = OperationConfig(
            monitoring_interval_seconds=60,
            self_heal_restart=True,
        )

        notifications = []

        async def capture_notify(text):
            notifications.append(text)

        shell = ShellExecutor()
        verify = VerifyPipeline(shell)
        from src.bot.agentic.server_diagnostics import DiagnosticsCollector

        monitor = WorkspaceMonitor(
            shell=shell,
            verify=verify,
            diagnostics=DiagnosticsCollector(shell),
            check_interval_seconds=1.0,
            max_heal_attempts=1,
        )
        monitor.set_profiles([profile])
        monitor.set_notify_callback(capture_notify)

        # First check: detect + attempt heal + fail → escalate
        await monitor._check_workspace(profile)

        health = monitor._health[str(workspace)]
        assert health.active_incident is not None
        assert health.active_incident.state == IncidentState.ESCALATED

    async def test_get_active_incidents(self, workspace):
        """get_active_incidents returns only unresolved incidents."""
        profile = _make_profile(
            workspace,
            commands={
                "health": f"bash {workspace / 'check_fail.sh'}",
            },
        )
        from src.bot.features.project_automation import OperationConfig

        profile.operations = OperationConfig(monitoring_interval_seconds=60)

        shell = ShellExecutor()
        verify = VerifyPipeline(shell)
        from src.bot.agentic.server_diagnostics import DiagnosticsCollector

        monitor = WorkspaceMonitor(
            shell=shell,
            verify=verify,
            diagnostics=DiagnosticsCollector(shell),
            check_interval_seconds=1.0,
        )
        monitor.set_profiles([profile])

        await monitor._check_workspace(profile)
        incidents = monitor.get_active_incidents()
        # Code error → escalated (not active)
        assert all(
            i.state in {IncidentState.DETECTED, IncidentState.HEALING}
            for i in incidents
        )


# =============================================================================
# Remediation Planner
# =============================================================================


class TestRemediationPlanner:
    """Tests for runbook-driven remediation planning."""

    def test_code_problem_safe_auto_resolve(self):
        """Code problems are safe to auto-resolve by default."""
        from src.bot.agentic.remediation_planner import (
            CautionLevel,
            build_remediation_plan,
        )

        diag = ProblemDiagnosis(
            problem_type=ProblemType.CODE,
            label="Ошибка в коде",
            failed_step_label="lint",
            short_cause="SyntaxError",
            safe_to_autofix=True,
            confidence=0.8,
        )
        plan = build_remediation_plan(diag)
        assert plan.safe_auto_resolve is True
        assert plan.caution_level == CautionLevel.NONE

    def test_service_problem_high_caution(self):
        """Service problems get high caution by default."""
        from src.bot.agentic.remediation_planner import (
            CautionLevel,
            build_remediation_plan,
        )

        diag = ProblemDiagnosis(
            problem_type=ProblemType.SERVICE,
            label="Проблема сервиса",
            failed_step_label="health",
            short_cause="Connection refused",
            safe_to_autofix=False,
            confidence=0.7,
        )
        plan = build_remediation_plan(diag)
        assert plan.safe_auto_resolve is False
        assert plan.caution_level == CautionLevel.HIGH

    def test_environment_problem_blocks_auto(self):
        """Environment problems block auto-resolve."""
        from src.bot.agentic.remediation_planner import (
            CautionLevel,
            build_remediation_plan,
        )

        diag = ProblemDiagnosis(
            problem_type=ProblemType.ENVIRONMENT,
            label="Проблема окружения",
            failed_step_label="disk",
            short_cause="no space left",
            safe_to_autofix=False,
            confidence=0.9,
        )
        plan = build_remediation_plan(diag)
        assert plan.safe_auto_resolve is False
        assert plan.caution_level == CautionLevel.BLOCK

    def test_runbook_hint_in_framing(self):
        """Runbook hint appears in resolve prompt framing."""
        from src.bot.agentic.remediation_planner import build_remediation_plan

        diag = ProblemDiagnosis(
            problem_type=ProblemType.CODE,
            label="Ошибка в коде",
            failed_step_label="test",
            short_cause="FAILED",
            safe_to_autofix=True,
            confidence=0.8,
        )
        hints = {"code": "Запусти make test локально перед деплоем"}
        plan = build_remediation_plan(diag, runbook_hints=hints)
        assert "make test" in plan.resolve_framing
        assert plan.runbook_note == "Запусти make test локально перед деплоем"

    def test_caution_keyword_raises_level(self):
        """Runbook hints with caution keywords raise the caution level."""
        from src.bot.agentic.remediation_planner import (
            CautionLevel,
            build_remediation_plan,
        )

        diag = ProblemDiagnosis(
            problem_type=ProblemType.CODE,
            label="Ошибка",
            failed_step_label="test",
            short_cause="Error",
            safe_to_autofix=True,
            confidence=0.8,
        )
        hints = {"code": "Осторожно! Проверь вручную перед правкой"}
        plan = build_remediation_plan(diag, runbook_hints=hints)
        assert plan.caution_level.value in ("high", "block")
        assert plan.safe_auto_resolve is False

    def test_low_confidence_raises_caution(self):
        """Low confidence raises caution level."""
        from src.bot.agentic.remediation_planner import (
            CautionLevel,
            build_remediation_plan,
        )

        diag = ProblemDiagnosis(
            problem_type=ProblemType.CODE,
            label="Ошибка",
            failed_step_label="test",
            short_cause="Error",
            safe_to_autofix=True,
            confidence=0.3,
        )
        plan = build_remediation_plan(diag)
        assert plan.caution_level != CautionLevel.NONE
        assert plan.confidence_note != ""

    def test_critical_step_raises_caution(self):
        """Critical step raises caution level from NONE to LOW."""
        from src.bot.agentic.remediation_planner import (
            CautionLevel,
            build_remediation_plan,
        )

        diag = ProblemDiagnosis(
            problem_type=ProblemType.CODE,
            label="Ошибка",
            failed_step_label="health",
            short_cause="Error",
            safe_to_autofix=True,
            confidence=0.8,
            is_critical_step=True,
        )
        plan = build_remediation_plan(diag)
        assert plan.caution_level == CautionLevel.LOW
        assert "критичный" in plan.resolve_framing.lower()

    def test_runbook_hint_from_step_label(self):
        """Runbook hint matched by step label when type doesn't match."""
        from src.bot.agentic.remediation_planner import build_remediation_plan

        diag = ProblemDiagnosis(
            problem_type=ProblemType.UNKNOWN,
            label="Неизвестная проблема",
            failed_step_label="health",
            short_cause="exit 1",
            safe_to_autofix=False,
        )
        hints = {"health": "Запусти systemctl status"}
        plan = build_remediation_plan(diag, runbook_hints=hints)
        assert plan.runbook_note == "Запусти systemctl status"

    def test_safe_keyword_lowers_caution(self):
        """Runbook hints with safe keywords can lower caution."""
        from src.bot.agentic.remediation_planner import (
            CautionLevel,
            build_remediation_plan,
        )

        diag = ProblemDiagnosis(
            problem_type=ProblemType.SERVICE,
            label="Проблема сервиса",
            failed_step_label="health",
            short_cause="inactive",
            safe_to_autofix=False,
            confidence=0.8,
        )
        hints = {"service": "Просто перезапусти: systemctl restart"}
        plan = build_remediation_plan(diag, runbook_hints=hints)
        # HIGH (service default) lowered to MODERATE by safe keyword
        assert plan.caution_level == CautionLevel.MODERATE


# =============================================================================
# Summary Formatter
# =============================================================================


class TestSummaryFormatter:
    """Tests for unified summary formatting."""

    def test_verify_success_short(self):
        """Success summaries are short."""
        from src.bot.agentic.summary_formatter import format_verify_summary

        step = VerifyStep(label="test", command="pytest")
        result = ShellActionResult(
            command="pytest",
            returncode=0,
            success=True,
            timed_out=False,
            stdout_text="5 passed",
            stderr_text="",
        )
        report = VerifyReport(
            results=[(step, result)],
            failed_step=None,
            logs_result=None,
        )
        diag = ProblemDiagnosis(
            problem_type=ProblemType.UNKNOWN,
            label="",
            failed_step_label="",
            short_cause="",
            safe_to_autofix=False,
        )
        text = format_verify_summary(report, diag, "/test")
        assert "Все проверки пройдены" in text
        assert len(text.splitlines()) <= 3

    def test_verify_failure_has_structure(self):
        """Failure summary has all required sections."""
        from src.bot.agentic.summary_formatter import format_verify_summary

        step = VerifyStep(label="lint", command="flake8")
        result = ShellActionResult(
            command="flake8",
            returncode=1,
            success=False,
            timed_out=False,
            stdout_text="E501: line too long",
            stderr_text="",
        )
        report = VerifyReport(
            results=[(step, result)],
            failed_step=step,
            logs_result=None,
        )
        diag = ProblemDiagnosis(
            problem_type=ProblemType.CODE,
            label="Ошибка в коде",
            failed_step_label="lint",
            short_cause="E501: line too long",
            safe_to_autofix=True,
            confidence=0.8,
        )
        text = format_verify_summary(report, diag, "/test")
        assert "Что не так" in text
        assert "Причина" in text
        assert "Прогресс" in text
        assert "Следующий шаг" in text

    def test_verify_failure_with_plan_shows_confidence(self):
        """Low confidence plan adds confidence note."""
        from src.bot.agentic.remediation_planner import build_remediation_plan
        from src.bot.agentic.summary_formatter import format_verify_summary

        step = VerifyStep(label="test", command="pytest")
        result = ShellActionResult(
            command="pytest",
            returncode=1,
            success=False,
            timed_out=False,
            stdout_text="Error",
            stderr_text="",
        )
        report = VerifyReport(
            results=[(step, result)],
            failed_step=step,
            logs_result=None,
        )
        diag = ProblemDiagnosis(
            problem_type=ProblemType.UNKNOWN,
            label="Неизвестная проблема",
            failed_step_label="test",
            short_cause="Error",
            safe_to_autofix=False,
            confidence=0.2,
        )
        plan = build_remediation_plan(diag)
        text = format_verify_summary(report, diag, "/test", plan=plan)
        assert "Точность" in text

    def test_resolve_success_summary(self):
        """Resolve success summary is concise."""
        from src.bot.agentic.summary_formatter import format_resolve_summary

        diag = ProblemDiagnosis(
            problem_type=ProblemType.CODE,
            label="Ошибка в коде",
            failed_step_label="test",
            short_cause="AssertionError",
            safe_to_autofix=True,
        )
        text = format_resolve_summary(diag, True, 1, False, None, 3, 3)
        assert "Исправлено" in text
        assert "Что было" in text
        assert "Результат" in text

    def test_resolve_rollback_summary(self):
        """Resolve rollback summary mentions rollback."""
        from src.bot.agentic.summary_formatter import format_resolve_summary

        diag = ProblemDiagnosis(
            problem_type=ProblemType.CODE,
            label="Ошибка в коде",
            failed_step_label="test",
            short_cause="Error",
            safe_to_autofix=True,
        )
        text = format_resolve_summary(diag, False, 2, True, None, 1, 3)
        assert "Откат" in text
        assert "Осталось" in text

    def test_service_success_short(self):
        """Service success summary is short."""
        from src.bot.agentic.summary_formatter import format_service_summary

        result = ShellActionResult(
            command="systemctl restart",
            returncode=0,
            success=True,
            timed_out=False,
            stdout_text="",
            stderr_text="",
        )
        text = format_service_summary("TestApp", "restart", True, result, True)
        assert "выполнено" in text
        assert len(text.splitlines()) <= 2

    def test_service_failure_informative(self):
        """Service failure summary explains what went wrong."""
        from src.bot.agentic.summary_formatter import format_service_summary

        result = ShellActionResult(
            command="systemctl restart",
            returncode=1,
            success=False,
            timed_out=False,
            stdout_text="",
            stderr_text="",
            error="permission denied",
        )
        text = format_service_summary("TestApp", "restart", False, result, False)
        assert "не удалось" in text
        assert "permission denied" in text

    def test_diagnosis_summary_with_plan(self):
        """Diagnosis summary includes caution from plan."""
        from src.bot.agentic.remediation_planner import build_remediation_plan
        from src.bot.agentic.summary_formatter import format_diagnosis_summary

        diag = ProblemDiagnosis(
            problem_type=ProblemType.ENVIRONMENT,
            label="Проблема окружения",
            failed_step_label="disk",
            short_cause="no space left",
            safe_to_autofix=False,
            confidence=0.9,
        )
        plan = build_remediation_plan(diag)
        text = format_diagnosis_summary(diag, plan)
        assert "осторожность" in text.lower()


# =============================================================================
# Enhanced Profile Validation
# =============================================================================


class TestEnhancedProfileValidation:
    """Tests for stronger policy/profile validation."""

    def _make_pa(self, overrides):
        from src.bot.features.project_automation import ProjectAutomationManager

        pa = ProjectAutomationManager.__new__(ProjectAutomationManager)
        pa._workspace_overrides = overrides
        return pa

    def test_monitoring_without_verify_path(self):
        """Warns when monitoring enabled but no verify/health commands."""
        from src.bot.features.project_automation import OperationConfig

        pa = self._make_pa(
            {
                "test_project": {
                    "display_name": "Test",
                    "aliases": (),
                    "operator_notes": "",
                    "commands": {},
                    "services": (),
                    "sort_priority": 0,
                    "operations": OperationConfig(monitoring_interval_seconds=300),
                }
            }
        )
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "test_project").mkdir()
            warnings = pa.validate_profiles(root)
            assert any("мониторинг" in w for w in warnings)

    def test_critical_step_unknown_command(self):
        """Warns when critical step doesn't match any command."""
        from src.bot.features.project_automation import OperationConfig

        pa = self._make_pa(
            {
                "test_project": {
                    "display_name": "Test",
                    "aliases": (),
                    "operator_notes": "",
                    "commands": {"test": "pytest", "lint": "flake8"},
                    "services": (),
                    "sort_priority": 0,
                    "operations": OperationConfig(
                        critical_steps=("health", "nonexistent"),
                    ),
                }
            }
        )
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "test_project").mkdir()
            warnings = pa.validate_profiles(root)
            assert any("nonexistent" in w for w in warnings)
            assert any("health" in w for w in warnings)

    def test_runbook_invalid_key(self):
        """Warns when runbook key is not a valid problem type or command."""
        from src.bot.features.project_automation import OperationConfig

        pa = self._make_pa(
            {
                "test_project": {
                    "display_name": "Test",
                    "aliases": (),
                    "operator_notes": "",
                    "commands": {"test": "pytest"},
                    "services": (),
                    "sort_priority": 0,
                    "operations": OperationConfig(
                        runbook_hints={"invalid_key": "some hint"},
                    ),
                }
            }
        )
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "test_project").mkdir()
            warnings = pa.validate_profiles(root)
            assert any("invalid_key" in w for w in warnings)

    def test_runbook_empty_hint(self):
        """Warns when runbook hint value is empty."""
        from src.bot.features.project_automation import OperationConfig

        pa = self._make_pa(
            {
                "test_project": {
                    "display_name": "Test",
                    "aliases": (),
                    "operator_notes": "",
                    "commands": {},
                    "services": (),
                    "sort_priority": 0,
                    "operations": OperationConfig(
                        runbook_hints={"code": "  "},
                    ),
                }
            }
        )
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "test_project").mkdir()
            warnings = pa.validate_profiles(root)
            assert any("пуста" in w for w in warnings)

    def test_empty_operations_config_warning(self):
        """Warns about empty operations config."""
        from src.bot.features.project_automation import OperationConfig

        pa = self._make_pa(
            {
                "test_project": {
                    "display_name": "Test",
                    "aliases": (),
                    "operator_notes": "",
                    "commands": {},
                    "services": (),
                    "sort_priority": 0,
                    "operations": OperationConfig(),
                }
            }
        )
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "test_project").mkdir()
            warnings = pa.validate_profiles(root)
            assert any("пустой" in w for w in warnings)

    def test_duplicate_alias_is_error(self):
        """Duplicate aliases across profiles are errors."""
        pa = self._make_pa(
            {
                "project1": {
                    "display_name": "P1",
                    "aliases": ("bot",),
                    "operator_notes": "",
                    "commands": {},
                    "services": (),
                    "sort_priority": 0,
                },
                "project2": {
                    "display_name": "P2",
                    "aliases": ("bot",),
                    "operator_notes": "",
                    "commands": {},
                    "services": (),
                    "sort_priority": 0,
                },
            }
        )
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "project1").mkdir()
            (root / "project2").mkdir()
            warnings = pa.validate_profiles(root)
            errors = [w for w in warnings if w.startswith("[error]")]
            assert any("bot" in w for w in errors)

    def test_verify_after_restart_without_health(self):
        """Warns about verify_after_restart without health command."""
        from src.bot.features.project_automation import OperationConfig

        pa = self._make_pa(
            {
                "test_project": {
                    "display_name": "Test",
                    "aliases": (),
                    "operator_notes": "",
                    "commands": {},
                    "services": (),
                    "sort_priority": 0,
                    "operations": OperationConfig(
                        self_heal_restart=True,
                        self_heal_verify_after_restart=True,
                    ),
                }
            }
        )
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "test_project").mkdir()
            warnings = pa.validate_profiles(root)
            assert any("verify_after_restart" in w for w in warnings)

    def test_valid_runbook_key_no_warning(self):
        """Valid runbook keys (problem types + commands) produce no warnings."""
        from src.bot.features.project_automation import OperationConfig

        pa = self._make_pa(
            {
                "test_project": {
                    "display_name": "Test",
                    "aliases": (),
                    "operator_notes": "",
                    "commands": {"health": "echo ok"},
                    "services": (),
                    "sort_priority": 0,
                    "operations": OperationConfig(
                        runbook_hints={
                            "service": "check logs",
                            "health": "run systemctl status",
                        },
                    ),
                }
            }
        )
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "test_project").mkdir()
            warnings = pa.validate_profiles(root)
            runbook_warns = [w for w in warnings if "runbook" in w]
            assert len(runbook_warns) == 0


# =============================================================================
# Suggested Next Action with Diagnosis
# =============================================================================


class TestSuggestedNextAction:
    """Tests for smarter suggested next action."""

    def test_low_confidence_suggests_manual(self):
        """Low confidence diagnosis suggests manual check instead of resolve."""
        from src.bot.agentic.digest import suggest_next_action
        from src.bot.agentic.ops_model import OperationalSnapshot, SuggestedActionType

        snap = OperationalSnapshot(
            workspace_path="/test",
            display_name="Test",
            last_verify_success=False,
        )
        diag = ProblemDiagnosis(
            problem_type=ProblemType.UNKNOWN,
            label="Неизвестная",
            failed_step_label="test",
            short_cause="",
            safe_to_autofix=False,
            confidence=0.2,
        )
        action = suggest_next_action(snap, diagnosis=diag)
        assert action is not None
        assert action.action_type == SuggestedActionType.MANUAL_CHECK

    def test_service_problem_suggests_manual(self):
        """Service problems suggest manual check."""
        from src.bot.agentic.digest import suggest_next_action
        from src.bot.agentic.ops_model import OperationalSnapshot, SuggestedActionType

        snap = OperationalSnapshot(
            workspace_path="/test",
            display_name="Test",
            last_verify_success=False,
        )
        diag = ProblemDiagnosis(
            problem_type=ProblemType.SERVICE,
            label="Сервис",
            failed_step_label="health",
            short_cause="down",
            safe_to_autofix=False,
            confidence=0.7,
        )
        action = suggest_next_action(snap, diagnosis=diag)
        assert action is not None
        assert action.action_type == SuggestedActionType.MANUAL_CHECK

    def test_code_problem_suggests_resolve(self):
        """Code problems still suggest resolve."""
        from src.bot.agentic.digest import suggest_next_action
        from src.bot.agentic.ops_model import OperationalSnapshot, SuggestedActionType

        snap = OperationalSnapshot(
            workspace_path="/test",
            display_name="Test",
            last_verify_success=False,
        )
        diag = ProblemDiagnosis(
            problem_type=ProblemType.CODE,
            label="Код",
            failed_step_label="lint",
            short_cause="E501",
            safe_to_autofix=True,
            confidence=0.9,
        )
        action = suggest_next_action(snap, diagnosis=diag)
        assert action is not None
        assert action.action_type == SuggestedActionType.RESOLVE

    def test_plan_blocks_auto_resolve(self):
        """Remediation plan blocking auto-resolve suggests manual."""
        from src.bot.agentic.digest import suggest_next_action
        from src.bot.agentic.ops_model import OperationalSnapshot, SuggestedActionType
        from src.bot.agentic.remediation_planner import build_remediation_plan

        snap = OperationalSnapshot(
            workspace_path="/test",
            display_name="Test",
            last_verify_success=False,
        )
        diag = ProblemDiagnosis(
            problem_type=ProblemType.ENVIRONMENT,
            label="Окружение",
            failed_step_label="disk",
            short_cause="no space",
            safe_to_autofix=False,
            confidence=0.9,
        )
        plan = build_remediation_plan(diag)
        action = suggest_next_action(snap, diagnosis=diag, remediation_plan=plan)
        assert action is not None
        assert action.action_type == SuggestedActionType.MANUAL_CHECK

    def test_everything_ok(self):
        """All ok suggests nothing to do."""
        from src.bot.agentic.digest import suggest_next_action
        from src.bot.agentic.ops_model import OperationalSnapshot, SuggestedActionType

        snap = OperationalSnapshot(
            workspace_path="/test",
            display_name="Test",
            last_verify_success=True,
            service_healthy=True,
        )
        action = suggest_next_action(snap)
        assert action is not None
        assert action.action_type == SuggestedActionType.NONE
