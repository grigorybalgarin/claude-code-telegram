"""Tests for agentic execution modules extracted from orchestrator."""

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.bot.agentic.context import (
    AgenticWorkspaceContext,
    ResolveResult,
    ShellActionResult,
    VerifyReport,
    VerifyStep,
)
from src.bot.agentic.panel_builder import PanelBuilder
from src.bot.agentic.resolve_runner import ResolveRunner
from src.bot.agentic.service_controller import ServiceController, ServiceFollowUpResult
from src.bot.agentic.shell_executor import ShellExecutor
from src.bot.agentic.verify_pipeline import VerifyPipeline


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ── ShellActionResult / VerifyReport ─────────────────────────────────


def test_verify_report_success_when_no_failed_step():
    report = VerifyReport(results=[], failed_step=None, logs_result=None)
    assert report.success is True


def test_verify_report_failure_when_failed_step_set():
    step = VerifyStep(label="test", command="echo fail")
    report = VerifyReport(results=[], failed_step=step, logs_result=None)
    assert report.success is False


# ── ShellExecutor ────────────────────────────────────────────────────


class TestShellExecutor:
    def test_tail_output_short_text(self):
        assert ShellExecutor.tail_output("hello", limit=100) == "hello"

    def test_tail_output_truncates_long_text(self):
        text = "a" * 2000
        result = ShellExecutor.tail_output(text, limit=100)
        assert len(result) == 100
        assert result == "a" * 100

    def test_tail_output_strips_whitespace(self):
        assert ShellExecutor.tail_output("  hello  ") == "hello"

    async def test_execute_success(self, tmp_dir):
        shell = ShellExecutor()
        result = await shell.execute(tmp_dir, "echo hello")
        assert result.success is True
        assert result.returncode == 0
        assert "hello" in result.stdout_text
        assert result.timed_out is False

    async def test_execute_failure(self, tmp_dir):
        shell = ShellExecutor()
        result = await shell.execute(tmp_dir, "exit 42")
        assert result.success is False
        assert result.returncode == 42

    async def test_execute_timeout(self, tmp_dir):
        shell = ShellExecutor()
        result = await shell.execute(tmp_dir, "sleep 60", timeout_seconds=1)
        assert result.success is False
        assert result.timed_out is True

    async def test_execute_captures_stderr(self, tmp_dir):
        shell = ShellExecutor()
        result = await shell.execute(tmp_dir, "echo err >&2")
        assert "err" in result.stderr_text

    def test_format_result_lines_success(self, tmp_dir):
        result = ShellActionResult(
            command="echo ok",
            returncode=0,
            success=True,
            timed_out=False,
            stdout_text="ok",
            stderr_text="",
        )
        lines = ShellExecutor.format_result_lines("Test", tmp_dir, tmp_dir, result)
        joined = "\n".join(lines)
        assert "<code>0</code>" in joined

    def test_format_result_lines_error(self, tmp_dir):
        result = ShellActionResult(
            command="fail",
            returncode=-1,
            success=False,
            timed_out=False,
            stdout_text="",
            stderr_text="",
            error="boom",
        )
        lines = ShellExecutor.format_result_lines("Test", tmp_dir, tmp_dir, result)
        joined = "\n".join(lines)
        assert "boom" in joined

    def test_summarize_short(self):
        result = ShellActionResult(
            command="x", returncode=0, success=True,
            timed_out=False, stdout_text="short output", stderr_text="",
        )
        assert ShellExecutor.summarize(result) == "short output"

    def test_summarize_truncates(self):
        result = ShellActionResult(
            command="x", returncode=0, success=True,
            timed_out=False, stdout_text="a" * 200, stderr_text="",
        )
        summary = ShellExecutor.summarize(result, limit=50)
        assert len(summary) == 50
        assert summary.endswith("...")

    def test_summarize_empty(self):
        result = ShellActionResult(
            command="x", returncode=0, success=True,
            timed_out=False, stdout_text="", stderr_text="",
        )
        assert ShellExecutor.summarize(result) == "\u043d\u0435\u0442 \u0432\u044b\u0432\u043e\u0434\u0430"

    def test_summarize_prefers_error_field(self):
        result = ShellActionResult(
            command="x", returncode=1, success=False,
            timed_out=False, stdout_text="out", stderr_text="",
            error="the error",
        )
        assert "the error" in ShellExecutor.summarize(result)


# ── VerifyPipeline ───────────────────────────────────────────────────


def _make_profile(commands=None, services=None, root_path=None):
    """Create a minimal profile-like object for testing."""
    return SimpleNamespace(
        commands=commands or {},
        services=services or [],
        root_path=root_path or Path("/tmp/test"),
        display_name="test-project",
        stacks=["python"],
        has_git_repo=False,
        operator_notes=None,
    )


class TestVerifyPipeline:
    def test_build_steps_empty_profile(self):
        profile = _make_profile()
        steps = VerifyPipeline.build_steps(profile)
        assert steps == []

    def test_build_steps_health_command(self):
        profile = _make_profile(commands={"health": "curl localhost"})
        steps = VerifyPipeline.build_steps(profile)
        assert len(steps) == 1
        assert steps[0].label == "health"
        assert steps[0].command == "curl localhost"

    def test_build_steps_deduplicates(self):
        profile = _make_profile(commands={
            "health": "check.sh",
            "build": "check.sh",  # same command
        })
        steps = VerifyPipeline.build_steps(profile)
        assert len(steps) == 1

    def test_build_steps_multiple_commands(self):
        profile = _make_profile(commands={
            "health": "curl localhost",
            "lint": "ruff check .",
            "test": "pytest",
            "build": "make build",
        })
        steps = VerifyPipeline.build_steps(profile)
        labels = [s.label for s in steps]
        assert labels == ["health", "\u043b\u0438\u043d\u0442", "\u0442\u0435\u0441\u0442\u044b", "\u0441\u0431\u043e\u0440\u043a\u0430"]

    def test_build_steps_service_fallback(self):
        service = SimpleNamespace(
            health_command="curl :8080/health",
            status_command="systemctl status app",
            logs_command="journalctl -u app",
            display_name="App",
            key="app",
        )
        profile = _make_profile(services=[service])
        steps = VerifyPipeline.build_steps(profile)
        assert len(steps) == 1
        assert steps[0].label == "App \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0430"
        assert steps[0].logs_command == "journalctl -u app"

    async def test_execute_all_pass(self, tmp_dir):
        shell = ShellExecutor()
        pipeline = VerifyPipeline(shell)
        profile = _make_profile(
            commands={"health": "echo ok", "build": "echo built"},
            root_path=tmp_dir,
        )
        report = await pipeline.execute(profile)
        assert report.success is True
        assert len(report.results) == 2
        assert report.failed_step is None

    async def test_execute_stops_on_failure(self, tmp_dir):
        shell = ShellExecutor()
        pipeline = VerifyPipeline(shell)
        profile = _make_profile(
            commands={"health": "exit 1", "build": "echo should-not-run"},
            root_path=tmp_dir,
        )
        report = await pipeline.execute(profile)
        assert report.success is False
        assert report.failed_step.label == "health"
        assert len(report.results) == 1  # stopped after first

    async def test_execute_calls_on_step(self, tmp_dir):
        shell = ShellExecutor()
        pipeline = VerifyPipeline(shell)
        profile = _make_profile(
            commands={"health": "echo ok"},
            root_path=tmp_dir,
        )
        calls = []
        async def on_step(index, total, step):
            calls.append((index, total, step.label))
        await pipeline.execute(profile, on_step=on_step)
        assert calls == [(1, 1, "health")]

    async def test_execute_fetches_logs_on_failure(self, tmp_dir):
        shell = ShellExecutor()
        pipeline = VerifyPipeline(shell)
        service = SimpleNamespace(
            health_command="exit 1",
            status_command=None,
            logs_command="echo log-output",
            display_name="Svc",
            key="svc",
        )
        profile = _make_profile(services=[service], root_path=tmp_dir)
        report = await pipeline.execute(profile)
        assert report.success is False
        assert report.logs_result is not None
        assert "log-output" in report.logs_result.stdout_text

    def test_format_report_success(self, tmp_dir):
        shell = ShellExecutor()
        pipeline = VerifyPipeline(shell)
        step = VerifyStep(label="health", command="echo ok")
        result = ShellActionResult(
            command="echo ok", returncode=0, success=True,
            timed_out=False, stdout_text="ok", stderr_text="",
        )
        report = VerifyReport(results=[(step, result)], failed_step=None, logs_result=None)
        profile = _make_profile(root_path=tmp_dir)
        html = pipeline.format_report(profile, tmp_dir, report)
        assert "\u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 \u0437\u0430\u0432\u0435\u0440\u0448\u0435\u043d\u0430" in html

    def test_format_report_failure(self, tmp_dir):
        shell = ShellExecutor()
        pipeline = VerifyPipeline(shell)
        step = VerifyStep(label="test", command="pytest")
        result = ShellActionResult(
            command="pytest", returncode=1, success=False,
            timed_out=False, stdout_text="FAILED", stderr_text="",
        )
        report = VerifyReport(results=[(step, result)], failed_step=step, logs_result=None)
        profile = _make_profile(root_path=tmp_dir)
        html = pipeline.format_report(profile, tmp_dir, report)
        assert "\u043d\u0435 \u043f\u0440\u043e\u0439\u0434\u0435\u043d\u0430" in html
        assert "test" in html

    def test_select_background_verification_health(self):
        profile = _make_profile(commands={"health": "curl :8080"})
        assert VerifyPipeline.select_background_verification(profile) == "curl :8080"

    def test_select_background_verification_service(self):
        service = SimpleNamespace(
            health_command="curl :3000/health",
            status_command="systemctl status app",
        )
        profile = _make_profile(services=[service])
        assert VerifyPipeline.select_background_verification(profile) == "curl :3000/health"

    def test_select_background_verification_none(self):
        profile = _make_profile()
        assert VerifyPipeline.select_background_verification(profile) is None

    def test_select_primary_service_by_key(self):
        svc1 = SimpleNamespace(key="db", display_name="DB")
        svc2 = SimpleNamespace(key="app", display_name="App")
        profile = _make_profile(services=[svc1, svc2])
        assert VerifyPipeline.select_primary_service(profile).key == "app"

    def test_select_primary_service_single(self):
        svc = SimpleNamespace(key="custom", display_name="Custom")
        profile = _make_profile(services=[svc])
        assert VerifyPipeline.select_primary_service(profile).key == "custom"

    def test_select_primary_service_none(self):
        profile = _make_profile()
        assert VerifyPipeline.select_primary_service(profile) is None


# ── ServiceController ────────────────────────────────────────────────


class TestServiceController:
    def test_resolve_service_found(self):
        svc = SimpleNamespace(key="app")
        profile = _make_profile(services=[svc])
        assert ServiceController.resolve_service(profile, "app") is svc

    def test_resolve_service_not_found(self):
        profile = _make_profile()
        assert ServiceController.resolve_service(profile, "missing") is None

    def test_resolve_service_no_profile(self):
        assert ServiceController.resolve_service(None, "app") is None

    def test_format_action_label_status(self):
        svc = SimpleNamespace(key="app", display_name="MyApp")
        label = ServiceController.format_action_label(svc, "status")
        assert "MyApp" in label

    def test_format_action_label_truncates_long_name(self):
        svc = SimpleNamespace(key="app", display_name="Very Long Service Name")
        label = ServiceController.format_action_label(svc, "restart")
        assert "Very" in label
        assert "Long" not in label

    def test_parse_systemd_units_success(self):
        result = ShellActionResult(
            command="list-units",
            returncode=0,
            success=True,
            timed_out=False,
            stdout_text=(
                "  claude-bot.service  loaded active running  Claude Bot\n"
                "  nginx.service      loaded active running  nginx\n"
                "  not-a-service      loaded active running  other\n"
            ),
            stderr_text="",
        )
        units = ServiceController.parse_systemd_units(result)
        assert units == ["claude-bot.service", "nginx.service"]

    def test_parse_systemd_units_failure(self):
        result = ShellActionResult(
            command="list-units", returncode=1, success=False,
            timed_out=False, stdout_text="", stderr_text="error",
        )
        assert ServiceController.parse_systemd_units(result) == []

    def test_parse_systemd_units_respects_limit(self):
        lines = "\n".join(f"svc{i}.service loaded active running Svc" for i in range(20))
        result = ShellActionResult(
            command="list-units", returncode=0, success=True,
            timed_out=False, stdout_text=lines, stderr_text="",
        )
        units = ServiceController.parse_systemd_units(result, limit=3)
        assert len(units) == 3

    async def test_run_follow_up_checks_restart(self, tmp_dir):
        shell = ShellExecutor()
        controller = ServiceController(shell)
        service = SimpleNamespace(
            key="app",
            display_name="App",
            command_for=lambda action: {
                "status": "echo running",
                "health": "echo healthy",
                "logs": "echo log",
            }.get(action),
        )
        follow_up = await controller.run_follow_up_checks(
            service, tmp_dir, "restart"
        )
        assert follow_up.all_passed is True
        assert len(follow_up.checks) == 2

    async def test_run_follow_up_checks_restart_failure_gets_logs(self, tmp_dir):
        shell = ShellExecutor()
        controller = ServiceController(shell)
        service = SimpleNamespace(
            key="app",
            display_name="App",
            command_for=lambda action: {
                "status": "exit 1",
                "health": None,
                "logs": "echo failure-logs",
            }.get(action),
        )
        follow_up = await controller.run_follow_up_checks(
            service, tmp_dir, "restart"
        )
        assert follow_up.all_passed is False
        assert follow_up.logs_result is not None
        assert "failure-logs" in follow_up.logs_result.stdout_text


# ── ResolveRunner ────────────────────────────────────────────────────


class TestResolveRunner:
    def _make_ctx(self, tmp_dir, has_automation=True):
        profile = _make_profile(
            commands={"health": "echo ok"},
            root_path=tmp_dir,
        )
        claude_integration = AsyncMock()
        claude_integration.run_command = AsyncMock(return_value=SimpleNamespace(
            session_id="sess-123",
            content="Fixed the issue",
        ))
        return AgenticWorkspaceContext(
            current_directory=tmp_dir,
            current_workspace=tmp_dir,
            boundary_root=tmp_dir,
            project_automation=MagicMock() if has_automation else None,
            profile=profile if has_automation else None,
            claude_integration=claude_integration,
            storage=None,
            change_guard=None,
        )

    async def test_resolve_skips_if_no_automation(self, tmp_dir):
        ctx = self._make_ctx(tmp_dir, has_automation=False)
        shell = ShellExecutor()
        verify = VerifyPipeline(shell)
        runner = ResolveRunner(verify)
        result = await runner.run(ctx, user_id=1, session_id=None, initial_report=VerifyReport(
            results=[], failed_step=VerifyStep(label="x", command="x"), logs_result=None,
        ))
        assert result.success is False
        assert result.error == "Project automation unavailable"

    async def test_resolve_returns_success_if_already_passing(self, tmp_dir):
        ctx = self._make_ctx(tmp_dir)
        shell = ShellExecutor()
        verify = VerifyPipeline(shell)
        runner = ResolveRunner(verify)
        report = VerifyReport(results=[], failed_step=None, logs_result=None)
        result = await runner.run(ctx, user_id=1, session_id=None, initial_report=report)
        assert result.success is True


# ── PanelBuilder ─────────────────────────────────────────────────────


class TestPanelBuilder:
    def _make_builder(self):
        shell = ShellExecutor()
        verify = VerifyPipeline(shell)
        services = ServiceController(shell)
        return PanelBuilder(verify, shell, services)

    def _make_ctx(self, tmp_dir, with_storage=False, with_claude=False):
        profile = _make_profile(
            commands={"health": "echo ok"},
            root_path=tmp_dir,
        )
        return AgenticWorkspaceContext(
            current_directory=tmp_dir,
            current_workspace=tmp_dir,
            boundary_root=tmp_dir,
            project_automation=MagicMock(),
            profile=profile,
            storage=MagicMock() if with_storage else None,
            claude_integration=AsyncMock() if with_claude else None,
        )

    def test_build_start_keyboard(self):
        builder = self._make_builder()
        markup = builder.build_start_keyboard()
        buttons = markup.inline_keyboard[0]
        assert len(buttons) == 3
        assert buttons[0].callback_data == "act:status"
        assert buttons[1].callback_data == "act:verify"
        assert buttons[2].callback_data == "act:resolve"

    def test_build_control_panel_markup(self):
        builder = self._make_builder()
        markup = builder.build_control_panel_markup()
        assert len(markup.inline_keyboard) == 1
        assert len(markup.inline_keyboard[0]) == 3

    def test_build_reply_keyboard(self):
        builder = self._make_builder()
        markup = builder.build_reply_keyboard()
        assert markup.is_persistent is True
        assert markup.resize_keyboard is True
        assert len(markup.keyboard[0]) == 3

    def test_map_reply_action_status(self):
        assert PanelBuilder.map_reply_action("\U0001f4c2 \u0421\u0442\u0430\u0442\u0443\u0441") == "status"

    def test_map_reply_action_verify(self):
        assert PanelBuilder.map_reply_action("\u2705 \u041f\u0440\u043e\u0432\u0435\u0440\u0438\u0442\u044c") == "verify"

    def test_map_reply_action_resolve(self):
        assert PanelBuilder.map_reply_action("\U0001f6e0 \u0420\u0430\u0437\u0431\u0435\u0440\u0438\u0441\u044c") == "resolve"

    def test_map_reply_action_unknown(self):
        assert PanelBuilder.map_reply_action("random text") is None

    def test_format_relative_path_root(self, tmp_dir):
        assert PanelBuilder.format_relative_path(tmp_dir, tmp_dir) == "/"

    def test_format_relative_path_subdir(self, tmp_dir):
        sub = tmp_dir / "myproject"
        sub.mkdir()
        assert PanelBuilder.format_relative_path(sub, tmp_dir) == "myproject"

    def test_format_relative_path_outside(self, tmp_dir):
        outside = Path("/some/other/path")
        result = PanelBuilder.format_relative_path(outside, tmp_dir)
        assert result == "/some/other/path"

    def test_format_job_status(self, tmp_dir):
        builder = self._make_builder()
        job = SimpleNamespace(
            workspace_root=tmp_dir,
            status="running",
            action_key="deploy",
            job_id="abcdef1234567890",
            verification_command=None,
        )
        text = builder.format_job_status(job, tmp_dir)
        assert "deploy" in text
        assert "abcdef12" in text

    def test_format_job_status_with_verification(self, tmp_dir):
        builder = self._make_builder()
        job = SimpleNamespace(
            workspace_root=tmp_dir,
            status="running",
            action_key="start",
            job_id="abcdef1234567890",
            verification_command="curl :8080",
            verification_status="passed",
            verification_attempts=2,
        )
        text = builder.format_job_status(job, tmp_dir)
        assert "\u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 \u043f\u0440\u043e\u0439\u0434\u0435\u043d\u0430" in text
        assert "(2x)" in text

    def test_format_job_verification_none_without_command(self):
        job = SimpleNamespace(verification_command=None)
        assert PanelBuilder.format_job_verification(job) is None

    def test_format_job_verification_pending(self):
        job = SimpleNamespace(
            verification_command="check",
            verification_status="pending",
            verification_attempts=0,
        )
        label = PanelBuilder.format_job_verification(job)
        assert "\u043e\u0436\u0438\u0434\u0430\u0435\u0442\u0441\u044f" in label

    async def test_build_status_text(self, tmp_dir):
        builder = self._make_builder()
        ctx = self._make_ctx(tmp_dir)
        text = await builder.build_status_text(ctx, user_id=1, session_id="abc123")
        assert "\u0421\u0442\u0430\u0442\u0443\u0441" in text
        assert "\u0430\u043a\u0442\u0438\u0432\u043d\u0430" in text

    async def test_build_status_text_no_session(self, tmp_dir):
        builder = self._make_builder()
        ctx = self._make_ctx(tmp_dir)
        text = await builder.build_status_text(ctx, user_id=1, session_id=None)
        assert "\u043d\u0435\u0442" in text

    async def test_build_status_text_with_active_task(self, tmp_dir):
        builder = self._make_builder()
        ctx = self._make_ctx(tmp_dir)
        text = await builder.build_status_text(
            ctx, user_id=1, session_id=None,
            active_task_elapsed=30, queue_size=2,
        )
        assert "30" in text
        assert "\u043e\u0447\u0435\u0440\u0435\u0434\u0438" in text

    async def test_build_panel_text(self, tmp_dir):
        builder = self._make_builder()
        ctx = self._make_ctx(tmp_dir)
        text = await builder.build_panel_text(
            ctx, user_id=1, session_id="abc", verbose_level=1,
        )
        assert "\u041f\u0430\u043d\u0435\u043b\u044c \u0443\u043f\u0440\u0430\u0432\u043b\u0435\u043d\u0438\u044f" in text
        assert "\u043d\u043e\u0440\u043c\u0430\u043b\u044c\u043d\u043e" in text

    async def test_build_recent_text_no_storage(self, tmp_dir):
        builder = self._make_builder()
        ctx = self._make_ctx(tmp_dir, with_storage=False)
        text = await builder.build_recent_text(ctx, user_id=1)
        assert "\u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u043d\u043e" in text

    async def test_build_jobs_text_no_runtime(self, tmp_dir):
        builder = self._make_builder()
        ctx = self._make_ctx(tmp_dir)
        text, markup = await builder.build_jobs_text(ctx)
        assert "\u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u043d\u044b" in text

    async def test_build_services_text_no_services(self, tmp_dir):
        builder = self._make_builder()
        profile = _make_profile(root_path=tmp_dir)
        profile.services = []
        ctx = AgenticWorkspaceContext(
            current_directory=tmp_dir,
            current_workspace=tmp_dir,
            boundary_root=tmp_dir,
            project_automation=MagicMock(),
            profile=profile,
        )
        text, markup = await builder.build_services_text(ctx)
        assert "\u043d\u0435 \u043d\u0430\u0441\u0442\u0440\u043e\u0435\u043d\u044b" in text

    async def test_build_workspace_catalog_no_automation(self, tmp_dir):
        builder = self._make_builder()
        subdir = tmp_dir / "project_a"
        subdir.mkdir()
        ctx = AgenticWorkspaceContext(
            current_directory=tmp_dir,
            current_workspace=tmp_dir,
            boundary_root=tmp_dir,
            project_automation=None,
            profile=None,
        )
        text, markup = await builder.build_workspace_catalog(ctx)
        assert "project_a" in text

    async def test_build_workspace_catalog_with_automation(self, tmp_dir):
        builder = self._make_builder()
        summary = SimpleNamespace(
            button_label="ProjectA",
            relative_path="project_a",
            root_path=tmp_dir / "project_a",
        )
        pa = MagicMock()
        pa.list_workspace_summaries.return_value = [summary]
        pa.describe_workspace_summary_lines.return_value = [
            "ProjectA stuff"
        ]
        ctx = AgenticWorkspaceContext(
            current_directory=tmp_dir,
            current_workspace=tmp_dir,
            boundary_root=tmp_dir,
            project_automation=pa,
            profile=_make_profile(root_path=tmp_dir),
        )
        text, markup = await builder.build_workspace_catalog(ctx)
        assert "\u041f\u0440\u043e\u0435\u043a\u0442\u044b" in text
        # Should have project button
        found = any(
            btn.callback_data == "cd:project_a"
            for row in markup.inline_keyboard
            for btn in row
        )
        assert found


# ── AgenticWorkspaceContext ──────────────────────────────────────────


class TestAgenticWorkspaceContext:
    def test_format_relative_path_root(self, tmp_dir):
        resolved = tmp_dir.resolve()
        ctx = AgenticWorkspaceContext(
            current_directory=resolved,
            current_workspace=resolved,
            boundary_root=resolved,
            project_automation=None,
            profile=None,
        )
        assert ctx.format_relative_path(resolved) == "/"

    def test_format_relative_path_subdir(self, tmp_dir):
        resolved = tmp_dir.resolve()
        sub = resolved / "sub"
        sub.mkdir()
        ctx = AgenticWorkspaceContext(
            current_directory=resolved,
            current_workspace=resolved,
            boundary_root=resolved,
            project_automation=None,
            profile=None,
        )
        assert ctx.format_relative_path(sub) == "sub"
