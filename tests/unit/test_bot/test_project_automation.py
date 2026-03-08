"""Tests for project automation profiles and commands."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.bot.features.project_automation import ProjectAutomationManager
from src.bot.handlers.command import (
    diag_command,
    playbooks_command,
    recent_activity_command,
    run_playbook_command,
    show_projects,
)
from src.config import create_test_config


def test_detects_python_profile_and_playbooks(tmp_path):
    """Python workspace detection should expose commands and playbooks."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        """
[tool.poetry]
name = "demo"
version = "0.1.0"

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100

[tool.mypy]
python_version = "3.11"
        """.strip(),
        encoding="utf-8",
    )
    (tmp_path / "poetry.lock").write_text("", encoding="utf-8")
    (tmp_path / "tests").mkdir()

    manager = ProjectAutomationManager()
    profile = manager.build_profile(tmp_path, tmp_path)
    playbooks = manager.list_playbooks(profile)

    assert profile.root_path == tmp_path
    assert "python" in profile.stacks
    assert "poetry" in profile.package_managers
    assert profile.commands["install"] == "poetry install"
    assert profile.commands["test"] == "poetry run pytest"
    assert profile.commands["lint"] == "poetry run ruff check ."
    assert profile.commands["format"] == "poetry run ruff format ."
    assert profile.commands["typecheck"] == "poetry run mypy ."
    assert [playbook.slug for playbook in playbooks] == [
        "doctor",
        "setup",
        "test",
        "quality",
        "review",
    ]


def test_detects_node_workspace_from_subdirectory(tmp_path):
    """Workspace root detection should climb to the package root."""
    project_root = tmp_path / "webapp"
    nested = project_root / "src" / "components"
    nested.mkdir(parents=True)
    (project_root / "pnpm-lock.yaml").write_text("", encoding="utf-8")
    (project_root / "package.json").write_text(
        json.dumps(
            {
                "name": "webapp",
                "scripts": {
                    "test": "vitest run",
                    "lint": "eslint .",
                    "typecheck": "tsc --noEmit",
                    "build": "vite build",
                },
            }
        ),
        encoding="utf-8",
    )

    manager = ProjectAutomationManager()
    profile = manager.build_profile(nested, tmp_path)

    assert profile.root_path == project_root
    assert "node" in profile.stacks
    assert "pnpm" in profile.package_managers
    assert profile.commands["install"] == "pnpm install"
    assert profile.commands["test"] == "pnpm run test"
    assert profile.commands["lint"] == "pnpm run lint"
    assert profile.commands["typecheck"] == "pnpm run typecheck"
    assert profile.commands["build"] == "pnpm run build"


def test_list_workspace_summaries_includes_nested_requirements_project(tmp_path):
    """Requirements-only Python projects should be discovered as nested workspaces."""
    container_root = tmp_path / "Gr_dev"
    container_root.mkdir()
    (container_root / ".git").mkdir()

    nested_root = container_root / "freelance-aggregator"
    nested_root.mkdir()
    (nested_root / "requirements.txt").write_text(
        "fastapi==0.115.0\nuvicorn==0.30.6\n",
        encoding="utf-8",
    )

    manager = ProjectAutomationManager()
    summaries = manager.list_workspace_summaries(tmp_path)

    assert "Gr_dev" in [summary.relative_path for summary in summaries]
    assert "Gr_dev/freelance-aggregator" in [
        summary.relative_path for summary in summaries
    ]


def test_build_playbook_prompt_includes_detected_commands_and_extra_note(tmp_path):
    """Generated prompts should include the profile and operator note."""
    (tmp_path / "pyproject.toml").write_text(
        "[tool.poetry]\nname='demo'\nversion='0.1.0'\n",
        encoding="utf-8",
    )
    (tmp_path / "poetry.lock").write_text("", encoding="utf-8")

    manager = ProjectAutomationManager()
    profile = manager.build_profile(tmp_path, tmp_path)
    prompt = manager.build_playbook_prompt(
        "setup", profile, extra_instructions="Prefer the API service first"
    )

    assert 'You are running the "setup" project playbook.' in prompt
    assert "poetry install" in prompt
    assert "Prefer the API service first" in prompt


def test_quality_playbook_prompt_mentions_typecheck_when_available(tmp_path):
    """Quality playbook should mention detected typecheck commands."""
    (tmp_path / "package.json").write_text(
        json.dumps(
            {
                "name": "webapp",
                "scripts": {
                    "lint": "eslint .",
                    "typecheck": "tsc --noEmit",
                },
            }
        ),
        encoding="utf-8",
    )

    manager = ProjectAutomationManager()
    profile = manager.build_profile(tmp_path, tmp_path)
    prompt = manager.build_playbook_prompt("quality", profile)

    assert "Run type checks before finishing" in prompt
    assert "npm run typecheck" in prompt


def test_build_automation_plan_matches_test_playbook(tmp_path):
    """Natural-language test requests should map to the test playbook."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "package.json").write_text(
        json.dumps({"name": "demo", "scripts": {"test": "vitest run"}}),
        encoding="utf-8",
    )
    manager = ProjectAutomationManager()

    plan = manager.build_automation_plan(
        "почини тесты и прогоняй их до конца", tmp_path, tmp_path
    )

    assert plan.matched_playbook == "test"
    assert plan.should_checkpoint is True
    assert plan.should_verify is True
    assert 'You are running the "test" project playbook.' in plan.prompt


def test_build_automation_plan_for_read_only_request_stays_read_only(tmp_path):
    """Diagnostic requests should stay read-only and skip checkpoints."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths=['tests']\n",
        encoding="utf-8",
    )
    manager = ProjectAutomationManager()

    plan = manager.build_automation_plan("что не так с проектом?", tmp_path, tmp_path)

    assert plan.matched_playbook == "doctor"
    assert plan.read_only is True
    assert plan.should_checkpoint is False
    assert plan.should_verify is False


def test_build_automation_plan_routes_to_named_workspace(tmp_path):
    """Autopilot should switch to the named workspace when the request is explicit."""
    current_root = tmp_path / "ClaudeBot"
    current_root.mkdir()
    (current_root / ".git").mkdir()
    (current_root / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths=['tests']\n",
        encoding="utf-8",
    )

    target_root = tmp_path / "FreelanceAggregator"
    target_root.mkdir()
    (target_root / ".git").mkdir()
    (target_root / "package.json").write_text(
        json.dumps({"name": "fa", "scripts": {"test": "vitest run"}}),
        encoding="utf-8",
    )

    manager = ProjectAutomationManager()
    plan = manager.build_automation_plan(
        "почини тесты в FreelanceAggregator",
        current_root,
        tmp_path,
    )

    assert plan.workspace_root == target_root
    assert plan.workspace_changed is True
    assert plan.matched_playbook == "test"


def test_build_automation_plan_routes_to_nested_workspace_from_relative_path(tmp_path):
    """Relative path mentions should find nested workspaces under the approved root."""
    current_root = tmp_path / "ClaudeBot"
    current_root.mkdir()
    (current_root / ".git").mkdir()
    (current_root / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths=['tests']\n",
        encoding="utf-8",
    )

    nested_root = tmp_path / "MacProjects" / "Poolych"
    nested_root.mkdir(parents=True)
    (nested_root / ".git").mkdir()
    (nested_root / "package.json").write_text(
        json.dumps({"name": "poolych", "scripts": {"test": "vitest run"}}),
        encoding="utf-8",
    )

    manager = ProjectAutomationManager()
    plan = manager.build_automation_plan(
        "сделай аудит в MacProjects/Poolych",
        current_root,
        tmp_path,
    )

    assert plan.workspace_root == nested_root
    assert plan.workspace_changed is True
    assert plan.matched_playbook == "review"
    assert plan.read_only is True


def test_build_automation_plan_keeps_current_workspace_when_request_is_generic(
    tmp_path,
):
    """Generic requests should stay in the current workspace."""
    current_root = tmp_path / "ClaudeBot"
    current_root.mkdir()
    (current_root / ".git").mkdir()
    (current_root / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths=['tests']\n",
        encoding="utf-8",
    )

    other_root = tmp_path / "FreelanceAggregator"
    other_root.mkdir()
    (other_root / ".git").mkdir()
    (other_root / "package.json").write_text(
        json.dumps({"name": "fa", "scripts": {"test": "vitest run"}}),
        encoding="utf-8",
    )

    manager = ProjectAutomationManager()
    plan = manager.build_automation_plan("почини тесты", current_root, tmp_path)

    assert plan.workspace_root == current_root
    assert plan.workspace_changed is False


def test_workspace_profile_overrides_apply_aliases_notes_and_priority(tmp_path):
    """Explicit workspace profiles should enrich detected workspace metadata."""
    claude_root = tmp_path / "ClaudeBot"
    claude_root.mkdir()
    (claude_root / ".git").mkdir()
    (claude_root / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths=['tests']\n",
        encoding="utf-8",
    )

    profiles_path = tmp_path / "workspace_profiles.yaml"
    profiles_path.write_text(
        """
workspaces:
  - path: ClaudeBot
    name: ClaudeBot
    aliases:
      - telegram bot
      - claude-bot
    priority: 50
    notes: Keep deploy workflow under ops/server unchanged.
        """.strip(),
        encoding="utf-8",
    )

    manager = ProjectAutomationManager(workspace_profiles_path=profiles_path)
    profile = manager.build_profile(claude_root, tmp_path)
    summaries = manager.list_workspace_summaries(tmp_path)

    assert profile.display_name == "ClaudeBot"
    assert "telegram bot" in profile.aliases
    assert profile.operator_notes == "Keep deploy workflow under ops/server unchanged."
    assert profile.sort_priority == 50
    assert summaries[0].display_name == "ClaudeBot"
    assert "telegram bot" in summaries[0].aliases


def test_list_operator_commands_uses_configured_overrides(tmp_path):
    """Operator command list should expose explicit health/build/start/dev/deploy actions."""
    workspace_root = tmp_path / "Poolbill"
    workspace_root.mkdir()
    (workspace_root / "package.json").write_text(
        json.dumps({"name": "poolbill", "scripts": {"test": "vitest run"}}),
        encoding="utf-8",
    )

    profiles_path = tmp_path / "workspace_profiles.yaml"
    profiles_path.write_text(
        """
workspaces:
  - path: Poolbill
    name: Poolbill
    commands:
      health: npm run typecheck
      build: npm run build
      start: npm run start
      dev: npm run dev
      deploy: ./deploy.sh
        """.strip(),
        encoding="utf-8",
    )

    manager = ProjectAutomationManager(workspace_profiles_path=profiles_path)
    profile = manager.build_profile(workspace_root, tmp_path)

    assert manager.list_operator_commands(profile) == [
        ("health", "npm run typecheck"),
        ("build", "npm run build"),
        ("start", "npm run start"),
        ("dev", "npm run dev"),
        ("deploy", "./deploy.sh"),
    ]


def test_workspace_profile_parses_managed_systemd_service(tmp_path):
    """Workspace profiles should parse deterministic managed services."""
    workspace_root = tmp_path / "ClaudeBot"
    workspace_root.mkdir()
    (workspace_root / ".git").mkdir()
    (workspace_root / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths=['tests']\n",
        encoding="utf-8",
    )

    profiles_path = tmp_path / "workspace_profiles.yaml"
    profiles_path.write_text(
        """
workspaces:
  - path: ClaudeBot
    name: ClaudeBot
    services:
      - key: app
        name: ClaudeBot Service
        type: systemd
        unit: claude-bot.service
        logs_tail: 50
        """.strip(),
        encoding="utf-8",
    )

    manager = ProjectAutomationManager(workspace_profiles_path=profiles_path)
    profile = manager.build_profile(workspace_root, tmp_path)
    summaries = manager.list_workspace_summaries(tmp_path)

    assert len(profile.services) == 1
    service = profile.services[0]
    assert service.key == "app"
    assert service.display_name == "ClaudeBot Service"
    assert service.status_command == "systemctl status claude-bot.service --no-pager"
    assert service.health_command == "systemctl is-active claude-bot.service"
    assert service.restart_command == "systemctl restart claude-bot.service"
    assert service.logs_command == "journalctl -u claude-bot.service -n 50 --no-pager"
    assert service.available_actions == (
        "status",
        "health",
        "logs",
        "restart",
        "start",
        "stop",
    )
    assert summaries[0].services_count == 1


def test_build_automation_plan_routes_to_workspace_alias(tmp_path):
    """Workspace aliases from the profile config should influence routing."""
    current_root = tmp_path / "ClaudeBot"
    current_root.mkdir()
    (current_root / ".git").mkdir()
    (current_root / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths=['tests']\n",
        encoding="utf-8",
    )

    target_root = tmp_path / "MacProjects" / "Poolych"
    target_root.mkdir(parents=True)
    (target_root / ".git").mkdir()
    (target_root / "package.json").write_text(
        json.dumps({"name": "poolych", "scripts": {"test": "vitest run"}}),
        encoding="utf-8",
    )

    profiles_path = tmp_path / "workspace_profiles.yaml"
    profiles_path.write_text(
        """
workspaces:
  - path: MacProjects/Poolych
    name: Poolych
    aliases:
      - billiards
      - pool game
        """.strip(),
        encoding="utf-8",
    )

    manager = ProjectAutomationManager(workspace_profiles_path=profiles_path)
    plan = manager.build_automation_plan(
        "почини тесты в billiards",
        current_root,
        tmp_path,
    )

    assert plan.workspace_root == target_root
    assert plan.workspace_changed is True
    assert plan.matched_playbook == "test"


def test_verification_commands_include_typecheck(tmp_path):
    """Auto-verify should include typecheck between lint and tests when available."""
    (tmp_path / "package.json").write_text(
        json.dumps(
            {
                "name": "webapp",
                "scripts": {
                    "lint": "eslint .",
                    "typecheck": "tsc --noEmit",
                    "test": "vitest run",
                    "build": "vite build",
                },
            }
        ),
        encoding="utf-8",
    )

    manager = ProjectAutomationManager()
    profile = manager.build_profile(tmp_path, tmp_path)

    assert manager.get_verification_commands(profile) == [
        "npm run lint",
        "npm run typecheck",
        "npm run test",
        "npm run build",
    ]


def test_list_workspace_summaries_skips_container_directories(tmp_path):
    """Workspace catalog should show real workspaces, not generic containers."""
    claude_root = tmp_path / "ClaudeBot"
    claude_root.mkdir()
    (claude_root / ".git").mkdir()
    (claude_root / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths=['tests']\n",
        encoding="utf-8",
    )

    nested_root = tmp_path / "MacProjects" / "Poolych"
    nested_root.mkdir(parents=True)
    (nested_root / ".git").mkdir()
    (nested_root / "package.json").write_text(
        json.dumps({"name": "poolych", "scripts": {"test": "vitest run"}}),
        encoding="utf-8",
    )

    manager = ProjectAutomationManager()
    summaries = manager.list_workspace_summaries(tmp_path)

    assert [summary.relative_path for summary in summaries] == [
        "ClaudeBot",
        "MacProjects/Poolych",
    ]


def test_resolve_workspace_reference_matches_name_and_path(tmp_path):
    """Manual workspace switches should accept both names and relative paths."""
    claude_root = tmp_path / "ClaudeBot"
    claude_root.mkdir()
    (claude_root / ".git").mkdir()
    (claude_root / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths=['tests']\n",
        encoding="utf-8",
    )

    nested_root = tmp_path / "MacProjects" / "Poolych"
    nested_root.mkdir(parents=True)
    (nested_root / ".git").mkdir()
    (nested_root / "package.json").write_text(
        json.dumps({"name": "poolych", "scripts": {"test": "vitest run"}}),
        encoding="utf-8",
    )

    manager = ProjectAutomationManager()

    by_name = manager.resolve_workspace_reference("Poolych", tmp_path)
    by_path = manager.resolve_workspace_reference("MacProjects/Poolych", tmp_path)

    assert by_name is not None
    assert by_name.root_path == nested_root
    assert by_path is not None
    assert by_path.root_path == nested_root


def test_resolve_workspace_reference_matches_alias(tmp_path):
    """Manual workspace switches should accept configured aliases."""
    nested_root = tmp_path / "MacProjects" / "Poolych"
    nested_root.mkdir(parents=True)
    (nested_root / ".git").mkdir()
    (nested_root / "package.json").write_text(
        json.dumps({"name": "poolych", "scripts": {"test": "vitest run"}}),
        encoding="utf-8",
    )

    profiles_path = tmp_path / "workspace_profiles.yaml"
    profiles_path.write_text(
        """
workspaces:
  - path: MacProjects/Poolych
    name: Poolych
    aliases:
      - billiards
        """.strip(),
        encoding="utf-8",
    )

    manager = ProjectAutomationManager(workspace_profiles_path=profiles_path)

    by_alias = manager.resolve_workspace_reference("billiards", tmp_path)
    summary_lines = manager.describe_workspace_summary_lines(
        manager.list_workspace_summaries(tmp_path)[0]
    )

    assert by_alias is not None
    assert by_alias.root_path == nested_root
    assert any("aliases:" in line for line in summary_lines)


@pytest.mark.asyncio
async def test_playbooks_command_lists_available_playbooks(tmp_path):
    """The Telegram command should render detected playbooks."""
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths=['tests']\n",
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    settings = create_test_config(approved_directory=str(tmp_path), agentic_mode=False)
    manager = ProjectAutomationManager()

    update = MagicMock()
    update.effective_user.id = 123
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.user_data = {"current_directory": tmp_path}
    context.bot_data = {
        "settings": settings,
        "audit_logger": None,
        "features": SimpleNamespace(get_project_automation=lambda: manager),
    }

    await playbooks_command(update, context)

    message = update.message.reply_text.call_args[0][0]
    assert "Workspace Playbooks" in message
    assert "doctor" in message
    assert "test" in message


@pytest.mark.asyncio
async def test_show_projects_renders_workspace_profiles(tmp_path):
    """Projects command should render discovered workspace profiles."""
    claude_root = tmp_path / "ClaudeBot"
    claude_root.mkdir()
    (claude_root / ".git").mkdir()
    (claude_root / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths=['tests']\n",
        encoding="utf-8",
    )

    nested_root = tmp_path / "MacProjects" / "Poolych"
    nested_root.mkdir(parents=True)
    (nested_root / ".git").mkdir()
    (nested_root / "package.json").write_text(
        json.dumps({"name": "poolych", "scripts": {"test": "vitest run"}}),
        encoding="utf-8",
    )

    settings = create_test_config(approved_directory=str(tmp_path), agentic_mode=False)
    manager = ProjectAutomationManager()

    update = MagicMock()
    update.effective_user.id = 123
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.user_data = {"current_directory": claude_root}
    context.bot_data = {
        "settings": settings,
        "audit_logger": None,
        "features": SimpleNamespace(get_project_automation=lambda: manager),
    }

    await show_projects(update, context)

    message = update.message.reply_text.call_args[0][0]
    assert "Workspace Profiles" in message
    assert "ClaudeBot" in message
    assert "MacProjects/Poolych" in message


@pytest.mark.asyncio
async def test_run_playbook_without_args_shows_usage(tmp_path):
    """The run command should show usage when no playbook is provided."""
    settings = create_test_config(approved_directory=str(tmp_path), agentic_mode=False)

    update = MagicMock()
    update.effective_user.id = 123
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.args = []
    context.user_data = {"current_directory": tmp_path}
    context.bot_data = {
        "settings": settings,
        "audit_logger": None,
        "claude_integration": None,
    }

    await run_playbook_command(update, context)

    message = update.message.reply_text.call_args[0][0]
    assert "Usage" in message
    assert "/playbooks" in message


@pytest.mark.asyncio
async def test_diag_command_renders_workspace_summary(tmp_path):
    """Diagnostics command should include storage and workspace information."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "package.json").write_text(
        json.dumps({"name": "demo", "scripts": {"test": "vitest run"}}),
        encoding="utf-8",
    )
    settings = create_test_config(approved_directory=str(tmp_path), agentic_mode=False)
    manager = ProjectAutomationManager()

    update = MagicMock()
    update.effective_user.id = 123
    update.message.reply_text = AsyncMock()

    storage = MagicMock()
    storage.health_check = AsyncMock(return_value=True)
    storage.audit.get_user_audit_log = AsyncMock(
        return_value=[
            SimpleNamespace(
                event_type="automation_run",
                event_data={
                    "details": {
                        "playbook": "doctor",
                        "workspace_root": str(tmp_path),
                        "read_only": True,
                    }
                },
            )
        ]
    )
    mem0_client = MagicMock()
    mem0_client.health = AsyncMock(return_value=True)

    context = MagicMock()
    context.user_data = {"current_directory": tmp_path}
    context.bot_data = {
        "settings": settings,
        "audit_logger": None,
        "storage": storage,
        "mem0_client": mem0_client,
        "claude_integration": None,
        "features": SimpleNamespace(
            get_project_automation=lambda: manager,
            get_git_integration=lambda: None,
        ),
    }

    await diag_command(update, context)

    message = update.message.reply_text.call_args[0][0]
    assert "Diagnostics" in message
    assert "Storage" in message
    assert "Workspace root" in message
    assert "Last autopilot" in message


@pytest.mark.asyncio
async def test_recent_activity_command_lists_autopilot_runs(tmp_path):
    """Recent activity should include the latest autopilot runs."""
    settings = create_test_config(approved_directory=str(tmp_path), agentic_mode=False)

    update = MagicMock()
    update.effective_user.id = 123
    update.message.reply_text = AsyncMock()

    storage = MagicMock()
    storage.messages.get_user_messages = AsyncMock(return_value=[])
    storage.audit.get_user_audit_log = AsyncMock(
        return_value=[
            SimpleNamespace(
                event_type="automation_run",
                event_data={
                    "details": {
                        "playbook": "test",
                        "workspace_root": str(tmp_path / "repo"),
                        "checkpoint_created": True,
                    }
                },
                success=True,
                timestamp=datetime.now(timezone.utc),
            )
        ]
    )

    context = MagicMock()
    context.user_data = {"current_directory": tmp_path}
    context.bot_data = {
        "settings": settings,
        "audit_logger": None,
        "storage": storage,
    }

    await recent_activity_command(update, context)

    message = update.message.reply_text.call_args[0][0]
    assert "Recent autopilot" in message
    assert "test" in message
    assert "verified" in message
