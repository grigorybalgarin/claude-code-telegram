"""Tests for project automation profiles and commands."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.bot.features.project_automation import ProjectAutomationManager
from src.bot.handlers.command import (
    diag_command,
    playbooks_command,
    run_playbook_command,
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
    assert profile.commands["build"] == "pnpm run build"


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


def test_build_automation_plan_matches_test_playbook(tmp_path):
    """Natural-language test requests should map to the test playbook."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "package.json").write_text(
        json.dumps({"name": "demo", "scripts": {"test": "vitest run"}}),
        encoding="utf-8",
    )
    manager = ProjectAutomationManager()

    plan = manager.build_automation_plan("почини тесты и прогоняй их до конца", tmp_path, tmp_path)

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
