"""Tests for the MessageOrchestrator."""

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.bot.features.operator_runtime import WorkspaceOperatorRuntime
from src.bot.features.project_automation import ProjectAutomationManager
from src.bot.orchestrator import MessageOrchestrator, _redact_secrets
from src.config import create_test_config


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def agentic_settings(tmp_dir):
    return create_test_config(approved_directory=str(tmp_dir), agentic_mode=True)


@pytest.fixture
def classic_settings(tmp_dir):
    return create_test_config(approved_directory=str(tmp_dir), agentic_mode=False)


@pytest.fixture
def group_thread_settings(tmp_dir):
    project_dir = tmp_dir / "project_a"
    project_dir.mkdir()
    config_file = tmp_dir / "projects.yaml"
    config_file.write_text(
        "projects:\n"
        "  - slug: project_a\n"
        "    name: Project A\n"
        "    path: project_a\n",
        encoding="utf-8",
    )
    return create_test_config(
        approved_directory=str(tmp_dir),
        agentic_mode=False,
        enable_project_threads=True,
        project_threads_mode="group",
        project_threads_chat_id=-1001234567890,
        projects_config_path=str(config_file),
    )


@pytest.fixture
def private_thread_settings(tmp_dir):
    project_dir = tmp_dir / "project_a"
    project_dir.mkdir()
    config_file = tmp_dir / "projects.yaml"
    config_file.write_text(
        "projects:\n"
        "  - slug: project_a\n"
        "    name: Project A\n"
        "    path: project_a\n",
        encoding="utf-8",
    )
    return create_test_config(
        approved_directory=str(tmp_dir),
        agentic_mode=False,
        enable_project_threads=True,
        project_threads_mode="private",
        projects_config_path=str(config_file),
    )


@pytest.fixture
def deps():
    return {
        "claude_integration": MagicMock(),
        "storage": MagicMock(),
        "security_validator": MagicMock(),
        "rate_limiter": MagicMock(),
        "audit_logger": MagicMock(),
    }


def test_agentic_registers_13_commands(agentic_settings, deps):
    """Agentic mode registers built-in commands plus diagnostics and playbooks."""
    orchestrator = MessageOrchestrator(agentic_settings, deps)
    app = MagicMock()
    app.add_handler = MagicMock()

    orchestrator.register_handlers(app)

    # Collect all CommandHandler registrations
    from telegram.ext import CommandHandler

    cmd_handlers = [
        call
        for call in app.add_handler.call_args_list
        if isinstance(call[0][0], CommandHandler)
    ]
    commands = [h[0][0].commands for h in cmd_handlers]

    assert len(cmd_handlers) == 13
    assert frozenset({"start"}) in commands
    assert frozenset({"new"}) in commands
    assert frozenset({"status"}) in commands
    assert frozenset({"diag"}) in commands
    assert frozenset({"recent"}) in commands
    assert frozenset({"playbooks"}) in commands
    assert frozenset({"run"}) in commands
    assert frozenset({"verbose"}) in commands
    assert frozenset({"repo"}) in commands
    assert frozenset({"restart"}) in commands
    assert frozenset({"stats"}) in commands


def test_classic_registers_18_commands(classic_settings, deps):
    """Classic mode registers the full command set including automation tools."""
    orchestrator = MessageOrchestrator(classic_settings, deps)
    app = MagicMock()
    app.add_handler = MagicMock()

    orchestrator.register_handlers(app)

    from telegram.ext import CommandHandler

    cmd_handlers = [
        call
        for call in app.add_handler.call_args_list
        if isinstance(call[0][0], CommandHandler)
    ]

    assert len(cmd_handlers) == 18


def test_agentic_registers_text_document_photo_handlers(agentic_settings, deps):
    """Agentic mode registers text, document, photo, and voice message handlers."""
    orchestrator = MessageOrchestrator(agentic_settings, deps)
    app = MagicMock()
    app.add_handler = MagicMock()

    orchestrator.register_handlers(app)

    from telegram.ext import CallbackQueryHandler, MessageHandler

    msg_handlers = [
        call
        for call in app.add_handler.call_args_list
        if isinstance(call[0][0], MessageHandler)
    ]
    cb_handlers = [
        call
        for call in app.add_handler.call_args_list
        if isinstance(call[0][0], CallbackQueryHandler)
    ]

    # 5 message handlers (text, document, photo, voice, video_note)
    assert len(msg_handlers) == 5
    # 2 callback handlers (act: quick actions + cd: project selection)
    assert len(cb_handlers) == 2


async def test_agentic_bot_commands(agentic_settings, deps):
    """Agentic mode returns command menu with diagnostics and playbooks."""
    orchestrator = MessageOrchestrator(agentic_settings, deps)
    commands = await orchestrator.get_bot_commands()

    assert len(commands) == 13
    cmd_names = [c.command for c in commands]
    assert "start" in cmd_names
    assert "new" in cmd_names
    assert "status" in cmd_names
    assert "diag" in cmd_names
    assert "recent" in cmd_names
    assert "playbooks" in cmd_names
    assert "run" in cmd_names
    assert "verbose" in cmd_names
    assert "stats" in cmd_names


async def test_classic_bot_commands(classic_settings, deps):
    """Classic mode returns the full command menu including automation tools."""
    orchestrator = MessageOrchestrator(classic_settings, deps)
    commands = await orchestrator.get_bot_commands()

    assert len(commands) == 18
    cmd_names = [c.command for c in commands]
    assert "start" in cmd_names
    assert "help" in cmd_names
    assert "diag" in cmd_names
    assert "recent" in cmd_names
    assert "playbooks" in cmd_names
    assert "run" in cmd_names
    assert "git" in cmd_names
    assert "restart" in cmd_names


async def test_restart_command_sends_sigterm(deps):
    """restart_command sends SIGTERM to the current process."""
    from unittest.mock import patch

    from src.bot.handlers.command import restart_command

    update = MagicMock()
    update.effective_user.id = 123
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.bot_data = {"audit_logger": None}

    with patch("src.bot.handlers.command.os.kill") as mock_kill:
        await restart_command(update, context)

    import os
    import signal

    mock_kill.assert_called_once_with(os.getpid(), signal.SIGTERM)
    # Verify confirmation message was sent
    update.message.reply_text.assert_called_once()
    msg = update.message.reply_text.call_args[0][0]
    assert "Restarting" in msg


async def test_agentic_start_no_keyboard(agentic_settings, deps):
    """Agentic /start sends the autopilot message with control buttons."""
    orchestrator = MessageOrchestrator(agentic_settings, deps)

    update = MagicMock()
    update.effective_user.first_name = "Alice"
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.user_data = {}
    context.bot_data = {"settings": agentic_settings}
    for k, v in deps.items():
        context.bot_data[k] = v

    await orchestrator.agentic_start(update, context)

    update.message.reply_text.assert_called_once()
    call_kwargs = update.message.reply_text.call_args
    assert "reply_markup" in call_kwargs.kwargs
    markup = call_kwargs.kwargs["reply_markup"]
    assert markup is not None
    labels = [
        button.text
        for row in markup.inline_keyboard
        for button in row
    ]
    assert "🎛️ Panel" in labels
    assert "📁 Projects" in labels
    assert "🧵 Jobs" in labels
    assert "🩺 Doctor" in labels
    assert "Alice" in call_kwargs.args[0]


async def test_agentic_new_resets_session(agentic_settings, deps):
    """Agentic /new clears session and sends brief confirmation."""
    orchestrator = MessageOrchestrator(agentic_settings, deps)

    update = MagicMock()
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.user_data = {"claude_session_id": "old-session-123"}

    await orchestrator.agentic_new(update, context)

    assert context.user_data["claude_session_id"] is None
    update.message.reply_text.assert_called_once()
    assert update.message.reply_text.call_args.args[0] == "Session reset. What's next?"
    assert update.message.reply_text.call_args.kwargs["reply_markup"] is not None


async def test_agentic_status_compact(agentic_settings, deps):
    """Agentic /status returns rich status text with control buttons."""
    orchestrator = MessageOrchestrator(agentic_settings, deps)

    update = MagicMock()
    update.effective_user.id = 123
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.user_data = {}
    context.bot_data = {"rate_limiter": None}

    await orchestrator.agentic_status(update, context)

    call_args = update.message.reply_text.call_args
    text = call_args.args[0]
    assert "Agentic Status" in text
    assert "Session" in text
    assert call_args.kwargs["reply_markup"] is not None


async def test_agentic_text_calls_claude(agentic_settings, deps):
    """Agentic text handler calls Claude and returns response without keyboard."""
    orchestrator = MessageOrchestrator(agentic_settings, deps)

    # Mock Claude response
    mock_response = MagicMock()
    mock_response.session_id = "session-abc"
    mock_response.content = "Hello, I can help with that!"
    mock_response.tools_used = []

    claude_integration = AsyncMock()
    claude_integration.run_command = AsyncMock(return_value=mock_response)

    update = MagicMock()
    update.effective_user.id = 123
    update.message.text = "Help me with this code"
    update.message.message_id = 1
    update.message.chat.send_action = AsyncMock()
    update.message.reply_text = AsyncMock()

    # Progress message mock
    progress_msg = AsyncMock()
    progress_msg.delete = AsyncMock()
    update.message.reply_text.return_value = progress_msg

    context = MagicMock()
    context.user_data = {}
    context.bot_data = {
        "settings": agentic_settings,
        "claude_integration": claude_integration,
        "storage": None,
        "rate_limiter": None,
        "audit_logger": None,
    }

    await orchestrator.agentic_text(update, context)

    # Claude was called
    claude_integration.run_command.assert_called_once()

    # Session ID updated
    assert context.user_data["claude_session_id"] == "session-abc"

    # Progress message deleted
    progress_msg.delete.assert_called_once()

    # Response sent without keyboard (reply_markup=None)
    response_calls = [
        c
        for c in update.message.reply_text.call_args_list
        if c != update.message.reply_text.call_args_list[0]
    ]
    for call in response_calls:
        assert call.kwargs.get("reply_markup") is None


async def test_agentic_text_uses_autopilot_workspace_and_prompt(agentic_settings, tmp_dir):
    """Agentic text should route through autopilot when project automation is available."""
    orchestrator = MessageOrchestrator(agentic_settings, {})

    workspace_root = tmp_dir / "project"
    workspace_root.mkdir()
    nested_dir = workspace_root / "src"
    nested_dir.mkdir()

    mock_response = MagicMock()
    mock_response.session_id = "session-xyz"
    mock_response.content = "Done."
    mock_response.tools_used = []

    claude_integration = AsyncMock()
    claude_integration.run_command = AsyncMock(return_value=mock_response)

    autopilot_plan = SimpleNamespace(
        prompt="AUTOPILOT PROMPT",
        workspace_root=workspace_root,
        workspace_changed=False,
        profile=SimpleNamespace(),
        matched_playbook=None,
        should_checkpoint=False,
        should_verify=False,
    )
    project_automation = MagicMock()
    project_automation.build_automation_plan.return_value = autopilot_plan
    change_guard = MagicMock()
    features = SimpleNamespace(
        get_project_automation=lambda: project_automation,
        get_project_change_guard=lambda: change_guard,
    )

    update = MagicMock()
    update.effective_user.id = 123
    update.message.text = "Fix this bug"
    update.message.message_id = 1
    update.message.chat.type = "private"
    update.message.chat.send_action = AsyncMock()
    update.message.reply_text = AsyncMock()

    progress_msg = AsyncMock()
    progress_msg.delete = AsyncMock()
    update.message.reply_text.return_value = progress_msg

    context = MagicMock()
    context.user_data = {
        "current_directory": nested_dir,
        "claude_session_id": "session-existing",
    }
    context.bot_data = {
        "settings": agentic_settings,
        "claude_integration": claude_integration,
        "storage": None,
        "rate_limiter": None,
        "audit_logger": None,
        "features": features,
    }

    await orchestrator.agentic_text(update, context)

    claude_integration.run_command.assert_called_once()
    kwargs = claude_integration.run_command.call_args.kwargs
    assert kwargs["prompt"] == "AUTOPILOT PROMPT"
    assert kwargs["working_directory"] == workspace_root
    assert kwargs["session_id"] == "session-existing"
    assert context.user_data["current_directory"] == nested_dir


async def test_agentic_text_switches_workspace_and_resumes_session(
    agentic_settings, tmp_dir
):
    """Autopilot should persist the new workspace and resume its session."""
    orchestrator = MessageOrchestrator(agentic_settings, {})

    current_root = tmp_dir / "ClaudeBot"
    current_root.mkdir()
    target_root = tmp_dir / "FreelanceAggregator"
    target_root.mkdir()

    mock_response = MagicMock()
    mock_response.session_id = "session-new"
    mock_response.content = "Done."
    mock_response.tools_used = []

    claude_integration = AsyncMock()
    claude_integration.run_command = AsyncMock(return_value=mock_response)
    claude_integration._find_resumable_session = AsyncMock(
        return_value=SimpleNamespace(session_id="session-resumed")
    )

    autopilot_plan = SimpleNamespace(
        prompt="AUTOPILOT PROMPT",
        workspace_root=target_root,
        workspace_changed=True,
        profile=SimpleNamespace(),
        matched_playbook="test",
        should_checkpoint=False,
        should_verify=False,
        read_only=False,
    )
    project_automation = MagicMock()
    project_automation.build_automation_plan.return_value = autopilot_plan
    change_guard = MagicMock()
    features = SimpleNamespace(
        get_project_automation=lambda: project_automation,
        get_project_change_guard=lambda: change_guard,
    )

    update = MagicMock()
    update.effective_user.id = 123
    update.message.text = "почини тесты в FreelanceAggregator"
    update.message.message_id = 1
    update.message.chat.type = "private"
    update.message.chat.send_action = AsyncMock()
    update.message.reply_text = AsyncMock()

    progress_msg = AsyncMock()
    progress_msg.delete = AsyncMock()
    update.message.reply_text.return_value = progress_msg

    context = MagicMock()
    context.user_data = {
        "current_directory": current_root,
        "claude_session_id": "old-session",
    }
    context.bot_data = {
        "settings": agentic_settings,
        "claude_integration": claude_integration,
        "storage": None,
        "rate_limiter": None,
        "audit_logger": None,
        "features": features,
    }

    await orchestrator.agentic_text(update, context)

    claude_integration._find_resumable_session.assert_awaited_once_with(123, target_root)
    kwargs = claude_integration.run_command.call_args.kwargs
    assert kwargs["working_directory"] == target_root
    assert kwargs["session_id"] == "session-resumed"
    assert context.user_data["current_directory"] == target_root


async def test_agentic_text_keeps_autopilot_prompt_when_custom_instructions_present(
    agentic_settings, tmp_dir
):
    """Custom instructions should wrap the autopilot prompt instead of replacing it."""
    orchestrator = MessageOrchestrator(agentic_settings, {})

    workspace_root = tmp_dir / "ClaudeBot"
    workspace_root.mkdir()

    mock_response = MagicMock()
    mock_response.session_id = "session-xyz"
    mock_response.content = "Done."
    mock_response.tools_used = []

    claude_integration = AsyncMock()
    claude_integration.run_command = AsyncMock(return_value=mock_response)

    autopilot_plan = SimpleNamespace(
        prompt="AUTOPILOT PROMPT",
        workspace_root=workspace_root,
        workspace_changed=False,
        profile=SimpleNamespace(),
        matched_playbook=None,
        should_checkpoint=False,
        should_verify=False,
    )
    project_automation = MagicMock()
    project_automation.build_automation_plan.return_value = autopilot_plan
    features = SimpleNamespace(
        get_project_automation=lambda: project_automation,
        get_project_change_guard=lambda: MagicMock(),
    )

    update = MagicMock()
    update.effective_user.id = 123
    update.message.text = "Fix this bug"
    update.message.message_id = 1
    update.message.chat.type = "private"
    update.message.chat.send_action = AsyncMock()
    update.message.reply_text = AsyncMock()

    progress_msg = AsyncMock()
    progress_msg.delete = AsyncMock()
    update.message.reply_text.return_value = progress_msg

    context = MagicMock()
    context.user_data = {
        "current_directory": workspace_root,
        "custom_instructions": ["Always explain the risk first"],
    }
    context.bot_data = {
        "settings": agentic_settings,
        "claude_integration": claude_integration,
        "storage": None,
        "rate_limiter": None,
        "audit_logger": None,
        "features": features,
    }

    await orchestrator.agentic_text(update, context)

    prompt = claude_integration.run_command.call_args.kwargs["prompt"]
    assert "Always explain the risk first" in prompt
    assert "AUTOPILOT PROMPT" in prompt
    assert "Fix this bug" not in prompt.split("\n\n", 1)[1]


async def test_agentic_callback_scoped_to_cd_pattern(agentic_settings, deps):
    """Agentic callback handler is registered with cd: pattern filter."""
    orchestrator = MessageOrchestrator(agentic_settings, deps)
    app = MagicMock()
    app.add_handler = MagicMock()

    orchestrator.register_handlers(app)

    from telegram.ext import CallbackQueryHandler

    cb_handlers = [
        call[0][0]
        for call in app.add_handler.call_args_list
        if isinstance(call[0][0], CallbackQueryHandler)
    ]

    assert len(cb_handlers) == 2
    # Should have act: and cd: callback handlers
    patterns = [h.pattern for h in cb_handlers if h.pattern]
    assert any(p.match("cd:my_project") for p in patterns)
    assert any(p.match("act:new") for p in patterns)


async def test_agentic_repo_lists_workspace_catalog(agentic_settings, tmp_dir):
    """Agentic /repo should render discovered workspaces, including nested ones."""
    orchestrator = MessageOrchestrator(agentic_settings, {})

    claude_root = tmp_dir / "ClaudeBot"
    claude_root.mkdir()
    (claude_root / ".git").mkdir()
    (claude_root / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths=['tests']\n",
        encoding="utf-8",
    )

    nested_root = tmp_dir / "MacProjects" / "Poolych"
    nested_root.mkdir(parents=True)
    (nested_root / ".git").mkdir()
    (nested_root / "package.json").write_text(
        '{"name":"poolych","scripts":{"test":"vitest run"}}',
        encoding="utf-8",
    )

    update = MagicMock()
    update.message.text = "/repo"
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.user_data = {"current_directory": claude_root}
    context.bot_data = {
        "features": SimpleNamespace(
            get_project_automation=lambda: ProjectAutomationManager()
        )
    }

    await orchestrator.agentic_repo(update, context)

    message = update.message.reply_text.call_args[0][0]
    assert "Workspaces" in message
    assert "ClaudeBot" in message
    assert "MacProjects/Poolych" in message


async def test_agentic_repo_switches_by_workspace_name(agentic_settings, tmp_dir):
    """Agentic /repo <name> should resolve discovered workspaces by name."""
    orchestrator = MessageOrchestrator(agentic_settings, {})

    current_root = tmp_dir / "ClaudeBot"
    current_root.mkdir()
    target_root = tmp_dir / "MacProjects" / "Poolych"
    target_root.mkdir(parents=True)
    (target_root / ".git").mkdir()

    claude_integration = AsyncMock()
    claude_integration._find_resumable_session = AsyncMock(
        return_value=SimpleNamespace(session_id="session-resumed")
    )

    update = MagicMock()
    update.effective_user.id = 123
    update.message.text = "/repo Poolych"
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.user_data = {"current_directory": current_root}
    context.bot_data = {
        "claude_integration": claude_integration,
        "features": SimpleNamespace(
            get_project_automation=lambda: ProjectAutomationManager()
        ),
    }

    await orchestrator.agentic_repo(update, context)

    resolved_target = target_root.resolve()
    claude_integration._find_resumable_session.assert_awaited_once_with(
        123, resolved_target
    )
    assert context.user_data["current_directory"] == resolved_target
    assert context.user_data["claude_session_id"] == "session-resumed"
    assert update.message.reply_text.call_args.kwargs["reply_markup"] is not None


async def test_agentic_quick_action_panel_renders_control_panel(
    agentic_settings, tmp_dir
):
    """The panel quick action should render the control panel with buttons."""
    orchestrator = MessageOrchestrator(agentic_settings, {})

    workspace_root = tmp_dir / "ClaudeBot"
    workspace_root.mkdir()
    (workspace_root / ".git").mkdir()
    (workspace_root / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths=['tests']\n",
        encoding="utf-8",
    )

    query = MagicMock()
    query.data = "act:panel"
    query.from_user.id = 123
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()

    update = MagicMock()
    update.callback_query = query

    context = MagicMock()
    context.user_data = {"current_directory": workspace_root}
    context.bot_data = {
        "features": SimpleNamespace(
            get_project_automation=lambda: ProjectAutomationManager()
        ),
        "claude_integration": None,
    }

    await orchestrator._agentic_quick_action(update, context)

    query.edit_message_text.assert_awaited_once()
    text = query.edit_message_text.call_args.args[0]
    assert "Control Panel" in text
    assert query.edit_message_text.call_args.kwargs["reply_markup"] is not None


async def test_agentic_quick_action_health_runs_workspace_command(
    agentic_settings, tmp_dir
):
    """Health quick action should execute the configured workspace health command."""
    orchestrator = MessageOrchestrator(agentic_settings, {})

    workspace_root = tmp_dir / "FreelanceAggregator"
    workspace_root.mkdir()
    for filename in ("main.py", "bot.py", "web.py"):
        (workspace_root / filename).write_text("print('ok')\n", encoding="utf-8")
    (workspace_root / "requirements.txt").write_text(
        "fastapi==0.115.0\n",
        encoding="utf-8",
    )

    profiles_path = tmp_dir / "workspace_profiles.yaml"
    profiles_path.write_text(
        """
workspaces:
  - path: FreelanceAggregator
    name: FreelanceAggregator
    commands:
      health: python3 -m py_compile main.py bot.py web.py
        """.strip(),
        encoding="utf-8",
    )
    manager = ProjectAutomationManager(workspace_profiles_path=profiles_path)

    status_msg = AsyncMock()
    status_msg.edit_text = AsyncMock()

    query = MagicMock()
    query.data = "act:health"
    query.from_user.id = 123
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    query.message.reply_text = AsyncMock(return_value=status_msg)

    update = MagicMock()
    update.callback_query = query

    context = MagicMock()
    context.user_data = {"current_directory": workspace_root}
    context.bot_data = {
        "features": SimpleNamespace(get_project_automation=lambda: manager),
        "audit_logger": None,
    }

    await orchestrator._agentic_quick_action(update, context)

    query.message.reply_text.assert_awaited_once()
    status_msg.edit_text.assert_awaited_once()
    result_text = status_msg.edit_text.call_args.args[0]
    assert "Health Check" in result_text
    assert "Exit code: <code>0</code>" in result_text


async def test_agentic_quick_action_start_launches_background_job(
    agentic_settings, tmp_dir
):
    """Start quick action should launch a persistent background job."""
    orchestrator = MessageOrchestrator(agentic_settings, {})

    workspace_root = tmp_dir / "FreelanceAggregator"
    workspace_root.mkdir()
    (workspace_root / "requirements.txt").write_text(
        "fastapi==0.115.0\n",
        encoding="utf-8",
    )

    profiles_path = tmp_dir / "workspace_profiles.yaml"
    profiles_path.write_text(
        """
workspaces:
  - path: FreelanceAggregator
    name: FreelanceAggregator
    commands:
      start: python3 -c "import time; print('boot'); time.sleep(10)"
      health: python3 -c "print('healthy')"
        """.strip(),
        encoding="utf-8",
    )
    manager = ProjectAutomationManager(workspace_profiles_path=profiles_path)
    operator_runtime = WorkspaceOperatorRuntime(tmp_dir / "operator_runtime")

    query = MagicMock()
    query.data = "act:start"
    query.from_user.id = 123
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()

    update = MagicMock()
    update.callback_query = query

    context = MagicMock()
    context.user_data = {"current_directory": workspace_root}
    context.bot_data = {
        "features": SimpleNamespace(
            get_project_automation=lambda: manager,
            get_workspace_operator=lambda: operator_runtime,
        ),
        "audit_logger": None,
    }

    await orchestrator._agentic_quick_action(update, context)

    text = query.edit_message_text.call_args.args[0]
    assert "Background Job Started" in text
    assert "Health verify" in text
    assert query.edit_message_text.call_args.kwargs["reply_markup"] is not None

    job = operator_runtime.get_latest_job(workspace_root)
    assert job is not None
    assert job.action_key == "start"
    assert job.is_active is True
    assert job.verification_command == 'python3 -c "print(\'healthy\')"'
    assert job.verification_mode == "while_running"

    stopped = await operator_runtime.stop_job(job.job_id)
    if stopped.is_active:
        await asyncio.sleep(0.3)


def test_format_agentic_job_status_includes_health_state(agentic_settings, deps, tmp_dir):
    """Compact job status should include health verification state when available."""
    orchestrator = MessageOrchestrator(agentic_settings, deps)

    workspace_root = tmp_dir / "FreelanceAggregator"
    workspace_root.mkdir()
    job = SimpleNamespace(
        workspace_root=workspace_root,
        status="running",
        action_key="start",
        verification_command="python3 -c \"print('healthy')\"",
        verification_status="running",
        job_id="abcdef123456",
    )

    text = orchestrator._format_agentic_job_status(job, tmp_dir)

    assert "running start" in text
    assert "health checking" in text
    assert "abcdef12" in text


async def test_agentic_callback_switches_nested_workspace_and_keeps_markup(
    agentic_settings, tmp_dir
):
    """Workspace callback should support nested paths and keep control buttons."""
    orchestrator = MessageOrchestrator(agentic_settings, {})

    current_root = tmp_dir / "ClaudeBot"
    current_root.mkdir()
    target_root = tmp_dir / "MacProjects" / "Poolych"
    target_root.mkdir(parents=True)
    (target_root / ".git").mkdir()
    (target_root / "package.json").write_text(
        '{"name":"poolych","scripts":{"test":"vitest run"}}',
        encoding="utf-8",
    )

    claude_integration = AsyncMock()
    claude_integration._find_resumable_session = AsyncMock(
        return_value=SimpleNamespace(session_id="session-resumed")
    )

    query = MagicMock()
    query.data = "cd:MacProjects/Poolych"
    query.from_user.id = 123
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()

    update = MagicMock()
    update.callback_query = query

    context = MagicMock()
    context.user_data = {"current_directory": current_root}
    context.bot_data = {
        "claude_integration": claude_integration,
        "features": SimpleNamespace(
            get_project_automation=lambda: ProjectAutomationManager()
        ),
        "audit_logger": None,
    }

    await orchestrator._agentic_callback(update, context)

    resolved_target = target_root.resolve()
    claude_integration._find_resumable_session.assert_awaited_once_with(
        123, resolved_target
    )
    assert context.user_data["current_directory"] == resolved_target
    assert context.user_data["claude_session_id"] == "session-resumed"
    assert query.edit_message_text.call_args.kwargs["reply_markup"] is not None


async def test_agentic_document_rejects_large_files(agentic_settings, deps):
    """Agentic document handler rejects files over 10MB."""
    orchestrator = MessageOrchestrator(agentic_settings, deps)

    update = MagicMock()
    update.effective_user.id = 123
    update.message.document.file_name = "big.bin"
    update.message.document.file_size = 20 * 1024 * 1024  # 20MB
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.bot_data = {"security_validator": None}

    await orchestrator.agentic_document(update, context)

    call_args = update.message.reply_text.call_args
    assert "too large" in call_args.args[0].lower()


async def test_agentic_voice_calls_claude(agentic_settings, deps):
    """Agentic voice handler transcribes and routes prompt to Claude."""
    orchestrator = MessageOrchestrator(agentic_settings, deps)

    mock_response = MagicMock()
    mock_response.session_id = "voice-session-123"
    mock_response.content = "Voice response from Claude"
    mock_response.tools_used = []

    claude_integration = AsyncMock()
    claude_integration.run_command = AsyncMock(return_value=mock_response)

    processed_voice = MagicMock()
    processed_voice.prompt = "Voice prompt text"

    voice_handler = MagicMock()
    voice_handler.process_voice_message = AsyncMock(return_value=processed_voice)

    features = MagicMock()
    features.get_voice_handler.return_value = voice_handler

    update = MagicMock()
    update.effective_user.id = 123
    update.message.voice = MagicMock()
    update.message.caption = "please summarize"
    update.message.message_id = 1
    update.message.chat.send_action = AsyncMock()
    update.message.reply_text = AsyncMock()

    progress_msg = AsyncMock()
    progress_msg.edit_text = AsyncMock()
    progress_msg.delete = AsyncMock()
    update.message.reply_text.return_value = progress_msg

    context = MagicMock()
    context.user_data = {}
    context.bot_data = {
        "settings": agentic_settings,
        "features": features,
        "claude_integration": claude_integration,
    }

    await orchestrator.agentic_voice(update, context)

    voice_handler.process_voice_message.assert_awaited_once_with(
        update.message.voice, "please summarize"
    )
    claude_integration.run_command.assert_awaited_once()
    assert context.user_data["claude_session_id"] == "voice-session-123"


async def test_agentic_voice_missing_handler_is_provider_aware(tmp_path, deps):
    """Missing voice handler guidance references the configured provider key."""
    settings = create_test_config(
        approved_directory=str(tmp_path),
        agentic_mode=True,
        voice_provider="openai",
    )
    orchestrator = MessageOrchestrator(settings, deps)

    features = MagicMock()
    features.get_voice_handler.return_value = None

    update = MagicMock()
    update.effective_user.id = 123
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.bot_data = {"features": features}
    context.user_data = {}

    await orchestrator.agentic_voice(update, context)

    call_args = update.message.reply_text.call_args
    assert "OPENAI_API_KEY" in call_args.args[0]


async def test_agentic_voice_transcription_failure_surfaces_user_error(
    agentic_settings, deps
):
    """Transcription failures are shown to users and do not call Claude."""
    orchestrator = MessageOrchestrator(agentic_settings, deps)

    voice_handler = MagicMock()
    voice_handler.process_voice_message = AsyncMock(
        side_effect=RuntimeError("Mistral transcription request failed: boom")
    )

    features = MagicMock()
    features.get_voice_handler.return_value = voice_handler

    claude_integration = AsyncMock()
    claude_integration.run_command = AsyncMock()

    update = MagicMock()
    update.effective_user.id = 123
    update.message.voice = MagicMock()
    update.message.caption = None
    update.message.chat.send_action = AsyncMock()
    update.message.reply_text = AsyncMock()

    progress_msg = AsyncMock()
    progress_msg.edit_text = AsyncMock()
    update.message.reply_text.return_value = progress_msg

    context = MagicMock()
    context.user_data = {}
    context.bot_data = {
        "settings": agentic_settings,
        "features": features,
        "claude_integration": claude_integration,
    }

    await orchestrator.agentic_voice(update, context)

    progress_msg.edit_text.assert_awaited_once()
    error_text = progress_msg.edit_text.call_args.args[0]
    assert "Mistral transcription request failed" in error_text
    assert progress_msg.edit_text.call_args.kwargs["parse_mode"] == "HTML"
    claude_integration.run_command.assert_not_awaited()


async def test_agentic_start_escapes_html_in_name(agentic_settings, deps):
    """Names with HTML-special characters are escaped safely."""
    orchestrator = MessageOrchestrator(agentic_settings, deps)

    update = MagicMock()
    update.effective_user.first_name = "A<B>&C"
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.user_data = {}

    await orchestrator.agentic_start(update, context)

    call_kwargs = update.message.reply_text.call_args
    text = call_kwargs.args[0]
    # HTML-special characters should be escaped
    assert "A&lt;B&gt;&amp;C" in text
    # parse_mode is HTML
    assert call_kwargs.kwargs.get("parse_mode") == "HTML"


async def test_agentic_text_logs_failure_on_error(agentic_settings, deps):
    """Failed Claude runs are logged with success=False."""
    orchestrator = MessageOrchestrator(agentic_settings, deps)

    claude_integration = AsyncMock()
    claude_integration.run_command = AsyncMock(side_effect=Exception("Claude broke"))

    audit_logger = AsyncMock()
    audit_logger.log_command = AsyncMock()

    update = MagicMock()
    update.effective_user.id = 123
    update.message.text = "do something"
    update.message.message_id = 1
    update.message.chat.send_action = AsyncMock()
    update.message.reply_text = AsyncMock()

    progress_msg = AsyncMock()
    progress_msg.delete = AsyncMock()
    update.message.reply_text.return_value = progress_msg

    context = MagicMock()
    context.user_data = {}
    context.bot_data = {
        "settings": agentic_settings,
        "claude_integration": claude_integration,
        "storage": None,
        "rate_limiter": None,
        "audit_logger": audit_logger,
    }

    await orchestrator.agentic_text(update, context)

    # Audit logged with success=False
    audit_logger.log_command.assert_called_once()
    call_kwargs = audit_logger.log_command.call_args
    assert call_kwargs.kwargs["success"] is False


# --- _redact_secrets / _summarize_tool_input tests ---


class TestRedactSecrets:
    """Ensure sensitive substrings are redacted from Bash command summaries."""

    def test_safe_command_unchanged(self):
        assert (
            _redact_secrets("poetry run pytest tests/ -v")
            == "poetry run pytest tests/ -v"
        )

    def test_anthropic_api_key_redacted(self):
        key = "sk-ant-api03-abc123def456ghi789jkl012mno345"
        cmd = f"ANTHROPIC_API_KEY={key}"
        result = _redact_secrets(cmd)
        assert key not in result
        assert "***" in result

    def test_sk_key_redacted(self):
        cmd = "curl -H 'Authorization: Bearer sk-1234567890abcdefghijklmnop'"
        result = _redact_secrets(cmd)
        assert "sk-1234567890abcdefghijklmnop" not in result
        assert "***" in result

    def test_github_pat_redacted(self):
        cmd = "git clone https://ghp_abcdefghijklmnop1234@github.com/user/repo"
        result = _redact_secrets(cmd)
        assert "ghp_abcdefghijklmnop1234" not in result
        assert "***" in result

    def test_aws_key_redacted(self):
        cmd = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
        result = _redact_secrets(cmd)
        assert "AKIAIOSFODNN7EXAMPLE" not in result
        assert "***" in result

    def test_flag_token_redacted(self):
        cmd = "mycli --token=supersecretvalue123"
        result = _redact_secrets(cmd)
        assert "supersecretvalue123" not in result
        assert "--token=" in result or "--token" in result

    def test_password_env_redacted(self):
        cmd = "PASSWORD=MyS3cretP@ss! ./run.sh"
        result = _redact_secrets(cmd)
        assert "MyS3cretP@ss!" not in result
        assert "***" in result

    def test_bearer_token_redacted(self):
        cmd = "curl -H 'Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig'"
        result = _redact_secrets(cmd)
        assert "eyJhbGciOiJIUzI1NiJ9.payload.sig" not in result

    def test_connection_string_redacted(self):
        cmd = "psql postgresql://admin:secret_password@db.host:5432/mydb"
        result = _redact_secrets(cmd)
        assert "secret_password" not in result

    def test_summarize_tool_input_bash_redacts(self, agentic_settings, deps):
        """_summarize_tool_input applies redaction to Bash commands."""
        orchestrator = MessageOrchestrator(agentic_settings, deps)
        result = orchestrator._summarize_tool_input(
            "Bash",
            {"command": "curl --token=mysupersecrettoken123 https://api.example.com"},
        )
        assert "mysupersecrettoken123" not in result
        assert "***" in result

    def test_summarize_tool_input_non_bash_unchanged(self, agentic_settings, deps):
        """Non-Bash tools don't go through redaction."""
        orchestrator = MessageOrchestrator(agentic_settings, deps)
        result = orchestrator._summarize_tool_input(
            "Read", {"file_path": "/home/user/.env"}
        )
        assert result == ".env"


# --- Typing heartbeat tests ---


class TestTypingHeartbeat:
    """Verify typing indicator stays alive independently of stream events."""

    async def test_heartbeat_sends_typing_action(self, agentic_settings, deps):
        """Heartbeat sends typing actions at the configured interval."""
        chat = AsyncMock()
        chat.send_action = AsyncMock()

        orchestrator = MessageOrchestrator(agentic_settings, deps)
        heartbeat = orchestrator._start_typing_heartbeat(chat, interval=0.05)

        # Let the heartbeat fire a few times
        await asyncio.sleep(0.2)
        heartbeat.cancel()
        try:
            await heartbeat
        except asyncio.CancelledError:
            pass

        # Should have been called multiple times
        assert chat.send_action.call_count >= 2
        chat.send_action.assert_called_with("typing")

    async def test_heartbeat_cancels_cleanly(self, agentic_settings, deps):
        """Cancelling the heartbeat task does not raise."""
        chat = AsyncMock()
        orchestrator = MessageOrchestrator(agentic_settings, deps)
        heartbeat = orchestrator._start_typing_heartbeat(chat, interval=0.05)

        heartbeat.cancel()
        # Should not raise
        try:
            await heartbeat
        except asyncio.CancelledError:
            pass

        assert heartbeat.cancelled() or heartbeat.done()

    async def test_heartbeat_survives_send_action_errors(self, agentic_settings, deps):
        """Heartbeat keeps running even if send_action raises."""
        chat = AsyncMock()
        call_count = [0]

        async def flaky_send_action(action: str) -> None:
            call_count[0] += 1
            if call_count[0] <= 2:
                raise Exception("Network error")

        chat.send_action = flaky_send_action

        orchestrator = MessageOrchestrator(agentic_settings, deps)
        heartbeat = orchestrator._start_typing_heartbeat(chat, interval=0.05)

        await asyncio.sleep(0.3)
        heartbeat.cancel()
        try:
            await heartbeat
        except asyncio.CancelledError:
            pass

        # Should have called send_action more than 2 times (survived errors)
        assert call_count[0] >= 3

    async def test_stream_callback_independent_of_typing(self, agentic_settings, deps):
        """Stream callback no longer sends typing — that's the heartbeat's job."""
        orchestrator = MessageOrchestrator(agentic_settings, deps)

        progress_msg = AsyncMock()
        tool_log: list = []  # type: ignore[type-arg]
        callback = orchestrator._make_stream_callback(
            verbose_level=1,
            progress_msg=progress_msg,
            tool_log=tool_log,
            start_time=0.0,
        )
        assert callback is not None

        # Verify the callback signature doesn't accept a 'chat' parameter
        # (typing is no longer handled by the stream callback)
        import inspect

        sig = inspect.signature(orchestrator._make_stream_callback)
        assert "chat" not in sig.parameters


async def test_group_thread_mode_rejects_non_forum_chat(group_thread_settings, deps):
    """Strict thread mode rejects updates outside configured forum chat."""
    orchestrator = MessageOrchestrator(group_thread_settings, deps)

    project_threads_manager = MagicMock()
    project_threads_manager.guidance_message.return_value = "Use project thread"
    deps["project_threads_manager"] = project_threads_manager

    called = {"value": False}

    async def dummy_handler(update, context):
        called["value"] = True

    wrapped = orchestrator._inject_deps(dummy_handler)

    update = MagicMock()
    update.effective_chat.id = -1002222222
    update.effective_message.reply_text = AsyncMock()
    update.callback_query = None

    context = MagicMock()
    context.bot_data = {}
    context.user_data = {}

    await wrapped(update, context)

    assert called["value"] is False
    update.effective_message.reply_text.assert_called_once()


async def test_thread_mode_loads_and_persists_thread_state(group_thread_settings, deps):
    """Thread mode loads per-thread context and writes updates back."""
    orchestrator = MessageOrchestrator(group_thread_settings, deps)

    project_path = group_thread_settings.approved_directory / "project_a"
    project = SimpleNamespace(
        slug="project_a",
        name="Project A",
        absolute_path=project_path,
    )

    project_threads_manager = MagicMock()
    project_threads_manager.resolve_project = AsyncMock(return_value=project)
    project_threads_manager.guidance_message.return_value = "Use project thread"
    deps["project_threads_manager"] = project_threads_manager

    async def dummy_handler(update, context):
        assert context.user_data["claude_session_id"] == "old-session"
        context.user_data["claude_session_id"] = "new-session"

    wrapped = orchestrator._inject_deps(dummy_handler)

    update = MagicMock()
    update.effective_chat.id = -1001234567890
    update.effective_message.message_thread_id = 777
    update.effective_message.reply_text = AsyncMock()
    update.callback_query = None

    context = MagicMock()
    context.bot_data = {}
    context.user_data = {
        "thread_state": {
            "-1001234567890:777": {
                "current_directory": str(project_path),
                "claude_session_id": "old-session",
            }
        }
    }

    await wrapped(update, context)

    assert (
        context.user_data["thread_state"]["-1001234567890:777"]["claude_session_id"]
        == "new-session"
    )


async def test_sync_threads_bypasses_thread_gate(group_thread_settings, deps):
    """sync_threads command bypasses strict thread routing gate."""
    orchestrator = MessageOrchestrator(group_thread_settings, deps)

    called = {"value": False}

    async def sync_threads(update, context):
        called["value"] = True

    project_threads_manager = MagicMock()
    project_threads_manager.guidance_message.return_value = "Use project thread"
    deps["project_threads_manager"] = project_threads_manager

    wrapped = orchestrator._inject_deps(sync_threads)

    update = MagicMock()
    update.effective_chat.id = -1002222222
    update.effective_message.reply_text = AsyncMock()
    update.callback_query = None

    context = MagicMock()
    context.bot_data = {}
    context.user_data = {}

    await wrapped(update, context)

    assert called["value"] is True


async def test_private_mode_start_bypasses_thread_gate(private_thread_settings, deps):
    """Private mode allows /start outside topics."""
    orchestrator = MessageOrchestrator(private_thread_settings, deps)
    called = {"value": False}

    async def start_command(update, context):
        called["value"] = True

    project_threads_manager = MagicMock()
    project_threads_manager.guidance_message.return_value = "Use project topic"
    deps["project_threads_manager"] = project_threads_manager

    wrapped = orchestrator._inject_deps(start_command)

    update = MagicMock()
    update.effective_chat.type = "private"
    update.effective_chat.id = 12345
    update.effective_chat.is_forum = False
    update.effective_message.reply_text = AsyncMock()
    update.callback_query = None

    context = MagicMock()
    context.bot_data = {}
    context.user_data = {}

    await wrapped(update, context)

    assert called["value"] is True
    project_threads_manager.resolve_project.assert_not_called()


async def test_private_mode_start_inside_topic_uses_thread_context(
    private_thread_settings, deps
):
    """/start in private topic should load mapped thread context."""
    orchestrator = MessageOrchestrator(private_thread_settings, deps)
    project_path = private_thread_settings.approved_directory / "project_a"
    project = SimpleNamespace(
        slug="project_a",
        name="Project A",
        absolute_path=project_path,
    )
    project_threads_manager = MagicMock()
    project_threads_manager.resolve_project = AsyncMock(return_value=project)
    project_threads_manager.guidance_message.return_value = "Use project topic"
    deps["project_threads_manager"] = project_threads_manager

    captured = {"dir": None}

    async def start_command(update, context):
        captured["dir"] = context.user_data.get("current_directory")

    wrapped = orchestrator._inject_deps(start_command)

    update = MagicMock()
    update.effective_chat.type = "private"
    update.effective_chat.id = 12345
    update.effective_message.message_thread_id = 777
    update.effective_message.reply_text = AsyncMock()
    update.callback_query = None

    context = MagicMock()
    context.bot_data = {}
    context.user_data = {
        "thread_state": {
            "12345:777": {
                "current_directory": str(project_path),
                "claude_session_id": "old",
            }
        }
    }

    await wrapped(update, context)

    project_threads_manager.resolve_project.assert_awaited_once_with(12345, 777)
    assert captured["dir"] == project_path


async def test_private_mode_rejects_help_outside_topics(private_thread_settings, deps):
    """Private mode rejects non-allowed commands outside mapped topics."""
    orchestrator = MessageOrchestrator(private_thread_settings, deps)
    called = {"value": False}

    async def help_command(update, context):
        called["value"] = True

    project_threads_manager = MagicMock()
    project_threads_manager.guidance_message.return_value = "Use project topic"
    deps["project_threads_manager"] = project_threads_manager

    wrapped = orchestrator._inject_deps(help_command)

    update = MagicMock()
    update.effective_chat.type = "private"
    update.effective_chat.id = 12345
    update.effective_chat.is_forum = False
    update.effective_message.message_thread_id = None
    update.effective_message.direct_messages_topic = None
    update.effective_message.reply_text = AsyncMock()
    update.callback_query = None

    context = MagicMock()
    context.bot_data = {}
    context.user_data = {}

    await wrapped(update, context)

    assert called["value"] is False
    update.effective_message.reply_text.assert_called_once()
